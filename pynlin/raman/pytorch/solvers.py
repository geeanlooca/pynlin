import math
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import torch
from scipy.constants import speed_of_light
from numpy import polyval

from pynlin.fiber import Fiber, MMFiber
from pynlin.raman.pytorch._torch_ode import torch_rk4
from pynlin.raman.response import impulse_response
from pynlin.utils import nu2lambda

import matplotlib.pyplot as plt
import seaborn as sns


class RamanAmplifier(torch.nn.Module):
    def __init__(
        self,
        fiber_length: float,
        integration_steps: int,
        num_pumps: int,
        signal_wavelengths: Union[list, NDArray],
        power_per_channel: float,
        fiber: Fiber,
        fs: float = 80e12,
        dF: float = 50e9,
        pump_direction: Union[int, NDArray] = 1,
    ):
        """A PyTorch model of a Multi-Mode Fiber Raman Amplifier.

        Parameters
        ----------
        length : float
          The length of the fiber [m].
        steps : int
          The number of integration steps.
        num_pumps : int
          The number of Raman pumps.
        signal_wavelength : torch.Tensor
          The input signal wavelenghts.
        power_per_channel : float
          The input power for each channel.
        fiber : pyraman.Fiber
          The Fiber object containing relevant parameters of the fiber.
        fs : float, optional
          The sampling frequency of the Raman impulse response, by default 80e12
        dF : float, optional
          The spacing between samples of the Raman frequency response, by default 50e9
        """

        super(RamanAmplifier, self).__init__()
        self.c0 = speed_of_light
        self.power_per_channel = power_per_channel
        self.num_pumps = num_pumps
        self.num_channels = signal_wavelengths.shape[0]
        self.signal_wavelengths = torch.from_numpy(signal_wavelengths)
        self.length = fiber_length
        self.steps = integration_steps
        z = torch.linspace(0, self.length, self.steps)

        if isinstance(signal_wavelengths, np.ndarray):
            signal_wavelengths = torch.from_numpy(signal_wavelengths).float()

        signal_power = self.power_per_channel * torch.ones((1, self.num_channels))

        # limit the polynomial fit of the attenuation spectrum to order 2
        num_loss_coeffs = len(fiber.losses)
        loss_coeffs = np.zeros((3,))

        for i in range(np.minimum(3, num_loss_coeffs)):
            loss_coeffs[i] = fiber.losses[i]

        # Compute the attenuation for each signal
        signal_loss = self._alpha_to_linear(
            loss_coeffs[2]
            + loss_coeffs[1] * (signal_wavelengths)
            + loss_coeffs[0] * (signal_wavelengths) ** 2
        )

        if isinstance(signal_loss, np.ndarray):
            signal_loss = torch.from_numpy(signal_loss)

        # signal_loss = signal_loss.repeat_interleave(self.modes).view(1, -1)

        self.raman_coefficient = fiber.raman_coefficient

        # Compute the raman gain spectrum with the given precision
        self.fs = fs
        self.dF = dF
        N = math.ceil(self.fs / self.dF)
        self.dF = self.fs / N
        resp, _ = impulse_response(None, self.fs, num_samples=N)
        resp -= resp.mean()
        self.effective_area = fiber.effective_area

        # Only save the positive spectrum
        self.response_length = math.ceil((N + 1) / 2)
        self.max_freq = (self.response_length - 1) * self.dF
        spectrum = np.fft.fft(resp)[: self.response_length]
        gain_spectrum = -np.imag(spectrum)

        # Normalize the peak to the raman coefficient of the fiber
        gain_spectrum /= np.max(np.abs(gain_spectrum))
        gain_spectrum *= self.raman_coefficient

        # Transform it to torch.Tensor, reshape it, and register as a buffer
        raman_response = torch.from_numpy(gain_spectrum).float()
        # Reshape the Tensor in a format accepted by grid_sample
        # It works on 4D (batch of multi-channel images) or 5D (volumetric data)
        # So Tensor must be of size (self.batch_size, 1, 1, response_length), meaning
        # we encode the response as a 1-channel, 1 pixel tall image
        raman_response = raman_response.view(1, 1, 1, -1)

        # Register buffers to make it work on a GPU
        self.register_buffer("signal_power", signal_power)
        self.register_buffer("z", z)
        self.register_buffer(
            "signal_frequency", self._lambda2frequency(signal_wavelengths)
        )
        self.register_buffer("loss_coefficients", torch.from_numpy(loss_coeffs).float())
        self.register_buffer("signal_loss", signal_loss)
        self.register_buffer("raman_response", raman_response)

        # # Doesn't matter, the pumps are turned off
        # pump_lambda = torch.linspace(1420, 1480, self.num_pumps) * 1e-9
        # pump_power = torch.zeros((num_pumps * modes))
        # x = torch.cat((pump_lambda, pump_power)).float().view(1, -1)

        if np.isscalar(pump_direction):
            pump_direction = np.ones((self.num_pumps,)) * pump_direction
        else:
            pump_direction = np.atleast_1d(pump_direction)
        signal_direction = np.ones((self.num_channels,))
        direction = np.concatenate((pump_direction, signal_direction))
        self.direction = torch.from_numpy(direction).float()

        # if counterpumping:
        #     self.counterpumping = True
        #     direction[: self.num_pumps * self.modes] = -1
        # else:
        #     self.counterpumping = False

        # Propagate the pumps to compute the output spectrum
        # if counterpumping:
        #     off_gain, _ = self.forward(x)
        #     off_gain = off_gain.view(1, -1)
        # else:
        #     off_gain = self.forward(x).view(1, -1)

        # Save it in a buffer
        # self.register_buffer("off_gain", off_gain)

    def _alpha_to_linear(self, alpha: NDArray) -> NDArray:
        """Convert attenuation constant from dB/m to linear units."""
        return alpha * np.log(10) / 10

    def _lambda2frequency(self, wavelength: NDArray) -> NDArray:
        """Convert wavelength in frequency."""
        return self.c0 / wavelength

    def _batch_diff(self, x: torch.Tensor) -> torch.Tensor:
        """Takes a Tensor of shape (B, N) and returns a Tensor of shape (B, N,
        N) where in position (i, :, :) is the matrix of differences of the
        input vector (i, :) with each one of its elements."""
        batch_size = x.shape[0]
        D = x.view(batch_size, 1, -1) - x.view(batch_size, -1, 1)
        return D

    def _interpolate_response(self, freqs: torch.Tensor) -> torch.Tensor:
        """Compute the Raman gain coefficient for the input frequencies."""
        batch_size = freqs.shape[0]

        # The input for `grid_sample` are defined on a [-1, 1] axis
        # so we need to normalize the input frequencies accordingly
        norm_freqs = 2 * freqs / self.max_freq - 1

        return torch.nn.functional.grid_sample(
            self.raman_response.expand(batch_size, 1, 1, self.response_length),
            norm_freqs,
            align_corners=True,
        )

    @staticmethod
    def ode(
        P: torch.Tensor,
        z: torch.Tensor,
        losses: torch.Tensor,
        gain_matrix: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        """Batched version of the singlemode Raman amplifier equations.

        Params
        ------
        P : torch.Tensor
          Size (B, S). Power of the signal on every mode-wavelength combination
        z : torch.Tensor
          Size (Z,) The evaluation point.
        losses : torch.Tensor.
          Size (B, S)
        gain_matrix : torch.Tensor
          Size (B, S, S)

        Returns
        -------
        torch.Tensor
          Propagated power values
        """
        batch_size = P.shape[0]
        dPdz = (
            (
                -losses.view(batch_size, -1, 1)
                + torch.matmul(gain_matrix, P.view(batch_size, -1, 1))
            )
            * P.view(batch_size, -1, 1)
        ).view(batch_size, -1)

        return dPdz * direction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Solves the propagation equation using a RK4 scheme.

        Parameters
        ----------
        x : torch.Tensor
          pump wavelength and pump power, size (N_batch, 2 * N_pumps)

        Returns
        -------
        torch.Tensor
          Gain on each mode (B, N_signals)
        """

        batch_size = x.shape[0]

        num_freqs = self.num_channels + self.num_pumps

        # This will be the input to the interpolation function
        interpolation_grid = torch.zeros(
            (batch_size, 1, num_freqs**2, 2),
            dtype=x.dtype,
            device=x.device,
        )

        pump_wavelengths = x[:, : self.num_pumps]

        # Compute the loss for each pump wavelength/mode
        pump_loss = self._alpha_to_linear(
            self.loss_coefficients[2]
            + self.loss_coefficients[1] * pump_wavelengths
            + self.loss_coefficients[0] * (pump_wavelengths) ** 2
        )

        # Concatenate the pump losses to the signal losses
        losses = torch.cat(
            (
                pump_loss,
                self.signal_loss.expand(batch_size, self.num_channels),
            ),
            dim=1,
        )

        # Concatenate input pump wavelengths with signal wavelengths in a (B, P + S)
        # Tensor
        pump_freqs = self._lambda2frequency(pump_wavelengths)

        total_freqs = torch.cat(
            (pump_freqs, self.signal_frequency.expand(batch_size, self.num_channels)),
            dim=1,
        )

        # Concatenate input pump power and signal power, making sure power > 0
        total_power = torch.cat(
            (
                torch.nn.functional.relu(torch.abs(x[:, self.num_pumps:])),
                self.signal_power.expand(batch_size, self.num_channels),
            ),
            1,
        )

        # Compute the difference in frequencies
        freqs_diff = self._batch_diff(total_freqs)

        # Collapse matrix of differences in a vector
        interpolation_grid[:, 0, :, 0] = freqs_diff.view(batch_size, -1)

        # Compute the raman gain between the signals and pumps
        # ! Should I precompute the raman gain between signals?
        # ! I could compute it in the __init__ and then expand/cat
        # ! when needed.
        idx = interpolation_grid[:, 0, :, 0] < 0
        gain = self._interpolate_response(torch.abs(interpolation_grid))[:, 0, 0, :]
        gain[idx] *= -1

        # Restore the batched matrices: (B, N * N) -> (B, N, N)
        gain = gain.view(batch_size, num_freqs, num_freqs)
        # diag = torch.diagonal(gain, dim1=-2, dim2=-1).fill_(0)

        # Compute the scaling factor
        one = torch.tensor(1, dtype=x.dtype, device=x.device)
        freq_scaling = torch.max(
            one,
            total_freqs.view(batch_size, -1, 1)
            / (total_freqs.view(batch_size, -1, 1).transpose(1, 2)),
        )

        gain *= freq_scaling

        G = gain / self.effective_area

        solution = torch_rk4(
            RamanAmplifier.ode,
            total_power,
            self.z,
            losses,
            G,
            self.direction,
        ).view(-1, num_freqs)

        signal_spectrum = solution[
            :,
            self.num_pumps:,
        ].clone()

        return signal_spectrum

        # if self.counterpumping:
        #     pump_initial_power = solution[:, : self.num_pumps].clone()
        #     return signal_spectrum, pump_initial_power
        # else:
        #     return signal_spectrum


class MMFRamanAmplifier(torch.nn.Module):
    def __init__(
        self,
        length,
        steps,
        num_pumps,
        signal_wavelength,
        power_per_channel,
        fiber: MMFiber,
        fs=80e12,
        dF=50e9,
        counterpumping=None,
    ):
        """A PyTorch model of a Multi-Mode Fiber Raman Amplifier.

        Parameters
        ----------
        length : float
          The length of the fiber [m].
        steps : int
          The number of integration steps.
        num_pumps : int
          The number of Raman pumps.
        signal_wavelength : torch.Tensor
          The input signal wavelenghts.
        power_per_channel : float
          The input power for each channel (wavelength/mode).
        fiber : pyraman.Fiber
          The Fiber object containing relevant parameters of the fiber.
        fs : float, optional
          The sampling frequency of the Raman impulse response, by default 80e12
        dF : float, optional
          The spacing between samples of the Raman frequency response, by default 50e9
        """

        super(MMFRamanAmplifier, self).__init__()
        self.c0 = speed_of_light
        self.power_per_channel = power_per_channel
        self.num_pumps = num_pumps
        self.num_channels = signal_wavelength.shape[0]
        self.modes = fiber.modes
        self.length = length
        self.steps = steps
        self.fiber = fiber
        self.overlap_integrals = fiber.overlap_integrals[:, :self.modes, :self.modes]
        overlap_integrals_tensor = torch.Tensor(fiber.overlap_integrals).float()

        z = torch.linspace(0, self.length, self.steps)

        if isinstance(signal_wavelength, np.ndarray):
            signal_wavelength = torch.from_numpy(signal_wavelength).float()

        signal_power = self.power_per_channel * torch.ones(
            (1, self.num_channels * self.modes)
        )

        # limit the polynomial fit of the attenuation spectrum to order 2
        num_loss_coeffs = len(fiber.losses)
        loss_coeffs = np.zeros((3,))

        for i in range(np.minimum(3, num_loss_coeffs)):
            loss_coeffs[i] = fiber.losses[i]

        # Compute the attenuation for each signal
        signal_loss = self._alpha_to_linear(polyval(loss_coeffs, signal_wavelength))

        if isinstance(signal_loss, np.ndarray):
            signal_loss = torch.from_numpy(signal_loss)

        signal_loss = signal_loss.repeat_interleave(self.modes).view(1, -1)

        self.raman_coefficient = fiber.raman_coefficient

        # Compute the raman gain spectrum with the given precision
        self.fs = fs
        self.dF = dF
        N = math.ceil(self.fs / self.dF)
        self.dF = self.fs / N
        resp, _ = impulse_response(None, self.fs, num_samples=N)
        resp -= resp.mean()

        # Only save the positive spectrum
        self.response_length = math.ceil((N + 1) / 2)
        self.max_freq = (self.response_length - 1) * self.dF
        spectrum = np.fft.fft(resp)[: self.response_length]
        gain_spectrum = -np.imag(spectrum)

        # Normalize the peak to the raman coefficient of the fiber
        gain_spectrum /= np.max(np.abs(gain_spectrum))
        gain_spectrum *= self.raman_coefficient

        # Transform it to torch.Tensor, reshape it, and register as a buffer
        raman_response = torch.from_numpy(gain_spectrum).float()
        # Reshape the Tensor in a format accepted by grid_sample
        # It works on 4D (batch of multi-channel images) or 5D (volumetric data)
        # So Tensor must be of size (self.batch_size, 1, 1, response_length), meaning
        # we encode the response as a 1-channel, 1 pixel tall image
        raman_response = raman_response.view(1, 1, 1, -1)

        # Register buffers to make it work on a GPU
        self.register_buffer("signal_power", signal_power)
        self.register_buffer("z", z)
        self.register_buffer(
            "signal_frequency", self._lambda2frequency(signal_wavelength)
        )
        self.register_buffer("loss_coefficients", torch.from_numpy(loss_coeffs).float())
        self.register_buffer("signal_loss", signal_loss)
        # self.register_buffer("overlap_integrals", overlap_integrals_tensor)
        self.register_buffer("raman_response", raman_response)

        # Doesn't matter, the pumps are turned off
        pump_lambda = torch.linspace(1420, 1480, self.num_pumps) * 1e-9
        pump_power = torch.zeros((num_pumps * self.modes))
        x = torch.cat((pump_lambda, pump_power)).float().view(1, -1)

        direction = torch.ones(
            ((self.num_pumps + self.num_channels) * self.modes,)
        ).float()

        if counterpumping:
            self.counterpumping = True
            direction[: self.num_pumps * self.modes] = -1
        else:
            self.counterpumping = False

        self.register_buffer("direction", direction)

        # Propagate the pumps to compute the output spectrum
        # if counterpumping:
        #   off_gain, _ = self.forward(x)
        #   off_gain = off_gain.view(1, -1)
        # else:
        #   off_gain = self.forward(x).view(1, -1)

        # # Save it in a buffer
        # self.register_buffer("off_gain", off_gain)

    def _alpha_to_linear(self, alpha):
        """Convert attenuation constant from dB/m to linear units."""
        return alpha * np.log(10) / 10

    def _lambda2frequency(self, wavelength):
        """Convert wavelength in frequency."""
        return self.c0 / wavelength

    def _frequency2lambda(self, frequency):
        """Convert wavelength in frequency."""
        return self.c0 / frequency

    def _batch_diff(self, x):
        """Takes a Tensor of shape (B, N) and returns a Tensor of shape (B, N,
        N) where in position (i, :, :) is the matrix of differences of the
        input vector (i, :) with each one of its elements."""
        batch_size = x.shape[0]
        D = x.view(batch_size, 1, -1) - x.view(batch_size, -1, 1)
        return D

    def _interpolate_response(self, freqs):
        """Compute the Raman gain coefficient for the input frequencies."""
        batch_size = freqs.shape[0]

        # The input for `grid_sample` are defined on a [-1, 1] axis
        # so we need to normalize the input frequencies accordingly
        norm_freqs = 2 * freqs / self.max_freq - 1

        return torch.nn.functional.grid_sample(
            self.raman_response.expand(batch_size, 1, 1, self.response_length),
            norm_freqs,
            align_corners=True,
        )

    @staticmethod
    def ode(P, z, losses, gain_matrix, direction):
        """Batched version of the multimode Raman amplifier equations.

        Params
        ------
        P : torch.Tensor
          Size (B, M * S). Power of the signal on every mode-wavelength combination
        z : torch.Tensor
          Size (Z,) The evaluation point.
        losses : torch.Tensor.
          Size (B, M * S)
        gain_matrix : torch.Tensor
          Size (B, M*S, M*S)

        Returns
        -------
        torch.Tensor
          Propagated power values
        """

        batch_size = P.shape[0]

        dPdz = (
            (
                -losses.view(batch_size, -1, 1)
                + torch.matmul(gain_matrix, P.view(batch_size, -1, 1))
            )
            * P.view(batch_size, -1, 1)
        ).view(batch_size, -1)

        return dPdz * direction

    def forward(self, x):
        """Solves the propagation equation using a RK4 scheme.

        Parameters
        ----------
        x : torch.Tensor
          pump wavelength and pump power, size (N_batch, N_pumps * (N_modes + 1)) all in (W)

        Returns
        -------
        signal_spectrum: torch.Tensor
          signal powers on each mode (B, N_signals, N_modes)
        """

        batch_size = x.shape[0]
        num_freqs = self.num_channels + self.num_pumps

        interpolation_grid = torch.zeros(
            (batch_size, 1, num_freqs ** 2, 2), dtype=x.dtype, device=x.device,
        )

        pump_wavelengths = x[:, : self.num_pumps]

        pump_loss = self._alpha_to_linear(
            self.loss_coefficients[2]
            + self.loss_coefficients[1] * pump_wavelengths
            + self.loss_coefficients[0] * (pump_wavelengths) ** 2
        ).repeat_interleave(self.modes, dim=1)

        losses = torch.cat(
            (
                pump_loss,
                self.signal_loss.expand(batch_size, self.num_channels * self.modes),
            ),
            dim=1,
        )

        # Concatenate input pump wavelengths with signal wavelengths in a (B, P + S)
        # Tensor
        pump_freqs = self._lambda2frequency(pump_wavelengths)

        total_freqs = torch.cat(
            (pump_freqs, self.signal_frequency.expand(batch_size, self.num_channels)),
            dim=1,
        )
        total_wavelenghts = self._frequency2lambda(total_freqs)

        # I don't want to allow for negative values of the pump power in the optimizer
        total_power = torch.cat(
            (
                torch.abs(x[:, self.num_pumps:]),
                self.signal_power.expand(batch_size, self.num_channels * self.modes),
            ),
            1,
        )
        freqs_diff = self._batch_diff(total_freqs)
        interpolation_grid[:, 0, :, 0] = freqs_diff.view(batch_size, -1)

        # Compute the raman gain between the signals and pumps
        # ! Should I precompute the raman gain between signals?
        # ! I could compute it in the __init__ and then expand/cat
        # ! when needed.
        idx = interpolation_grid[:, 0, :, 0] < 0
        gain = self._interpolate_response(torch.abs(interpolation_grid))[:, 0, 0, :]
        gain[idx] *= -1

        # Restore the batched matrices: (B, N * N) -> (B, N, N)
        gain = gain.view(batch_size, num_freqs, num_freqs)
        # diag = torch.diagonal(gain, dim1=-2, dim2=-1).fill_(0)

        # Compute the scaling factor (enforce energy conservation)
        one = torch.tensor(1, dtype=x.dtype, device=x.device)
        freq_scaling = torch.max(
            one,
            total_freqs.view(batch_size, -1, 1)
            / (total_freqs.view(batch_size, -1, 1).transpose(1, 2)),
        )

        gain *= freq_scaling

        # np.save("tc_gains.npy", total_freqs[0, :].detach().numpy())
        """
     build gain matrix, tiling gains by the number of modes and multiplying
     for the corresponding overlap integral
     In the gain matrix, each entry corresponds to the gain between two frequencies
     Now we have to repeat each row and column by the number of the fiber modes, so
     that a block-like matrix is built. Each block contains the gain all the modes
     at two frequencies
    
                F1        F2
            +---------+---------+
            | m11 m12 | m11 m12 |
       F1   | m21 m22 | m21 m22 |
            +---------+---------+
            | m11 m12 | m11 m12 |
       F2   | m21 m22 | m21 m22 |
            +---------+---------+
            
     mantain the same topology?
     gain = gain.repeat_interleave(self.modes, dim=1).repeat_interleave(
     self.modes, dim=2
     )
         beware, dim = 0 is the batch dimension
    """

        gain = gain.repeat_interleave(self.modes, dim=1).repeat_interleave(
            self.modes, dim=2
        ).float()
        # print("self.modes", self.modes)

        # oi = torch.from_numpy(self.fiber.get_oi_matrix_torch(range(self.modes), 3e8 / total_freqs))
        oi = self.fiber.torch_oi.evaluate_oi_tensor(total_wavelenghts)

        # print("\nTORCH OI: ", oi.shape)
        # oi_numpy = oi.detach().numpy()
        """
            +---------+   +---------+  +---------+
            | c11 c12 |   | c11 c12 |  | c11 c12 |
            | c21 c22 |   | c21 c22 |  | c21 c22 |
            +---------+0  +---------+1 +---------+2

            A1                        B1                      X
            +---------+---------+    +---------+---------+   +---------+---------+
            | c11 c12 | c11 c12 |    | c11 c12 | c11 c12 |   | c11 c12 | c11 c12 |
            | c21 c22 | c21 c22 |    | c21 c22 | c21 c22 |   | c21 c22 | c21 c22 |
            +---------+---------+    +---------+---------+   +---------+---------+
            | c11 c12 | c11 c12 |    | c11 c12 | c11 c12 |   | c11 c12 | c11 c12 |
            | c21 c22 | c21 c22 |    | c21 c22 | c21 c22 |   | c21 c22 | c21 c22 |
            +---------+---------+0   +---------+---------+1  +---------+---------+2

            F1                    F2                    FX
            +-------+-------+     +-------+-------+     +---------+---------+
            | f1 f1 | f2 f2 |     | f1 f1 | f1 f1 |     | c11 c12 | c11 c12 |
            | f1 f1 | f2 f2 |     | f1 f1 | f1 f1 |     | c21 c22 | c21 c22 |
            +-------+-------+  *  +-------+-------+  =  +---------+---------+
            | f1 f1 | f2 f2 |     | f2 f2 | f2 f2 |     | c11 c12 | c11 c12 |
            | f1 f1 | f2 f2 |     | f2 f2 | f2 f2 |     | c21 c22 | c21 c22 |
            +-------+-------+0    +-------+-------+1    +---------+---------+2
    """

        # oi = torch.from_numpy(self.overlap_integrals_avg[None, :, :].repeat(num_freqs, axis=1).repeat(num_freqs, axis=2)).float()
        G = gain * oi
        solution = torch_rk4(
            MMFRamanAmplifier.ode, total_power, self.z, losses, G, self.direction,
        ).view(-1, num_freqs, self.modes)
        signal_spectrum = solution[:, self.num_pumps:, :].clone()

        # if self.counterpumping:
        #   pump_initial_power = solution[:, : self.num_pumps, :].clone()
        #   return signal_spectrum, pump_initial_power
        # else:
        # print("signal_spectrum", signal_spectrum)

        return signal_spectrum
