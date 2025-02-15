from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.integrate
import scipy.optimize
from numpy import polyval
from scipy.constants import nu2lambda, lambda2nu

import pynlin.utils
from pynlin.fiber import Fiber, MMFiber
from pynlin.pulses import Pulse
from pynlin.raman.response import gain_spectrum, impulse_response
from pynlin.utils import (
    alpha_to_linear,
    dBm2watt,
    watt2dBm,
    wavelength_to_frequency,
)
from scipy.constants import Boltzmann as kB
from scipy.constants import Planck as h_planck


class TimeOut(Exception):
    """Raise a timeout exception to stop the shooting algorithm."""
    pass


class RamanAmplifier:
    def __init__(
        self,
        fiber: Fiber,
        response_bandwidth: float = 40e12,
        viz: bool = False
    ):

        self.response_bandwidth = response_bandwidth

        self.spline = self._compute_gain_spectrum(response_bandwidth)
        self.fiber = fiber

        super().__init__()

    def _compute_gain_spectrum(self, bandwidth, spacing=100e9):

        fs = 2 * bandwidth

        num_samples = np.ceil(fs / spacing)

        dt = 1 / fs
        duration = (num_samples - 1) * dt
        dF = fs / num_samples

        f = np.arange(num_samples) * dF

        resp, t = impulse_response(None, fs, num_samples=num_samples)

        resp -= resp.mean()

        spectrum = np.fft.fft(resp)

        gain_spectrum = -np.imag(spectrum)

        # Normalize the peak to 1
        gain_spectrum /= np.max(np.abs(gain_spectrum))

        # Compute the spline representation of the signal for later use
        # This is done in the constructor so that each call of the
        # `propagate` method does not have to recompute it. It will only do
        # so in case the maximum frequency does not fall in the bandwidth
        # of the precomputed spectrum
        spline = scipy.interpolate.splrep(f, gain_spectrum, k=3)

        return spline

    def _interpolate_gain(self, frequencies):
        # if negative frequencies are passed, just compute the gain
        # for the absolute value and then change the sign

        negative_idx = frequencies < 0

        pos_freqs = np.abs(frequencies)

        max_freq = np.max(pos_freqs)

        if max_freq > self.response_bandwidth:
            print("Warning: recomputing spline representation")
            self.response_bandwidth = 2 * max_freq

            # Recompute the gain spectrum with a bigger bandwidth
            self.spline = self._compute_gain_spectrum(self.response_bandwidth)

        # compute the interpolate values from the spline representation
        # we computed in the constructor
        gains = scipy.interpolate.splev(pos_freqs, self.spline)

        # switch the sign for negative frequencies
        gains[negative_idx] *= -1

        return gains

    def propagate(self):
        pass

    def solve(
        self,
        signal_power,
        signal_wavelength,
        pump_power,
        pump_wavelength,
        z,
        pump_direction=1,
        use_power_at_fiber_start=False,
        check_photon_count=False,
        reference_bandwidth=0.1,
        temperature=300,
    ):

        # raman_coefficient = self.fiber.raman_coefficient
        # effective_core_area = self.fiber.effective_area

        num_signals = signal_power.shape[0]
        num_pumps = pump_power.shape[0]
        total_signals = num_pumps + num_signals
        num_ase = num_signals

        signal_direction = np.ones_like(signal_power)
        if np.isscalar(pump_direction):
            pump_direction = np.sign(pump_direction) * np.ones_like(pump_power)
        else:
            pump_direction = np.atleast_1d(pump_direction)

        # organization of gain matrix
        self.direction = np.concatenate((pump_direction, signal_direction))

        # check if we need a shooting algorithm
        shooting = np.any(self.direction < 0)

        self.pump_power = pump_power
        self.signal_power = signal_power
        self.pump_wavelengths = pump_wavelength
        self.signal_wavelengths = signal_wavelength

        wavelengths = np.concatenate((self.pump_wavelengths, self.signal_wavelengths))
        frequencies = lambda2nu(wavelengths)

        num_ase = num_signals

        # input_power = np.concatenate((self.pump_power, self.signal_power))
        input_power_with_ase = np.concatenate(
            (self.pump_power, self.signal_power, np.zeros(num_ase)))

        losses_linear = self.get_linear_losses(wavelengths)

        gain_matrix = self.compute_gain_matrix(frequencies)

        # Compute the frequency shifts for each wave
        frequency_shifts = np.zeros((total_signals, total_signals))
        for i in range(total_signals):
            frequency_shifts[i, :] = frequencies - frequencies[i]

        h_planck = 6.626e-34
        kB = 1.380649e-23
        # Compute the phonon occupancy factor (adopt the view pump+ase)
        Hinv = np.exp(h_planck * np.abs(frequency_shifts) / (kB * temperature)) - 1
        np.fill_diagonal(Hinv, 1e56)  # random fill the diagonal
        eta_plus = 1 + 1 / Hinv
        np.fill_diagonal(eta_plus, 0)
        gain_matrix_ase = eta_plus * gain_matrix

        if shooting:
            if not use_power_at_fiber_start:
                raise NotImplementedError(
                    "Cannot yet use counter-propagating pumps without using the power at the fiber start, i.e. no shooting implemented"
                )

            sol = scipy.integrate.odeint(
                RamanAmplifier.raman_ode_with_ase,
                input_power_with_ase,
                z,
                args=(losses_linear, gain_matrix, gain_matrix_ase, np.hstack((self.direction, np.ones(
                    num_ase))), temperature, reference_bandwidth, num_pumps, num_signals, frequencies),
            )

            pump_solution = sol[:, :num_pumps]
            signal_solution = sol[:, num_pumps:num_pumps + num_signals]
            ase_solution = sol[:, -num_ase:]

        else:
            sol = scipy.integrate.odeint(
                RamanAmplifier.raman_ode_with_ase,
                input_power_with_ase,
                z,
                args=(losses_linear, gain_matrix, gain_matrix_ase, np.hstack((self.direction, np.ones(
                    num_ase))), temperature, reference_bandwidth, num_pumps, num_signals, frequencies),
            )

            pump_solution = sol[:, :num_pumps]
            signal_solution = sol[:, num_pumps:num_pumps + num_signals]
            ase_solution = sol[:, -num_ase:]
        if check_photon_count:
            photon_count = np.sum(sol / frequencies, axis=1)
            return pump_solution, signal_solution, photon_count
        else:
            return pump_solution, signal_solution, ase_solution

    def compute_gain_matrix(self, frequencies):
        """Generate the matrix of Raman gains between each pair of frequencies."""
        num_frequencies = len(frequencies)
        # Compute the frequency shifts for each signal
        frequency_shifts = np.zeros((num_frequencies, num_frequencies))
        for i in range(num_frequencies):
            frequency_shifts[i, :] = frequencies - frequencies[i]

        gains = self._interpolate_gain(frequency_shifts)

        gains *= self.fiber.raman_coefficient / self.fiber.effective_area

        # Force diagonal to be 0
        np.fill_diagonal(gains, 0)

        # gains = np.triu(gains) + np.triu(gains, 1).T

        # compute the frequency scaling factor
        freqs = np.expand_dims(frequencies, axis=-1)
        freq_scaling = np.maximum(1, freqs * (1 / freqs.T))

        # gain matrix
        gain_matrix = freq_scaling * gains
        return gain_matrix

    @staticmethod
    def raman_ode(P, z, losses, gain_matrix, direction):
        """System of equations describing a Raman amplifier. No ASE. No Rayleigh back-reflection."""
        try:
            dPdz = (
                -losses[:, np.newaxis] + np.matmul(gain_matrix, P[:, np.newaxis])
            ) * P[:, np.newaxis]
        except ValueError:
            breakpoint()
        return direction * np.squeeze(dPdz)

    @staticmethod
    def raman_ode_with_ase(P, z, losses, gain_matrix, gain_matrix_ase, direction, temperature, ref_bandwidth, num_pumps, num_signals, frequencies):
        """System of equations describing a Raman amplifier. With ASE. No Rayleigh back-reflection. No ASE-to-pump power transfer"""
        h_planck = 6.626e-34

        num_ase = num_signals
        num_power = (num_signals + num_pumps)

        P_ = P[:num_power]
        P_ase = P[-num_ase:]
        losses_ase = losses[-num_ase:]

        gain_factor = np.matmul(gain_matrix, P_[:, np.newaxis])

        #      | Pumps | Signals |
        # Pumps     A        B
        # ------
        # ASE      C        D
        #
        try:

            dPowerdz = (-losses[:, np.newaxis] + gain_factor) * P_[:, np.newaxis]

            gain_factor_ase = np.matmul(gain_matrix_ase, P_[:, np.newaxis])
            gain_factor_ase = gain_factor_ase[-num_ase:]

            dASEdz = -losses[-num_ase:, np.newaxis] * P_ase[:, np.newaxis]
            dASEdz += gain_factor[-num_ase:] * P_ase[:, np.newaxis]
            dASEdz += (
                gain_factor_ase * 2 * h_planck *
                frequencies[-num_ase:, np.newaxis] * ref_bandwidth
            )

        except ValueError:
            breakpoint()
        dPdz = np.vstack((dPowerdz, dASEdz))
        return direction * np.squeeze(dPdz)

    @staticmethod
    def extended_raman_ode(P, z, losses, gain_matrix, direction):
        """System of equations describing a Raman amplifier. No ASE. No Rayleigh back-reflection."""

        # breakpoint()
        dPdz = np.ones_like(P)
        for z_ind in range(P.shape[1]):
            spectrum = P[:, z_ind]
            dPdz_z = (
                -losses[:, np.newaxis] + np.matmul(gain_matrix, spectrum[:, np.newaxis])
            ) * spectrum[:, np.newaxis]
            dPdz[:, z_ind] = direction * np.squeeze(dPdz_z)
        return dPdz

    def get_linear_losses(self, wavelengths):
        """Compute the linear loss coefficient for the given wavelengths."""
        losses = self.fiber.loss_profile(wavelengths)
        losses_linear = alpha_to_linear(losses)
        return losses_linear

    def solve_shooting(self, pump_power, signal_power, z, solver="scipy"):
        """Shooting algorithm presented in [1].

        References
        ----------
        [1] Hai Ming Jiang and Kang Xie, "Efficient and robust shooting algorithm
          for numerical design of bidirectionally pumped Raman fiber amplifiers,"
          J. Opt. Soc. Am. B 29, 8-14 (2012)
        """

        if solver == "scipy":
            return self.solve_shooting_scipy(pump_power, signal_power, z)
        elif solver == "jiang":
            return self.solve_shooting_jiang(pump_power, signal_power, z)

    def solve_shooting_scipy(self, pump_power, signal_power, z):
        num_pumps = len(pump_power)
        alpha = 0.01

        # 1): construct the vector of initial conditions

        # 1.a) Order the frequencies from largest to smallest

        def reverse_sort(a, b):
            sort_idx = np.argsort(a)
            a = a[sort_idx]
            b = b[sort_idx]
            return a[::-1], b[::-1]

        pump_frequencies = lambda2nu(self.pump_wavelengths)
        sorted_pump_frequencies, sorted_pump_powers = reverse_sort(
            pump_frequencies, pump_power
        )

        gain_matrix = self.compute_gain_matrix(sorted_pump_frequencies)
        sorted_pump_losses = self.get_linear_losses(nu2lambda(sorted_pump_frequencies))
        signal_losses = self.get_linear_losses(self.signal_wavelengths)

        direction_tmp = np.ones_like(sorted_pump_powers)

        sol = scipy.integrate.odeint(
            RamanAmplifier.raman_ode,
            sorted_pump_powers,
            z,
            args=(sorted_pump_losses, gain_matrix, direction_tmp),
        )

        pump_solution_initial_cond = sol[::-1, :num_pumps]

        # signal_solution_initial_cond = sorted_pump_powers * np.exp(
        #     -z[::-1, np.newaxis] * sorted_pump_losses
        # )

        signal_solution_initial_cond = self.signal_power * np.exp(
            -z[:, np.newaxis] * signal_losses
        )

        # plt.figure()
        # plt.plot(z, pynlin.utils.watt2dBm(signal_solution_initial_cond))
        # plt.plot(z, pynlin.utils.watt2dBm(pump_solution_initial_cond))
        # # plt.show()

        initial_conditions = np.hstack(
            (pump_solution_initial_cond, signal_solution_initial_cond)
        ).T

        wavelengths = np.concatenate(
            (nu2lambda(sorted_pump_frequencies), self.signal_wavelengths)
        )
        frequencies = lambda2nu(wavelengths)
        losses_linear = self.get_linear_losses(wavelengths)
        gain_matrix = self.compute_gain_matrix(frequencies)
        target_power_spectrum = np.concatenate((sorted_pump_powers, self.signal_power))

        def boundary_residuals(ya, yb):
            residuals = np.zeros_like(ya)
            fwd_idx = self.direction > 0
            bwd_idx = self.direction < 0
            residuals[fwd_idx] = ya[fwd_idx]
            residuals[bwd_idx] = yb[bwd_idx]
            return target_power_spectrum - residuals

        def ode(z, P):
            return RamanAmplifier.extended_raman_ode(
                P, z, losses_linear, gain_matrix, self.direction
            )

        result = scipy.integrate.solve_bvp(
            ode, boundary_residuals, z, initial_conditions, verbose=1
        )
        return result.y.transpose()

    def solve_shooting_jiang(self, pump_power, signal_power, z):
        """Shooting algorithm presented in [1].

        References
        ----------
        [1] Hai Ming Jiang and Kang Xie, "Efficient and robust shooting algorithm
          for numerical design of bidirectionally pumped Raman fiber amplifiers,"
          J. Opt. Soc. Am. B 29, 8-14 (2012)
        """
        num_pumps = len(pump_power)
        alpha = 0.01

        # 1): construct the vector of initial conditions

        # 1.a) Order the frequencies from largest to smallest

        def reverse_sort(a, b):
            sort_idx = np.argsort(a)
            a = a[sort_idx]
            b = b[sort_idx]
            return a[::-1], b[::-1]

        pump_frequencies = lambda2nu(self.pump_wavelengths)
        sorted_pump_frequencies, sorted_pump_powers = reverse_sort(
            pump_frequencies, pump_power
        )

        gain_matrix = self.compute_gain_matrix(sorted_pump_frequencies)
        sorted_pump_losses = self.get_linear_losses(nu2lambda(sorted_pump_frequencies))

        # 1.b) Propagate the pumps adding them one by one.
        x0 = np.zeros_like(sorted_pump_powers)
        input_power_tmp = np.zeros_like(sorted_pump_powers)
        direction_tmp = np.ones_like(sorted_pump_powers)

        for i, Pp in enumerate(sorted_pump_powers):

            input_power_tmp[i] = sorted_pump_powers[i]

            sol = scipy.integrate.odeint(
                RamanAmplifier.raman_ode,
                input_power_tmp,
                z,
                args=(sorted_pump_losses, gain_matrix, direction_tmp),
            )

            pump_solution = sol[:, :num_pumps]
            signal_solution = sol[:, num_pumps:]

            P_current_pump_z0 = pump_solution[-1, i]
            x0[i] = P_current_pump_z0

            # plt.figure()
            # plt.plot(z[::-1], pynlin.utils.watt2dBm(pump_solution))
            # plt.title(f"Phase 1, iteration {i+1}: find initial guesses")

        # 1.c) Determine the scaling vector S
        S = np.ones_like(x0) / 1e3
        x0 = x0 * S

        # Step 2) Iterate to correct the initial guesses

        wavelengths = np.concatenate(
            (nu2lambda(sorted_pump_frequencies), self.signal_wavelengths)
        )
        frequencies = lambda2nu(wavelengths)
        losses_linear = self.get_linear_losses(wavelengths)
        gain_matrix = self.compute_gain_matrix(frequencies)

        def solve_system(x):
            """Get the solution of the system for `x` input pump powers."""
            input_power = np.concatenate((x, self.signal_power))
            sol = scipy.integrate.odeint(
                RamanAmplifier.raman_ode,
                input_power,
                z,
                args=(losses_linear, gain_matrix, self.direction),
            )
            return sol

        def get_output_pump_powers(x):
            """Get the power of the pumps at the end of the fiber for the current guess `x`."""
            sol = solve_system(x)
            return sol[-1, :num_pumps]

        def compute_error(x):
            P_out = get_output_pump_powers(x)
            D = P_out - pump_power
            return D

        def pump_mse(x):
            err = compute_error(x)
            return np.mean(err**2)

        def gradient(x, h=1e-3):
            grad = np.zeros((x.shape[0],))

            for i in range(num_pumps):
                x_p = np.copy(x)
                x_m = np.copy(x)
                x_p[i] = x_p[i] + h
                x_m[i] = x_m[i] - h
                mse_p = pump_mse(x_p)
                mse_m = pump_mse(x_m)
                grad[i] = (mse_p - mse_m) / (2 * h)
            return grad

        def compute_jacobian(x0, h=1e-3):
            """Compute the Jacobian matrix."""
            J = np.zeros((x0.shape[0], x0.shape[0]))
            D = compute_error(x0)

            for i in range(num_pumps):
                x_p = np.copy(x0)
                x_m = np.copy(x0)
                x_p[i] = x_p[i] + h
                x_m[i] = x_m[i] - h
                D_p = compute_error(x_p)
                D_m = compute_error(x_m)

                J[:, i] = (D_p - D_m) / (2 * h)

            return J

        # Iterate on the initial guesses
        num_iter = 0

        Ds = []

        while num_iter < 1000:
            D = compute_error(x0)

            print(f"Iteration {num_iter}")
            print(f"\tx0={pynlin.utils.watt2dBm(x0)} dBm")
            print(f"\tError={D * 1e3} mW")
            Ds.append(D)

            sol = solve_system(x0)
            pump_solution = sol[:, :num_pumps]
            signal_solution = sol[:, num_pumps:]

            # fig, ax = plt.subplots(ncols=2)
            # ax[0].plot(z, pynlin.utils.watt2dBm(pump_solution), color="red")
            # ax[0].plot(z, pynlin.utils.watt2dBm(signal_solution), color="black")
            # ax[0].set_title(f"Iteration {num_iter}")
            # ax[1].semilogy(np.abs(np.stack(Ds)), marker="x")
            # plt.show()

            eta = 0.0001
            try:

                G = gradient(x0)
                # J = compute_jacobian(x0)

                # delta_P = -np.dot(np.linalg.inv(J), D)
                # cap the maximum change to 10 mW
                # signs = np.sign(delta_P)
                # abs = np.abs(delta_P)
                # delta_P_capped = np.minimum(abs * 1e3, 100)
                # delta_P = signs * delta_P_capped * 1e-3

                #     # breakpoint()

                # print(f"\tdelta_P={delta_P * 1e3} mW")
                # x0 = x0 + alpha * delta_P
                x0 = x0 - eta * G
                print(f"\tx0_new={pynlin.utils.watt2dBm(x0)} dBm")
                num_iter += 1
            except np.linalg.LinAlgError:
                break

        return solve_system(x0)


class MMFRamanAmplifier(RamanAmplifier):
    # def __init__(self, bandwidth=40e12):

    #     super().__init__()

    def solve(
        self,
        signal_power,
        signal_wavelength,
        pump_power,
        pump_wavelength,
        z,
        fiber,
        counterpumping=False,
        ase=False,
        reference_bandwidth=0.1 * 1e-9,
        temperature=300,
        shooting=None,
        initial_guesses=None,
        direction=None
    ):
        """Solve the multi-mode Raman amplifier equations [1].

        Params
        ------
        signal_power: np.ndarray
          The input signal power in a 2d ndarray of shape (wavelengths, modes)
          expressed in Watt.
        signal_wavelength: np.ndarray
          The input signal wavelengths in a 2d ndarray of shape (wavelengths,)
          expressed in meters.
        pump_power: np.ndarray
          The input pump power in a 2d ndarray of shape (wavelengths, modes)
          expressed in Watt.
        pump_wavelength: np.ndarray
          The input pump wavelengths in a 1d ndarray of shape (wavelengths,)
          expressed in meters.
        z: np.ndarray
          The z-axis along which to integrate the equations, expressed in
          meters.
        fiber: pyraman.Fiber
          Fiber object defining the amplifier
        counterpumping: bool, optional
          If True, `pump_power` is considered to be the pump power
          at the start of the fiber, i.e. at z = 0. The signs
          of the equations for the pump power evolution are set to -1,
          so the losses actually amplify the pump power during
          propagation in the z: 0->L direction. Optional, by default False.
        reference_bandwidth: float, optional
          The reference optical bandwidth (nm) for ASE measurement.
          Optional, by default 0.1 nm.
        temperature: float, optional
          The optical fiber temperature (K). Optional, by default 300 K.
        shooting : bool, optional
          Use a shooting method to solve the counterpumping case,
          by default None.

        References
        ----------
        .. [1] Ryf, Roland, Rene Essiambre, Johannes von Hoyningen-Huene,
            and Peter Winzer. 2012. “Analysis of Mode-Dependent Gain in
            Raman Amplified Few-Mode Fiber.” In Optical Fiber Communication
            Conference, Los Angeles, California: OSA, OW1D.2.
            https://www.osapublishing.org/abstract.cfm?uri=OFC-2012-OW1D.2
            (June 5, 2019).
        """
        num_signals = signal_power.shape[0]
        num_pumps = pump_power.shape[0]

        total_wavelengths = num_signals + num_pumps
        num_modes = fiber.modes
        total_signals = total_wavelengths
        pump_power_ = pump_power.reshape((num_modes * num_pumps))
        signal_power_ = signal_power.reshape((num_modes * num_signals))

        # Structure of the waves: modes and frequencies
        # - waves: | (pumps)
        #          | freq1                        | freqn                      |
        # array:   | LP01                LPxx     | LP01                LPxx   |

        wavelengths = np.concatenate((pump_wavelength, signal_wavelength))
        input_power = np.concatenate((pump_power_, signal_power_))
        frequencies = wavelength_to_frequency(wavelengths)

        loss_coeffs = fiber.losses

        losses_ = polyval(loss_coeffs, wavelengths)

        losses_linear = alpha_to_linear(losses_)
        losses_linear = np.repeat(losses_linear, fiber.modes)

        # Compute the frequency shifts for each signal
        frequency_shifts = np.zeros((total_wavelengths, total_wavelengths))
        for i in range(total_wavelengths):
            frequency_shifts[i, :] = frequencies - frequencies[i]

        gains = self._interpolate_gain(frequency_shifts)

        gains *= fiber.raman_coefficient

        # Must be multiplied by overlap integrals

        # Force diagonal to be 0
        np.fill_diagonal(gains, 0)
        # gains = np.triu(gains) + np.triu(gains, 1).T

        # compute the frequency scaling factor
        freqs = np.expand_dims(frequencies, axis=-1)
        freq_scaling = np.maximum(1, freqs * (1 / freqs.T))

        mode_list = np.array(range(fiber.modes))
        # change the order of creation
        oi = fiber.get_oi_matrix(mode_list, wavelengths)
        
        gain_matrix = freq_scaling * gains

        gain_matrix = gain_matrix.repeat(fiber.modes, axis=0).repeat(
            fiber.modes, axis=1
        )

        gains_mmf = gain_matrix * oi
    
        np.save("np_gains.npy", frequencies)
        if not ase:
            if direction is None:
                direction = np.ones((total_wavelengths * fiber.modes,))

            if counterpumping or shooting:
                direction[: num_pumps * fiber.modes] = -1

            if not shooting:
                sol = scipy.integrate.odeint(
                    MMFRamanAmplifier.raman_ode,
                    input_power,
                    z,
                    args=(losses_linear, gains_mmf, direction),
                )
                sol = sol.reshape((len(z), total_signals, fiber.modes))
                pump_solution = sol[:, :num_pumps, :]
                signal_solution = sol[:, num_pumps:, :]
            else:
                # SHOOTING METHOD

                pump_losses = losses_linear[: num_pumps * fiber.modes]

                if initial_guesses is None:
                    initial_guesses = pump_power_ * np.exp(-z[-1] * pump_losses) / 10

                callback_info = {
                    "max_error": float("inf"),
                    "iter": 0,
                    "threshold": 0.01,
                    "params": None,
                }

                def callback(x):
                    max_error = callback_info["max_error"]
                    threshold = callback_info["threshold"]
                    print(f"Max error: {max_error} mW")
                    if max_error < threshold:
                        # print(f"Shooting error < {threshold}, stopping optimization")
                        callback_info["params"] = np.copy(x)
                        raise TimeOut("Stopping optimization")

                def optim_fun(x0):

                    x0_ = 10 ** (x0 / 10)
                    input_power = np.concatenate((x0_, signal_power_))

                    sol = scipy.integrate.odeint(
                        MMFRamanAmplifier.raman_ode,
                        input_power,
                        z,
                        args=(losses_linear, gains_mmf, direction),
                    )

                    sol = sol.reshape((len(z), total_signals, fiber.modes))

                    pump_solution = sol[-1, :num_pumps, :].flatten()

                    # return the MSE between desired solution and obtained values
                    cost = np.sqrt(
                        np.mean((pump_solution * 1e3 - pump_power_ * 1e3) ** 2)
                    )

                    max_error = np.max(np.abs(pump_solution - pump_power_)) * 1e3

                    callback_info["max_error"] = max_error

                    # print(f"RMSE: {cost:.3f} mW\tMax. Error {max_error:.2f} mW")

                    return cost

                # bounds = [(0, None) for _ in range(num_pumps * fiber.modes)]

                try:
                    result = scipy.optimize.minimize(
                        # optim_fun, initial_guesses, method="L-BFGS-B", bounds=bounds
                        optim_fun,
                        10 * np.log10(initial_guesses),
                        method="L-BFGS-B",
                        options={"maxfun": 1e10, "maxiter": 1e10, "ftol": 1e-15},
                        # bounds=bounds,
                        callback=callback,
                    )

                    x0 = result.x
                except TimeOut:
                    x0 = callback_info["params"]

                x0 = 10 ** (x0 / 10)

                # Result of the optimization process: pump power at z=0

                # Propagate
                input_power = np.concatenate((x0, signal_power_))

                sol = scipy.integrate.odeint(
                    MMFRamanAmplifier.raman_ode,
                    input_power,
                    z,
                    args=(losses_linear, gains_mmf, direction),
                )
                sol = sol.reshape((len(z), total_signals, fiber.modes))

                pump_solution = sol[:, :num_pumps, :]
                signal_solution = sol[:, num_pumps:, :]
            return pump_solution, signal_solution
        else:
            direction = np.ones(((total_wavelengths + num_signals) * fiber.modes,))

            # Compute the phonon occupancy factor
            Hinv = np.exp(h_planck * np.abs(frequency_shifts) / (kB * temperature)) - 1

            eta = 1 + 1 / Hinv
            np.fill_diagonal(eta, 0)
            eta = np.repeat(np.repeat(eta, fiber.modes, axis=0), fiber.modes, axis=1)

            # Compute the new Raman gain matrix
            gain_matrix_ase = eta * gains_mmf

            # Convert reference bandwidth in hertz using the
            # central signal wavelength as reference
            central_wavelength = (signal_wavelength.max() + signal_wavelength.min()) / 2
            w_a = central_wavelength - reference_bandwidth / 2
            w_b = central_wavelength + reference_bandwidth / 2
            f_a = lambda2nu(w_a)
            f_b = lambda2nu(w_b)
            reference_bandwidth_hz = np.abs(f_a - f_b)

            if counterpumping or shooting:
                direction[: num_pumps * fiber.modes] = -1

            # Initial conditions, ase power must be 0 at z=0
            input_power_ase = np.zeros((input_power.size + num_signals * fiber.modes,))
            input_power_ase[: input_power.size] = input_power

            signal_frequencies = wavelength_to_frequency(signal_wavelength)
            ase_frequencies = np.repeat(signal_frequencies, fiber.modes)

            sol = scipy.integrate.odeint(
                MMFRamanAmplifier.raman_ode_with_ase,
                input_power_ase,
                z,
                args=(
                    losses_linear,
                    gains_mmf,
                    gain_matrix_ase,
                    ase_frequencies,
                    reference_bandwidth_hz,
                    direction,
                    num_signals,
                    num_pumps,
                    fiber.modes,
                ),
            )

            sol = sol.reshape((len(z), total_signals + num_signals, fiber.modes))
            power_solution = sol[:, :total_signals, :]
            pump_solution = power_solution[:, :num_pumps, :]
            signal_solution = power_solution[:, num_pumps:, :]
            ase_solution = sol[:, -num_signals:, :]

            return pump_solution, signal_solution, ase_solution

    def solve_shooting():
        pass

    @staticmethod
    def raman_ode(P, z, losses, gain_matrix, direction):
        """Integration step of the multimode Raman system."""
        dPdz = (-losses[:, np.newaxis] + np.matmul(gain_matrix,
                                                   P[:, np.newaxis])) * P[:, np.newaxis]

        return np.squeeze(dPdz) * direction

    def raman_ode_with_ase(
        P,
        z,
        losses,
        gain_matrix,
        gain_matrix_ase,
        frequencies,
        ref_bandwidth,
        direction,
        num_signals,
        num_pumps,
        num_modes,
    ):
        """Integration step of the multimode Raman system with ASE."""
        num_ase = num_signals * num_modes
        num_power = (num_signals + num_pumps) * num_modes

        P_ = P[:num_power]
        P_ase = P[-num_ase:]
        losses_ase = losses[-num_ase:]

        gain_factor = np.matmul(gain_matrix, P_[:, np.newaxis])

        dPowerdz = (-losses[:, np.newaxis] + gain_factor) * P_[:, np.newaxis]

        gain_factor_ase = np.matmul(gain_matrix_ase, P_[:, np.newaxis])
        gain_factor_ase = gain_factor_ase[-num_ase:]

        dASEdz = -losses_ase[:, np.newaxis] * P_ase[:, np.newaxis]
        dASEdz += gain_factor[-num_ase:] * P_ase[:, np.newaxis]
        dASEdz += (
            gain_factor_ase * 2 * h_planck * frequencies[:, np.newaxis] * ref_bandwidth
        )

        dPdz = np.vstack((dPowerdz, dASEdz))
        return np.squeeze(dPdz) * direction
