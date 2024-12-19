import os
import tqdm
from multiprocessing import Pool
from matplotlib import pyplot as plt
import numpy as np
import torch
from scipy.constants import lambda2nu, nu2lambda
from matplotlib.cm import viridis
from scripts.modules.load_fiber_values import load_group_delay
from modules import cfg

import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
import pynlin.fiber
from pynlin.raman.pytorch.gain_optimizer import GainOptimizer
from pynlin.raman.pytorch.solvers import MMFRamanAmplifier
from pynlin.raman.solvers import MMFRamanAmplifier as NumpyMMFRamanAmplifier
from pynlin.utils import dBm2watt, watt2dBm
import pynlin.constellations

def ct_solver(power_per_pump,
              pump_band_a,
              pump_band_b,
              learning_rate,
              epochs,
              lock_wavelengths,
              use_precomputed=False,
              optimize=False
              ):
  
    cf = cfg.load_toml_to_struct("./input/config.toml")

    oi_fit = np.load('oi_fit.npy')
    oi_avg = np.load('oi_avg.npy')
    use_avg_oi = True
    num_original_modes = oi_avg[0].shape[0]
    matrix_avg = oi_avg
    matrix_zeros = np.tile(np.zeros((num_original_modes, num_original_modes))[
                           None, :, :], (5, 1, 1))
    oi_avg_complete = np.stack((*matrix_zeros, matrix_avg), axis=0)
    if use_avg_oi:
        oi_set = oi_avg_complete
    else:
        oi_set = oi_fit
    oi_fit = oi_avg_complete

    ref_bandwidth = cf.baud_rate
    fiber = pynlin.fiber.MMFiber(
        effective_area=80e-12,
        n_modes=cf.n_modes,
        overlap_integrals=oi_set,
        group_delay=load_group_delay()
    )
    actual_fiber = pynlin.fiber.MMFiber(
        effective_area=80e-12,
        n_modes=cf.n_modes,
        overlap_integrals=oi_fit,
    )
    wdm = pynlin.wdm.WDM(
        spacing=cf.channel_spacing,
        num_channels=cf.n_channels,
        center_frequency=cf.center_frequency
    )
    integration_steps = 100
    z_max = np.linspace(0, fiber.length, integration_steps)

    np.save("z_max.npy", z_max)
    pbar_description = "Optimizing vs signal power"
    pbar = tqdm.tqdm(cf.launch_power, leave=False)
    pbar.set_description(pbar_description)
    print(
        f"> Running optimization for Pin = {cf.launch_power:.2e} dBm, and gain = {cf.raman_gain:.2e} dB.\n")
    if use_precomputed and os.path.exists("results/pump_solution_ct_power" + str(cf.launch_power) + "_opt_gain_" + str(cf.raman_gain) + ".npy"):
        print("Result already computed for power: ",
              cf.launch_power, " and gain: ", cf.raman_gain)
        return
    else:
        print("Computing the power: ", cf.launch_power, " and gain: ", cf.raman_gain)
    initial_pump_frequencies = lambda2nu(
        np.linspace(pump_band_a, pump_band_b, cf.n_pumps))

    power_per_channel = dBm2watt(cf.launch_power)
    signal_wavelengths = wdm.wavelength_grid()
    initial_pump_wavelengths = nu2lambda(initial_pump_frequencies[:cf.n_pumps])

    initial_pump_powers = np.ones_like(initial_pump_wavelengths) * power_per_pump
    initial_pump_powers = initial_pump_powers.repeat(cf.n_modes, axis=0)
    torch_amplifier_ct = MMFRamanAmplifier(
        fiber.length,
        integration_steps,
        cf.n_pumps,
        signal_wavelengths,
        power_per_channel,
        actual_fiber,
        counterpumping=True
    )

    if use_precomputed:
        try:
            initial_pump_wavelengths = np.load("results/opt_pump_wavelengths.npy")
            initial_pump_powers = np.load("results/opt_pump_powers.npy")
        except:
            print("The precomputed values are either")

    optimizer = GainOptimizer(
        torch_amplifier_ct,
        torch.from_numpy(initial_pump_wavelengths),
        torch.from_numpy(initial_pump_powers),
    )

    signal_powers = np.ones_like(signal_wavelengths) * power_per_channel
    signal_powers = signal_powers[:, None].repeat(cf.n_modes, axis=1)
    target_spectrum = watt2dBm(signal_powers)[None, :, :] + cf.raman_gain

    if optimize:
        pump_wavelengths, pump_powers = optimizer.optimize(
            target_spectrum=target_spectrum,
            epochs=epochs,
            learning_rate=learning_rate,
            lock_wavelengths=lock_wavelengths,
        )
        np.save("results/opt_pump_wavelengths.npy", pump_wavelengths)
        np.save("results/opt_pump_powers.npy", pump_powers)
        print("--" * 30)
    else:
        pump_wavelengths = initial_pump_wavelengths
        pump_powers = initial_pump_powers

    amplifier = NumpyMMFRamanAmplifier(fiber)
    pump_powers = pump_powers.reshape((cf.n_pumps, cf.n_modes))
    pump_solution, signal_solution = amplifier.solve(
        signal_powers,
        signal_wavelengths,
        pump_powers,
        pump_wavelengths,
        z_max,
        actual_fiber,
        counterpumping=True,
        reference_bandwidth=ref_bandwidth
    )
    # fixed mode
    flatness = np.max(
        watt2dBm(signal_solution[-1, :, :])) - watt2dBm(np.min(signal_solution[-1, :, :]))
    loss = -0.2e-3 * cf.fiber_length
    on_off_gain = -loss + cf.raman_gain
    flatness_percent = flatness / on_off_gain * 100
    # /  10**(cf.raman_gain/10) * 100
    avg_gain = np.mean(watt2dBm(signal_solution[-1, :, :]))
    print("_" * 35)
    print("Pump powers")
    print(watt2dBm(pump_powers.reshape((cf.n_pumps, cf.n_modes))))
    print("Initial pump powers")
    print(watt2dBm(initial_pump_powers.reshape((cf.n_pumps, cf.n_modes))))
    print("Pump wavelenghts")
    print(pump_wavelengths)
    print("Initial pump wavelenghts")
    print(initial_pump_wavelengths)
    print("_" * 35)
    print(f"> final flatness: {flatness:.2f} dB | {flatness_percent:.2f} %")
    print("_" * 35)

    plot_profiles(signal_wavelengths, signal_solution,
                  pump_wavelengths, pump_solution, cf)
    return


def plot_profiles(signal_wavelengths,
                  signal_solution,
                  pump_wavelengths,
                  pump_solution,
                  cf: cfg.Config):
    plt.clf()
    plt.figure(figsize=(4, 3))
    cmap = viridis
    z_plot = np.linspace(0, cf.fiber_length, len(pump_solution[:, 0, 0])) * 1e-3
    lss = ["-", "--", "-.", ":", "-"]
    mode_labels = ["LP01", "LP11", "LP21", "LP02"]
    for i in range(cf.n_modes):
        print(i)
        plt.plot(z_plot,
                 watt2dBm(signal_solution[:, :, i]), color=cmap(i / cf.n_modes + 0.2), alpha=0.3)
    plt.ylabel(r"$P$ [dBm]")
    plt.xlabel(r"$z$ [km]")
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.savefig("media/signal_profile.pdf")
    plt.clf()

    plt.figure(figsize=(4, 3))
    cmap = viridis
    z_plot = np.linspace(0, cf.fiber_length, len(pump_solution[:, 0, 0])) * 1e-3

    for i in range(cf.n_modes):
        plt.plot(z_plot,
                 watt2dBm(pump_solution[:, :, i]), color=cmap(i / cf.n_modes + 0.2), alpha=0.3)
    plt.grid(False)
    plt.ylabel(r"$P$ [dBm]")
    plt.xlabel(r"$z$ [km]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("media/pump_profile.pdf")

    loss = -0.2e-3 * cf.fiber_length
    on_off_gain = -loss + cf.raman_gain
    plt.clf()
    plt.figure(figsize=(4, 3))
    for i in range(cf.n_modes):
        plt.plot(signal_wavelengths * 1e6,
                 watt2dBm(signal_solution[-1, :, i]) - cf.launch_power - loss, 
                 label=mode_labels[i], 
                 color=cmap(i / cf.n_modes + 0.2))
    plt.legend()
    plt.axhline(on_off_gain, ls="--", color="black")
    plt.xlabel(r"Channel Wavelength [$\mu$ m]")
    plt.ylabel("Gain [dB]")
    plt.tight_layout()
    plt.savefig("media/flatness.pdf")
    return


ct_solver(power_per_pump   = dBm2watt(3.0),
          pump_band_a      = 1410e-9,
          pump_band_b      = 1520e-9,
          learning_rate    = 1e-4,
          epochs           = 1000,
          lock_wavelengths = 100,
          use_precomputed  = False,
          optimize         = True
          )

# Still need to find a good starting point for the pumps