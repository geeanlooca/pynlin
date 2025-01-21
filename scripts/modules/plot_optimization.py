from .cfg import Config, get_next_filename
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import viridis
from pynlin.utils import watt2dBm


def plot_profiles(signal_wavelengths,
                  signal_solution,
                  ase_solution,
                  pump_wavelengths,
                  pump_solution,
                  cf: Config):
    plt.clf()
    plt.figure(figsize=(4, 3))
    cmap = viridis
    z_plot = np.linspace(0, cf.fiber_length, len(pump_solution[:, 0, 0])) * 1e-3
    # lss = ["-", "--", "-.", ":", "-"]
    mode_labels = ["LP01", "LP11", "LP21", "LP02"]
    for i in range(cf.n_modes):
        plt.plot(z_plot,
                 watt2dBm(signal_solution[:, :, i]), color=cmap(i / cf.n_modes + 0.2), alpha=0.3)
        try:
          plt.plot(z_plot,
                 watt2dBm(ase_solution[:, :, i]), color=cmap(i / cf.n_modes + 0.2), alpha=0.3, ls="-.")
        except:
          print(f"calculation without ASE.")
    pass
    plt.ylabel(r"$P$ [dBm]")
    plt.xlabel(r"$z$ [km]")
    # plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(get_next_filename("media/optimization/signal_ase_profile", "pdf"))
    plt.clf()
    #
    plt.figure(figsize=(4, 3))
    cmap = viridis
    z_plot = np.linspace(0, cf.fiber_length, len(pump_solution[:, 0, 0])) * 1e-3  
    #
    for i in range(cf.n_modes):
        plt.plot(z_plot,
                 watt2dBm(pump_solution[:, :, i]), color=cmap(i / cf.n_modes + 0.2), alpha=0.3)
    plt.grid(False)
    plt.ylabel(r"$P$ [dBm]")
    plt.xlabel(r"$z$ [km]")
    # plt.legend()
    plt.tight_layout()
    plt.savefig(get_next_filename("media/optimization/pump_profile", "pdf"))
    #
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
    plt.savefig(get_next_filename("media/optimization/flatness", "pdf"))
    print(f"Plot saved.")
    return


def analyze_optimization(
  signal_wavelengths, 
  signal_solution, # in Watt
  ase_solution, # in Watt
  pump_wavelengths,
  pump_solution, # in Watt
  cf):
  signal_solution_dBm = watt2dBm(signal_solution)
  pump_solution_dBm = watt2dBm(pump_solution)
  flatness = np.max(signal_solution_dBm[-1, :, :]) - np.min(signal_solution_dBm[-1, :, :])
  approx_loss = -0.2e-3 * cf.fiber_length
  avg_pump_power_0 = np.mean(pump_solution_dBm[0, :, :])
  avg_pump_power_L = np.mean(pump_solution_dBm[-1, :, :])
  print(f"{'Optimization metric':<30} | {'Value':>10}")
  print("-" * 43)
  print(f"{'Flatness':<30} | {flatness:.5e} dB")
  print(f"{'Loss':<30} | {approx_loss:.5e} dB")
  try:
    ase_solution_dBm = watt2dBm(ase_solution)
    avg_ase = np.mean(ase_solution_dBm[-1, :, :])
    print(f"{'ASE':<30} | {avg_ase:.5e} dB")
  except:
    print(f"calculation without ASE.")
    pass
  print(f"{'Average pump power at z=0':<30} | {avg_pump_power_0:.5e} dBm")
  print(f"{'Average pump power at z=L':<30} | {avg_pump_power_L:.5e} dBm")
  return