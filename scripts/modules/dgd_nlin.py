import logging
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import plotly.graph_objects as go
import seaborn as sns
import pynlin.wdm
from pynlin.utils import nu2lambda
from scripts.modules.load_fiber_values import load_group_delay, load_dummy_group_delay
from numpy import polyval
from pynlin.fiber import MMFiber
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
import scripts.modules.cfg as cfg


def get_nlin(cf, 
             dgd_threshold = 3e-15):
  oi_fit = np.load('results/oi_fit.npy')
  print(f"Loading a ITU-T standardized WDM grid \n [spacing: {cf.channel_spacing*1e-9:.3e}GHz, center: {cf.center_frequency*1e-12:.3e}THz] \n")
  beta1_params = load_group_delay()
  wdm = pynlin.wdm.WDM(
      spacing=cf.channel_spacing,
      num_channels=cf.n_channels,
      center_frequency=cf.center_frequency
  )
  freqs = wdm.frequency_grid()
  modes = range(cf.n_modes)

  fiber = MMFiber(
      effective_area=80e-12,
      overlap_integrals=oi_fit,
      group_delay=beta1_params,
      length=100e3
  )

  beta1 = np.zeros((len(modes), len(freqs)))
  for i in modes:
    beta1[i, :] = fiber.group_delay.evaluate_beta1(i, freqs)
  beta2 = np.zeros((len(modes), len(freqs)))
  for i in modes:
    beta2[i, :] = fiber.group_delay.evaluate_beta2(i, freqs)
  beta1 = np.array(beta1)
  beta2 = np.array(beta2)

  # for each channel, we compute the total number of collisions that
  # needs to be computed for evaluating the total noise on that channel.
  T = 1 / cf.baud_rate
  L = cf.fiber_length

  collisions = np.zeros((len(modes), len(freqs)))
  for i in range(len(modes)):
      for j in range(len(freqs)):
          collisions[i, j] = np.floor(np.abs(np.sum(beta1 - beta1[i, j])) * L / T)

  collisions_single = np.zeros((1, len(freqs)))
  for j in range(len(freqs)):
      collisions_single[0, j] = np.floor(
          np.abs(np.sum(beta1[0:] - beta1[0, j])) * L / T)

  nlin = np.zeros((len(modes), len(freqs)))
  # only flat up to this point

  ############################
  # GAUSSIAN NOISE\
  def pair_noise(dgd):
      return np.where(
          dgd > dgd_threshold, # this needs to be set carefully
          L / (T * dgd),
          np.sqrt(np.pi) * (L / (T * np.sqrt(2 * np.pi)))**2
      )
  #
  for i in modes:
      for j in range(len(freqs)):
          nlin[i, j] = np.sum(pair_noise(np.abs(beta1 - beta1[i, j]))[(beta1 - beta1[i, j]) != 0])
  return nlin, beta1


def noise_plot(dgd_threshold = 3e-15,):
  formatter = ScalarFormatter()
  formatter.set_scientific(True)
  formatter.set_powerlimits([0, 0])
  rc('text', usetex=True)
  dpi = 300
  grid = False
  
  smf_file = "./input/smf.toml"
  mmf_file = "./input/mmf.toml"
  cf_smf = cfg.load_toml_to_struct(smf_file)
  cf_mmf = cfg.load_toml_to_struct(mmf_file)
  assert(cf_smf.fiber_length == cf_mmf.fiber_length)
  assert(cf_smf.baud_rate == cf_mmf.baud_rate)
  assert(cf_smf.n_channels == cf_mmf.n_channels)
  assert(cf_smf.center_frequency == cf_mmf.center_frequency)
  wdm = pynlin.wdm.WDM(
      spacing=cf_smf.channel_spacing,
      num_channels=cf_smf.n_channels,
      center_frequency=cf_smf.center_frequency
  )
  freqs = wdm.frequency_grid()
  
  nlin_smf = get_nlin(cf_smf, dgd_threshold=dgd_threshold)
  nlin_mmf = get_nlin(cf_mmf, dgd_threshold=dgd_threshold)
  # for each channel, we compute the total number of collisions that
  # needs to be computed for evaluating the total noise on that channel.
  T = 1 / cf_mmf.baud_rate
  L = cf_mmf.fiber_length

  n_lines = 5
  nlin = []
  nlin.append(nlin_mmf)
  nlin.append(nlin_smf)

  ############################
  # GAUSSIAN NOISE
  def plot_nlin(dgd_threshold = 3e-15, 
                use_kappa     = False):
    colors = ["blue", "orange", "green", "red", "gray"]
    #
    plt.clf()
    plt.figure(figsize=(3.6, 3.2))
    nlin.append(nlin_mmf)
    nlin.append(nlin_smf)
    for i in range(n_lines):
        plt.semilogy(freqs * 1e-12, 
                     nlin[i, :] * 1e-30, 
                     lw=1.2,
                     color = colors[i])
    #
    plt.xlabel(r'$f \; [\mathrm{THz}]$')
    plt.ylabel(r'$\mathrm{NLIN} \; [\mathrm{km}^2/\mathrm{ps}^{2}]$')
    plt.legend(labelspacing=0.1)
    plt.grid(grid)
    plt.tight_layout()
    plt.ylim([2e-1, 1e2])
    plt.savefig(f"media/nlin.pdf", dpi=dpi)