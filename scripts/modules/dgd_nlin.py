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
from pynlin.utils import watt2dBm

def get_nlin(cf, 
             dgd_threshold = 3e-15, 
             use_kappa = False):
  oi_fit = np.load('results/oi_fit.npy')
  print("WARN: the kappa for the LP01-LP01 should be 1.0, but it is not.") 
  kappa = np.loadtxt('input/kappa.csv', delimiter=',')
  kappa /= kappa[0, 0]
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
      nlin_vec = np.where(
          dgd > dgd_threshold, # this needs to be set carefully
          L / (T * dgd),
          np.sqrt(np.pi) * (L / (T * np.sqrt(2 * np.pi)))**2
      )
      nlin_vec = np.where(
          dgd == 0,
          0,
          nlin_vec
      )
      return nlin_vec
  #
  for i in modes:
      for j in range(len(freqs)):
        if use_kappa:
          print("WARN: we are neglecting the fact that the kappa matrix is not symmetric.")
          weighted_nlin = np.matmul(kappa, pair_noise(np.abs(beta1 - beta1[i, j])))
          # print(weighted_nlin.shape)
          nlin[i, j] = np.sum(weighted_nlin[i])
        else: 
          nlin[i, j] = np.sum(pair_noise(np.abs(beta1 - beta1[i, j])))
  return nlin


def noise_plot(dgd_threshold = 3e-15,
               use_kappa = False, 
               use_smf = False):
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
  print(f"Loading a ITU-T standardized WDM grid \n [spacing: {cf_smf.channel_spacing*1e-9:.3e}GHz, center: {cf_smf.center_frequency*1e-12:.3e}THz] \n")
  assert(cf_smf.fiber_length == cf_mmf.fiber_length)
  assert(cf_smf.baud_rate == cf_mmf.baud_rate)
  assert(cf_smf.n_channels == cf_mmf.n_channels * cf_mmf.n_modes)
  assert(cf_smf.center_frequency == cf_mmf.center_frequency)
  wdm = pynlin.wdm.WDM(
      spacing=cf_smf.channel_spacing,
      num_channels=cf_smf.n_channels,
      center_frequency=cf_smf.center_frequency
  )
  freqs_smf = wdm.frequency_grid()
  wdm = pynlin.wdm.WDM(
      spacing=cf_mmf.channel_spacing,
      num_channels=cf_mmf.n_channels,
      center_frequency=cf_mmf.center_frequency
  )
  freqs_mmf = wdm.frequency_grid()
  
  nlin_smf = get_nlin(cf_smf, dgd_threshold=dgd_threshold, use_kappa=False)
  nlin_mmf = get_nlin(cf_mmf, dgd_threshold=dgd_threshold, use_kappa=use_kappa)
  # for each channel, we compute the total number of collisions that
  # needs to be computed for evaluating the total noise on that channel.
  T = 1 / cf_mmf.baud_rate
  L = cf_mmf.fiber_length
    
  colors = ["blue", "orange", "green", "red", "gray"]
  linestyles = ["-", "-", "-", "-", "--"]
  
  plt.clf()
  plt.figure(figsize=(3.6, 3.2))
  for i in range(cf_mmf.n_modes):
      plt.semilogy(freqs_mmf * 1e-12, 
                   nlin_mmf[i, :] * 1e-30, 
                   lw=1.2,
                   color = colors[i],
                   ls=linestyles[i])
  plt.semilogy(freqs_smf * 1e-12, 
               nlin_smf[0, :] * 1e-30, 
               lw=1.2,
               color = colors[-1],
               ls=linestyles[-1])
  #
  plt.xlabel(r'$f \; [\mathrm{THz}]$')
  plt.ylabel(r'$\mathrm{NLIN} \; [\mathrm{km}^2/\mathrm{ps}^{2}]$')
  # plt.legend(labelspacing=0.1)
  plt.grid(grid)
  plt.tight_layout()
  plt.ylim([2e-1, 1e2])
  plt.savefig(f"media/nlin.pdf", dpi=dpi)
  print("The figure is saved as media/nlin.pdf")
  
  avg_nlin_mmf = np.mean(nlin_mmf)
  avg_nlin_smf = np.mean(nlin_smf)
  print(f"Average NLIN coeff per channel: MMF -> {avg_nlin_mmf:4.3e} | SMF -> {avg_nlin_smf:4.3e}")
  # apply QAM 16 and -10 dBm input power
  gamma = 1.27e-3
  P_in = 0.1e-3
  constellation_factor = 1
  nlin_prefactor = P_in**3 * gamma**2 * constellation_factor / (cf_mmf.baud_rate**2) 
  print(f"Average NLIN power per channel: MMF -> {watt2dBm(nlin_prefactor * avg_nlin_mmf):4.1f} dBm | SMF -> {watt2dBm(nlin_prefactor * avg_nlin_smf):4.1f} dBm")