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
from pynlin.utils import watt2dBm, dBm2watt

def get_nlin_prefactor(cf):
  gamma = 1.3e-3
  P_in = dBm2watt(-1.5)
  constellation_factor = 0.32
  nlin_prefactor = P_in**3 * gamma**2 * constellation_factor / (cf.baud_rate**2) 
  return nlin_prefactor

def get_nlin(cf, 
             dgd_threshold = 3e-15, 
             use_kappa = False, 
             use_fB = False):
  oi_fit = np.load('results/oi_fit.npy')
  print("WARN: the kappa for the LP01-LP01 should be 1.0, but it is not.") 
  kappa = np.loadtxt('input/kappa.csv', delimiter=',')
  kappa /= kappa[0, 0]
  
  if cf.n_modes == 1:
    print("WARN: not doing the right thing for SMF fB")
    signal_powers = np.load("results/signal_power.npy").repeat(4, axis=1) 
  else:
    solutions = np.load("results/ct_solution-5_gain_0.0.npy", allow_pickle=True).item()
    signal_powers = solutions['signal_sol']
    
  signal_powers_swp = np.swapaxes(signal_powers, 1, 2)
  assert(cf.n_modes == signal_powers_swp.shape[1])
  assert(cf.n_channels == signal_powers_swp.shape[2])
  fB = signal_powers_swp / signal_powers_swp[0]
  print(fB.shape)
  # exit()
  z_axis = np.linspace(0, cf.fiber_length, len(fB))
  dz = z_axis[1] - z_axis[0]
  # coeffs = np.polyfit(z_axis, fB, 6)
  coeffs = np.apply_along_axis(lambda sig: np.polyfit(z_axis, sig, deg=6), axis=0, arr=fB)
  print(coeffs.shape)
  def fB_function(z, mode, freq):
    return np.polyval(coeffs[:, mode, freq], z)
  if use_fB:
    high_dgd_fB = (np.sum(fB, axis=0) * dz / cf.fiber_length)**2
    low_dgd_fB  = np.sum(fB**2, axis=0) * dz / cf.fiber_length
  else:
    high_dgd_fB = 1.0
    low_dgd_fB  = 1.0
    
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
          L / (T * dgd) * high_dgd_fB,
          np.sqrt(np.pi) * (L / (T * np.sqrt(2 * np.pi)))**2 * low_dgd_fB
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
               use_smf = False, 
               use_fB = False,
               use_dBm_scale = False):
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
  
  nlin_mmf = get_nlin(cf_mmf, 
                      dgd_threshold=dgd_threshold, 
                      use_kappa=use_kappa,
                      use_fB=use_fB)
  nlin_smf = get_nlin(cf_smf, 
                      dgd_threshold=dgd_threshold, 
                      use_kappa=False, 
                      use_fB=use_fB)
  # for each channel, we compute the total number of collisions that
  # needs to be computed for evaluating the total noise on that channel.
  T = 1 / cf_mmf.baud_rate
  L = cf_mmf.fiber_length
    
  colors = ["blue", "orange", "green", "red", "gray"]
  linestyles = ["-", "-", "-", "-", "--"]
  labels = ["LP01", "LP1", "LP11", "LP11", "SMF(LP01)"]
  
  nlin_prefactor = get_nlin_prefactor(cf_mmf)
  if use_dBm_scale:
    y_rescale = nlin_prefactor
    ylabel = r'$P_\mathrm{NLIN} \; [\mathrm{dBm}]$'
    plot_function = plt.plot
    y_function = lambda x: watt2dBm(x * nlin_prefactor)
  else:
    ylabel = r'$\mathrm{NLIN} \; [\mathrm{km}^2/\mathrm{ps}^{2}]$'
    plot_function = plt.semilogy
    y_function = lambda x: x * 1e-30
    
  plt.clf()
  plt.figure(figsize=(3.6, 3.2))
  for i in range(cf_mmf.n_modes):
      plot_function(freqs_mmf * 1e-12, 
                   y_function(nlin_mmf[i, :]),
                   lw=1.2,
                   color = colors[i],
                   ls=linestyles[i],
                   label=labels[i])
  if use_smf:
    plot_function(freqs_smf * 1e-12, 
                y_function(nlin_smf[0, :]), 
                lw=1.2,
                color = colors[-1],
                ls=linestyles[-1],
                label=labels[-1])
  #
  plt.xlabel(r'$f \; [\mathrm{THz}]$')
  plt.ylabel(ylabel)
  # plt.legend(labelspacing=0.1)
  plt.grid(grid)
  plt.tight_layout()
  # plt.ylim([2e-2, 1e0])
  plt.savefig(f"media/nlin.pdf", dpi=dpi)
  print("The figure is saved as media/nlin.pdf")
 
  functions = [np.mean, np.median, np.max, np.min]
  function_names = ["mean  ", "median", "max   ", "min   "]
  for foo, name in zip(functions, function_names): 
    avg_nlin_mmf = foo(nlin_mmf)
    avg_nlin_smf = foo(nlin_smf)
    print(name+f" NLIN coeff per channel: MMF -> {avg_nlin_mmf:4.3e} | SMF -> {avg_nlin_smf:4.3e}")
    # apply QAM 16 and -10 dBm input power
    print(name + f" NLIN power per channel: MMF -> {watt2dBm(nlin_prefactor * avg_nlin_mmf):4.1f} dBm | SMF -> {watt2dBm(nlin_prefactor * avg_nlin_smf):4.1f} dBm")
    print("-"*20)


def noise_histogram(dgd_threshold = 3e-15,
               use_kappa = False, 
               use_smf = False, 
               use_fB = False,
               use_dBm_scale = False):
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
  
  nlin_mmf = get_nlin(cf_mmf, 
                      dgd_threshold=dgd_threshold, 
                      use_kappa=use_kappa,
                      use_fB=use_fB)
  nlin_smf = get_nlin(cf_smf, 
                      dgd_threshold=dgd_threshold, 
                      use_kappa=False, 
                      use_fB=use_fB)
  # for each channel, we compute the total number of collisions that
  # needs to be computed for evaluating the total noise on that channel.
  T = 1 / cf_mmf.baud_rate
  L = cf_mmf.fiber_length
    
  colors = ["blue", "orange", "green", "red", "gray"]
  linestyles = ["-", "-", "-", "-", "--"]
  labels = ["LP01", "LP1", "LP11", "LP11", "SMF(LP01)"]
  
  nlin_prefactor = get_nlin_prefactor(cf_mmf)
  if True:
    y_rescale = nlin_prefactor
    ylabel = r'$\mathrm{n.\;  of\; channels}$'
    plot_function = plt.hist
    y_function = lambda x: watt2dBm(x * nlin_prefactor)
  else:
    ylabel = r'$\mathrm{NLIN} \; [\mathrm{km}^2/\mathrm{ps}^{2}]$'
    plot_function = plt.hist
    y_function = lambda x: x * 1e-30
    
  plt.clf()
  plt.figure(figsize=(3.6, 2.8))
  y_data = y_function(nlin_mmf)
  y_extremes = [np.min(y_data), np.max(y_data)]
  n_bins = 30
  bin_width = (y_extremes[1] - y_extremes[0]) / n_bins
  bins = np.arange(y_extremes[0], y_extremes[1] + bin_width, bin_width)
  # n_bins=[15, 8, 8, 8, 8]
  alpha=0.2
  alpha2=0.9
  lw = 1.5
  for i in range(cf_mmf.n_modes):
      plot_function(y_data[i, :],
                    bins=bins,
                    histtype='stepfilled',
                    alpha=alpha,
                    lw=lw,
                    color=colors[i],)
      plot_function(y_data[i, :],
                    bins=bins,
                    histtype='step',
                    lw=lw,
                    alpha=alpha2,
                    color=colors[i],)
  if use_smf:
    print(nlin_smf.shape)
    plot_function(y_function(nlin_smf[0, :]), 
                  bins=bins,
                  histtype='stepfilled',
                  alpha=alpha,
                  lw=lw,
                  color=colors[-1],)
    plot_function(y_function(nlin_smf[0, :]), 
                  bins=bins,
                  histtype='step',
                  lw=lw,
                  alpha=alpha2,
                  linestyle='--',
                  color=colors[-1],)
  #
  plt.xlabel(r'$P_\mathrm{NLIN} \; [\mathrm{dBm}]$')
  plt.ylabel(ylabel)
  plt.yscale('log')
  # plt.legend(labelspacing=0.1)
  plt.grid(grid)
  plt.tight_layout()
  # plt.ylim([2e-2, 1e0])
  plt.savefig(f"media/6-noise.pdf", dpi=dpi)
  print("The figure is saved as media/6-noise.pdf")
 