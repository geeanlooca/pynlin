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
from pynlin.fiber import *
from pynlin.pulses import *
from pynlin.nlin import compute_all_collisions_time_integrals, get_dgd, X0mm_space_integral, get_gvd
from matplotlib.gridspec import GridSpec
import scripts.modules.cfg as cfg


def get_space_integrals(m, z, I):
    '''
      Read the time integral file and compute the space integrals 
    '''
    X0mm = np.zeros_like(m)
    X0mm = X0mm_space_integral(z, I, amplification_function=None)
    return X0mm

def get_fig2():
  rc('text', usetex=True)
  cf = cfg.load_toml_to_struct("./input/config_collision.toml")
  oi_fit = np.load('results/oi_fit.npy')

  beta1_params = load_group_delay()
  wdm = pynlin.wdm.WDM(
      spacing=cf.channel_spacing,
      num_channels=cf.n_channels,
      center_frequency=cf.center_frequency
  )
  freqs = wdm.frequency_grid()
  modes = [0, 1, 2, 3]
  mode_names = ['LP01', 'LP11', 'LP21', 'LP02']

  dummy_fiber = MMFiber(
      effective_area=80e-12,
      overlap_integrals=oi_fit,
      group_delay=beta1_params,
      length=100e3,
      n_modes=4
  )

  beta1 = np.zeros((len(modes), len(freqs)))
  for i in modes:
      beta1[i, :] = dummy_fiber.group_delay.evaluate_beta1(i, freqs)
  beta2 = np.zeros((len(modes), len(freqs)))
  for i in modes:
      beta2[i, :] = dummy_fiber.group_delay.evaluate_beta2(i, freqs)
  beta1 = np.array(beta1)
  beta2 = np.array(beta2)

  dgd1 = 1e-16
  dgd2g = 1e-13
  dgd2n = 1e-15
  n_samples_numeric_g = 10
  n_samples_numeric_n = 3
  px = 0
  gvds = [-35e-27]

  print(
      f"Computing the channel-pair NLIN coefficient insides [{dgd1*1e12:.1e}, {dgd2g*1e12:.1e}] ps/m ")
  for gvd in gvds:
      for px in [0]:
          if px == 0:
              pulse = GaussianPulse(
                  baud_rate=cf.baud_rate,
                  num_symbols=1e2,
                  samples_per_symbol=2**5,
              )
          else:
              pulse = NyquistPulse(
                  baud_rate=cf.baud_rate,
                  num_symbols=1e3,
                  samples_per_symbol=2**5,
                  rolloff=0.0,
              )

          n_samples_analytic = 500
          dgd1 = 1e-17
          if px == 0:
              dgd2 = dgd2g
              n_samples_numeric = n_samples_numeric_g
          else:
              dgd2 = dgd2n
              n_samples_numeric = n_samples_numeric_n
          # 6e-9 for our fiber
          dgds_numeric = np.logspace(np.log10(dgd1), np.log10(dgd2), n_samples_numeric)
          dgds_analytic = np.linspace(dgd1, dgd2, n_samples_analytic)

          zwL_numeric = 1 / (pulse.baud_rate * dgds_numeric * dummy_fiber.length)
          zwL_analytic = 1 / (pulse.baud_rate * dgds_analytic * dummy_fiber.length)

          partial_nlin = np.zeros(n_samples_numeric)
          a_chan = (1, 1)
          b_chan = (1, 2)
          if True:
              for id, dgd in enumerate(dgds_numeric):
                  z, I, m = compute_all_collisions_time_integrals(
                      a_chan, b_chan, dummy_fiber, wdm, pulse, dgd, gvd)
                  # space integrals
                  X0mm = get_space_integrals(m, z, I)
                  partial_nlin[id] = np.sum(X0mm**2)
              if px == 0:
                  np.save("results/partial_nlin_gaussian" +
                          str(gvd) + "B2.npy", partial_nlin)
              else:
                  np.save("results/partial_nlin_nyquist" +
                          str(gvd) + "B2.npy", partial_nlin)

  T = 100e-12
  L = dummy_fiber.length

  # partial_nlin_nyquist = np.load("results/partial_nlin_nyquist.npy")
  # partial_nlin_gaussian = np.load("results/partial_nlin_gaussian.npy")


  LD_eff = pulse.T0**2 / np.abs(gvd)

  dgd2 = dgd2g
  dgds_analytic = np.linspace(dgd1, dgd2, n_samples_analytic)
  analytic_nlin = L / (T * dgds_analytic)
  analytic_nlin[analytic_nlin > 1e30] = np.nan
  n_samples_numeric = 10
  dgds_numeric_g = np.logspace(np.log10(dgd1), np.log10(dgd2g), n_samples_numeric_g)
  n_samples_numeric = 3
  dgds_numeric_n = np.logspace(np.log10(dgd1), np.log10(dgd2n), n_samples_numeric_n)


  dpi = 300
  grid = False
  fig = plt.figure(figsize=(5, 3.5))
  plt.plot(dgds_analytic * 1e12, analytic_nlin * 1e-30, lw=1, color='red')
  # Fra LOWER
  # plt.plot(dgds_analytic * 1e12, np.ones_like(dgds_analytic) *
  #          (LD_eff/ (T * np.sqrt(2 * np.pi)) * np.arcsinh(L / LD_eff))**2 * 1e-30, color='blue', lw=1, label=r'$N^<$')

  plt.plot(dgds_analytic * 1e12, np.ones_like(dgds_analytic) *
          (L / (T * np.sqrt(2 * np.pi)))**2 * 1e-30, color='blue', lw=1, label=r'$N^<$')
  # Fra UPPER
  plt.plot(dgds_analytic * 1e12, np.ones_like(dgds_analytic) * 1.77 * (LD_eff / (T * np.sqrt(2 * np.pi))
          * np.arcsinh(L / LD_eff))**2 * 1e-30, color='blue', lw=1, ls=":", label=r'$N^>$')

  # # Fra Series
  # plt.plot(dgds_analytic * 1e12, np.ones_like(dgds_analytic) * (LD_eff/ (T * np.sqrt(2 * np.pi))
  #          * np.arcsinh(L / LD_eff))**2 * 1e-30, color='blue', ls=":", label=r'$N^>$')
  plt.plot(dgds_analytic * 1e12, np.ones_like(dgds_analytic) *
          0.406, color='green', ls="--", lw=1, label='Marco')
  # plt.scatter(dgds_numeric_g * 1e12, partial_nlin_gaussian * 1e-30,
  #             color='green', label='Gaussian', marker="x")

  # plt.scatter(dgds_numeric_n * 1e12, partial_nlin_nyquist * 1e-30,
  #             color='green', label='Nyq.'+str(-35e-27), marker="x")

  lowest_dgd = 0.0
  for ix, gvd in enumerate(gvds):
      partial_B2g = (np.load("results/partial_nlin_gaussian" + str(gvd) + "B2.npy"))
      if ix == 0:
          lowest_dgd = partial_B2g[0]
      print(partial_B2g)
      plt.scatter(dgds_numeric_g * 1e12, partial_B2g * 1e-30,
                  label='Gauss.' + str(gvd), color="blue", marker="x")
      partial_B2n = (np.load("results/partial_nlin_nyquist" + str(gvd) + "B2.npy"))
      plt.scatter(dgds_numeric_n * 1e12, partial_B2n * 1e-30,
                  label='Nyq.' + str(gvd), color="green", marker="x")

  print(f"DGD low. num = {lowest_dgd:.3e}, ra < = {(L/ (T * np.sqrt(2 * np.pi)))**2:.3e}")

  plt.xlabel('DGD [ps/m]')
  plt.legend()
  plt.yscale('log')
  plt.xscale('log')
  plt.ylabel(r'channel-pair NLIN [km$^2$/ps$^2$]')
  plt.tight_layout()
  plt.savefig(f"media/2-threshold.pdf", dpi=dpi)

if __name__ == "__main__":
  get_fig2()