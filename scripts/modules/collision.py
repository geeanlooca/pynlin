import numpy as np
from pynlin.nlin import m_th_time_integral_general
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
from modules.beta_utils import beta2rms, beta2rms_complementary, beta2avg, beta2avg_complementary
from pynlin.pulses import NyquistPulse, GaussianPulse

formatter = ScalarFormatter()
formatter.set_scientific(True)
formatter.set_powerlimits([0, 0])
 

def plot_illustrative(fiber, wdm, cf, recompute=False):
    """
    Plot of Marco
    """
    print("Plotting Fig.1 (pulse collisions)...")
    
    nyquist_pulse = NyquistPulse(
    baud_rate=cf.baud_rate,
    num_symbols=200, # CHANGING THIS SOLVES THE ALIASING PROBLEM TODO
    samples_per_symbol=10,
    rolloff=0.0,
    )
    gaussian_pulse = GaussianPulse(
        baud_rate=cf.baud_rate,
        num_symbols=5e2, # CHANGING THIS SOLVES THE ALIASING PROBLEM TODO
        samples_per_symbol=2**5,
    )
    plt.figure(figsize=(4, 2.2))
    ls = ["-", "--"]
    for ipulse, pulse in enumerate([gaussian_pulse]):
      m = [-10, -90]
      dgds = [1e-12, 1e-12]
      m1 = -10
      m2 = -90
      dgd_hi = 100e-15
      dgd_lo = 1e-16
      beta2a = -2000e-27  # TODO, check con Marco
      beta2b = -1e-40
      # beta2bar = beta2rms(beta2a, beta2b)
      LDbar = 1/(pulse.baud_rate**2 * np.abs(beta2a))
      z = np.linspace(0, fiber.length, 1000)
      
      # 1. the two pulses
      # cases are related to single collisions
      # assume beta2a = beta2b
      cases = [(dgd_hi, beta2a, beta2a, -10),
               (dgd_hi, beta2a, beta2a, -40),
              #  (dgd_hi, beta2a, beta2a, -50),
               (dgd_hi, beta2a, beta2a, -60),
               (dgd_hi, beta2a, beta2a, -90),]
      I_list = []
      for dgd, beta2a, _, m in cases:
          I = np.real(m_th_time_integral_general(pulse, fiber, wdm, (0, 0), (0, 0), 0.0, m, z, dgd, None, beta2b, beta2rms_complementary(beta2a, beta2b)))
          I_list.append(I)

      # 2. the set of pulse peaks:
      # cases are related to set of parameters, not to single collisions
      # we utilize the beta2rms_complementary function to compute the beta2b value
      print(f"B2A = {beta2a:.2e} | B2B = {beta2b:.2e}  B2complement = {beta2rms_complementary(beta2a, beta2b)}|")
      cases_peaks = [(dgd_hi, beta2a, beta2a, m1), 
                    (dgd_hi, beta2rms_complementary(beta2a, beta2b), beta2b, m2),]
      zw = 1/(pulse.baud_rate * dgd_hi)
      m_max = fiber.length / zw
      print(f"  > m_max: {m_max}")
      m_axis = -np.array(range(int(round(m_max))))
      m_axis = m_axis[::10]
      peaks   = np.zeros(2)[np.newaxis, :].repeat(len(m_axis), axis=0)
      z_peaks = np.zeros(2)[np.newaxis, :].repeat(len(m_axis), axis=0)
      #
      if not os.path.exists("results/fig1_peaks.npy") or recompute:
        print("  Computing peaks...")
        for im, m in enumerate(m_axis):
          print(f"  m: {m:>10}")
          for ic, (dgd, beta2a_example, beta2b_example, _) in enumerate(cases_peaks):
            I = np.real(m_th_time_integral_general(pulse, fiber, wdm, (0, 0), (0, 0), 0.0, m, z, dgd, None, beta2a_example, beta2b_example))
            peaks[im, ic]   = np.max(I)
            z_peaks[im, ic] = z[np.argmax(I)]
        np.save("results/fig1_peaks.npy", (peaks, z_peaks))
        print("  Done computing and saving peaks.")
      else:
        print("  Loading peaks...")
        peaks, z_peaks = np.load("results/fig1_peaks.npy")
        print("  Done loading peaks.")
      print(peaks)
      
      # 3. case of very low DGD (almost zero)
      I_low = np.real(m_th_time_integral_general(pulse, fiber, wdm, (0, 0), (0, 0), 0.0, 0, z, 1e-20, None, beta2a, beta2a))
      I_low_2 = np.real(m_th_time_integral_general(pulse, fiber, wdm, (0, 0), (0, 0), 0.0, 0, z, 1e-20, None, beta2b, beta2rms_complementary(beta2a, beta2b)))
      
      # 4. Plotting
      colors  = ["grey", "blue"]
      markers = ["x", "o"]
      marker_sizes = [5, 3]
      lw = 0.2
      # plotting the peaks
      for ip, (peak, z_peak) in enumerate(zip(peaks, z_peaks)):
        for ic in range(len(cases_peaks)):
          if ip == 0:
            plt.plot(z_peak[ic]/LDbar, 
                    peak[ic]/pulse.baud_rate, 
                    marker=markers[ic],
                    markersize=marker_sizes[ic],
                    label='case'+str(ic),
                    color=colors[ic],
                    linewidth=lw)
          else:
            plt.plot(z_peak[ic]/LDbar, 
                    peak[ic]/pulse.baud_rate,  
                    markersize=marker_sizes[ic], 
                    marker=markers[ic],
                    color=colors[ic],
                    linewidth=lw)
      # plotting the low DGD case
      plt.plot(z/LDbar, 
              I_low / pulse.baud_rate, 
              label='low DGD', 
              color="green", 
              linewidth=lw,
              linestyle=ls[ipulse])
      plt.plot(z/LDbar, 
              I_low_2 / pulse.baud_rate, 
              label='low DGD', 
              color="orange", 
              linewidth=lw,
              linestyle=ls[ipulse])
      # plotting the pulses
      for ii, I in enumerate(I_list):
        if ii == 0:
          plt.plot(z/LDbar, 
                  I / pulse.baud_rate, 
                  label=f'try', 
                  color="red",
                  linewidth=lw,
                  linestyle=ls[ipulse], )
        else:
          print(len(z))
          plt.plot(z/LDbar, 
                  I / pulse.baud_rate, 
                  color="red",
                  linewidth=lw,
                  linestyle=ls[ipulse], )
    # plt.legend()
    plt.xlabel(r'$z / \bar{L_D}$')
    plt.ylabel(r'$I(z) \cdot T $')
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig("media/1-quovadis.pdf")
    print("Done plotting Fig.1.")