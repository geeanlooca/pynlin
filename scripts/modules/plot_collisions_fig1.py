import numpy as np
from pynlin.nlin import m_th_time_integral_general
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
from modules.beta_utils import beta2rms, beta2_complementary

formatter = ScalarFormatter()
formatter.set_scientific(True)
formatter.set_powerlimits([0, 0])
 

def plot_illustrative(fiber, pulse, wdm, recompute=False):
    """
    Plot of Marco
    """
    print("Plotting Fig.1 (pulse collisions)...")
    m = [-10, -90]
    dgds = [1e-12, 1e-12]
    m1 = -10
    m2 = -90
    dgd_hi = 100e-15
    dgd_lo = 1e-16
    beta2a = -35e-27  # TODO, check con Marco
    beta2b = -25e-27
    beta2bar = beta2rms(beta2a, beta2b)
    LDbar = 1/(pulse.baud_rate**2 * np.abs(beta2bar))
    z = np.linspace(0, fiber.length, 1000)
    
    # 1. the two pulses 
    # cases are related to single collisions
    cases = [(dgd_hi, beta2a, beta2b, m1), 
             (dgd_hi, beta2a, beta2b, m2),]
    I_list = []
    for dgd, beta2a, beta2b, m in cases:
        I = np.real(m_th_time_integral_general(pulse, fiber, wdm, (0, 0), (0, 0), 0.0, m, z, dgd, None, beta2a, beta2b))
        I_list.append(I)

    # 2. the set of pulse peaks:
    # cases are related to set of parameters, not to single collisions
    print(f"B2A = {beta2a:.2e} | B2B = {beta2b:.2e}  B2complement = {beta2_complementary(beta2a, beta2b)}|")
    cases_peaks = [(dgd_hi, beta2a, beta2a, m1), 
                   (dgd_hi, beta2b, beta2_complementary(beta2a, beta2b), m2),]
    zw = 1/(pulse.baud_rate * dgd_hi)
    m_max = fiber.length / zw
    print(f"  > m_max: {m_max}")
    m_axis = -np.array(range(int(round(m_max))))
    m_axis = m_axis[::5]
    peaks   = np.zeros(2)[np.newaxis, :].repeat(len(m_axis), axis=0)
    z_peaks = np.zeros(2)[np.newaxis, :].repeat(len(m_axis), axis=0)
    #
    if not os.path.exists("results/fig1_peaks.npy") or recompute:
      print("  Computing peaks...")
      for im, m in enumerate(m_axis):
        print(f"  m: {m:>10}")
        for ic, (dgd, beta2a, beta2b, _) in enumerate(cases_peaks):
          I = np.real(m_th_time_integral_general(pulse, fiber, wdm, (0, 0), (0, 0), 0.0, m, z, dgd, None, beta2a, beta2b))
          peaks[im, ic]   = np.max(I)
          z_peaks[im, ic] = z[np.argmax(I)]
      np.save("results/fig1_peaks.npy", (peaks, z_peaks))
      print("  Done computing and saving peaks.")
    else:
      print("  Loading peaks...")
      peaks, z_peaks = np.load("results/fig1_peaks.npy")
      print("  Done loading peaks.")
    print(peaks)
    
    
    # 3. Plotting
    colors  = ["blue", "grey"]
    markers = ["x", "o"]
    lw = 0.5
    size = 5
    plt.figure(figsize=(4, 3))
    # plotting the pulses
    for I in I_list:
      plt.plot(z/LDbar, I / pulse.baud_rate, label=f'try', color="red")
    # plotting the peaks
    for ip, (peak, z_peak) in enumerate(zip(peaks, z_peaks)):
      for ic in range(len(cases_peaks)):
        if ip == 0:
          plt.plot(z_peak[ic]/LDbar, 
                   peak[ic]/pulse.baud_rate, 
                  marker=markers[ic],
                  markersize=size,
                  label='case'+str(ic),
                  color=colors[ic],
                  linewidth=lw)
        else:
          plt.plot(z_peak[ic]/LDbar, 
                  peak[ic]/pulse.baud_rate,  
                  markersize=size, 
                  marker=markers[ic],
                  color=colors[ic],
                  linewidth=lw)
    plt.legend()
    plt.xlabel(r'$z / \bar{L_D}$')
    plt.ylabel(r'$I(z) \cdot T $')
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig("media/1-quovadis.pdf")
    print("Done plotting Fig.1.")