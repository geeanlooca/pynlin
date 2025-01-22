import numpy as np
from pynlin.nlin import m_th_time_integral_general
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os

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
    beta2b = -75e-27  
    z = np.linspace(0, fiber.length, 1000)
    # 1. the two pulses 
    cases = [(dgd_hi, beta2a, beta2b, m1), 
             (dgd_hi, beta2a, beta2b, m2),]
    I_list = []
    for dgd, beta2a, beta2b, m in cases:
        I = np.real(m_th_time_integral_general(pulse, fiber, wdm, (0, 0), (0, 0), 0.0, m, z, dgd, None, beta2a, beta2b))
        I_list.append(I)

    # 2. the set of pulse peaks:
    zw = 1/(pulse.baud_rate * dgd_hi)
    m_max = fiber.length / zw
    print(f"  > m_max: {m_max}")
    m_axis = -np.array(range(int(round(m_max))))
    peaks   = np.zeros(2)[np.newaxis, :].repeat(len(m_axis), axis=0)
    z_peaks = np.zeros(2)[np.newaxis, :].repeat(len(m_axis), axis=0)
    #
    if not os.path.exists("results/fig1_peaks.npy") or recompute:
      print("  Computing peaks...")
      for im, m in enumerate(m_axis):
        print(f"  m: {m:>10}")
        for ic, (dgd, beta2a, beta2b, _) in enumerate(cases):
          I = np.real(m_th_time_integral_general(pulse, fiber, wdm, (0, 0), (0, 0), 0.0, m, z, dgd_hi, None, beta2a, beta2b))
          peaks[im, ic]   = np.max(I)
          z_peaks[im, ic] = z[np.argmax(I)]
      np.save("results/fig1_peaks.npy", (peaks, z_peaks))
      print("  Done computing and saving peaks.")
    else:
      print("  Loading peaks...")
      peaks, z_peaks = np.load("results/fig1_peaks.npy")
      print("  Done loading peaks.")
    
    
    # 3. Plotting
    cases = [(dgd_hi, beta2a, beta2a, m1), 
             (dgd_lo, beta2b, beta2b, m2),]
    colors = ["blue", "grey"]
    plt.figure(figsize=(4, 3))
    # plotting the pulses
    for I in I_list:
      plt.plot(z*1e-3, I*1e-12, label=f'try', color="red")
    # plotting the peaks
    for ip, (peak, z_peak) in enumerate(zip(peaks, z_peaks)):
      # print(peak, z_peak)
      for ic in range(len(cases)):
        if ip == 0:
          plt.plot(z_peak[ic]*1e-3, peak[ic]*1e-12, 
                  marker="x",
                  markersize=0.1,
                  label='case'+str(ic),
                  color=colors[ic])
        plt.plot(z_peak[ic]*1e-3, peak[ic]*1e-12, 
                marker="x",
                color=colors[ic])
    plt.legend()
    plt.xlabel(r'$z \; [\mathrm{km}]$')
    plt.ylabel(r'$I(z) \; [\mathrm{ps^{-1}}]$')
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig("media/1-quovadis.pdf")
    print("Done plotting Fig.1.")