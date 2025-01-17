"""
script for finding the overall noise on a single channel 
given a fiber link configuration consisting 
"""
import pynlin.fiber
from scripts.modules.space_integrals_general import *
from scripts.modules.time_integrals import do_time_integrals
from scripts.modules.load_fiber_values import *
import matplotlib.pyplot as plt
import scipy
import pynlin.fiber
from pynlin.nlin import m_th_time_integral_general
import pynlin.wdm
import pynlin.pulses
from scripts.modules.load_fiber_values import load_group_delay
from modules import cfg
from matplotlib.ticker import ScalarFormatter
formatter = ScalarFormatter()
formatter.set_scientific(True)
formatter.set_powerlimits([0, 0])

def plot_illustrative(fiber, pulse, wdm):
    """
    Plot an illustrative case of the time integrals.
    """
    m = [-10, -90]
    dgds = [1e-12, 1e-12]
    m1 = -10
    m2 = -90
    dgd_hi = 1e-12
    dgd_lo = 1e-16
    beta2a = -35e-27  
    beta2b = -35e-27  
    z = np.linspace(0, fiber.length, 1000)
    # cases=[(dgd, beta2a, beta2b, m), 
    cases = [(dgd_hi, beta2a, beta2b, m1), 
             (dgd_hi, beta2a, beta2b, m2),]
    I_list = []
    for dgd, beta2a, beta2b, m in cases:
        I = np.real(m_th_time_integral_general(pulse, fiber, wdm, (0, 0), (0, 0), 0.0, m, z, dgd, None, beta2a, beta2b))
        I_list.append(I)
    I = np.real(m_th_time_integral_general(pulse, fiber, wdm, (0, 0), (0, 0), 0.0, m1, z, dgd, None, beta2a, beta2b))
    
    plt.figure(figsize=(4, 3))
    for I in I_list:
      plt.plot(z*1e-3, I*1e-12, label=f'try')
    plt.xlabel(r'$z \; [\mathrm{km}]$')
    plt.ylabel(r'$I(z) \; [\mathrm{ps^{-1}}]$')
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig("media/quo_vadis.pdf")
    

cf = cfg.load_toml_to_struct("./input/config_collision.toml")
oi_fit = np.load('oi_fit.npy')
oi_avg = np.load('oi_avg.npy')

beta2 = -pynlin.utils.dispersion_to_beta2(
    cf.dispersion, 1550e-9
)

wdm = pynlin.wdm.WDM(
    spacing=cf.channel_spacing,
    num_channels=cf.n_channels,
    center_frequency=cf.center_frequency
)

fiber = pynlin.fiber.MMFiber(
      effective_area=80e-12,
      overlap_integrals = oi_fit,
      group_delay = load_group_delay(),
      length=cf.fiber_length,
      n_modes = 4
  )

# print(f"beta_2 = {beta2:.9e}")

# make the time integral take as an input (pulse, fiber, wdm)
pulse = pynlin.pulses.GaussianPulse(
    baud_rate=cf.baud_rate,
    num_symbols=1e2,
    samples_per_symbol=2**5,
    rolloff=0.0,
)

freqs = wdm.frequency_grid()

s_limit = 1460e-9
l_limit = 1625e-9
s_freq = 3e8 / s_limit
l_freq = 3e8 / l_limit

# print(s_freq * 1e-12)
# print(l_freq * 1e-12)
delta = (s_freq - l_freq) * 1e-12
avg = ((s_freq + l_freq) * 1e-12 / 2)
# beta_file = './results/fitBeta.mat'
# mat = scipy.io.loadmat(beta_file)['fitParams'] * 1.0

# beta_file = './results/fitBeta.mat'
# omega = 2 * np.pi * freqs
# omega_norm = scipy.io.loadmat(beta_file)['omega_std']
# omega_n = (omega - scipy.io.loadmat(beta_file)['omega_mean']) / omega_norm
# beta1 = np.zeros((4, len(freqs)))

# write the results file in ../results/general_results.h5 with the correct time integrals
# the file contains, for each interferent channel, the values (z, m, I) of the z
# channel of interest is set to channel 0, and interferent channel index start from 0 for simplicity
# a_chan = (0, 0)
# print("@@@@@@@@ Time integrals  @@@@@@@@")
# # do_time_integrals(a_chan, fiber, wdm, pulse, overwrite=True)
# print("@@@@@@@@ Space integrals @@@@@@@@")
# b_chan = (1, 99)
# print("dgd: ", pynlin.nlin.get_dgd(a_chan, b_chan, fiber, wdm))

### Manually set the DGD and beta2 values for both the channels
plot_illustrative(fiber, pulse, wdm)
# compare_interferent(fiber, wdm, pulse, dgd=dgd, beta2a=beta2a, beta2b=beta2b)