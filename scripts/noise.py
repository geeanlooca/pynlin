"""
Finds the overall noise on a single channel 
given a fiber link configuration.

Generates figs:

- Statistics of channels over DGD
- NLIN thresholding and approximation

TODO: 

- Final computation of the noise including the fB

"""
import pynlin.fiber
from scripts.modules.space_integrals_general import *
from scripts.modules.time_integrals import do_time_integrals
from scripts.modules.load_fiber_values import *
import pynlin.fiber
import pynlin.wdm
import pynlin.pulses
from scripts.modules.load_fiber_values import load_group_delay
from modules import cfg
from scripts.modules.fig1 import plot_illustrative
 
cf = cfg.load_toml_to_struct("./input/config_collision.toml")
oi_fit = np.load('results/oi_fit.npy')
oi_avg = np.load('results/oi_avg.npy')

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

freqs = wdm.frequency_grid()

s_limit = 1460e-9
l_limit = 1625e-9
s_freq = 3e8 / s_limit
l_freq = 3e8 / l_limit

# print(s_freq * 1e-12)
# print(l_freq * 1e-12)
delta = (s_freq - l_freq) * 1e-12
avg = ((s_freq + l_freq) * 1e-12 / 2)

fig_to_generate = [1, 2]
if 1 in fig_to_generate:
  ### Manually set the DGD and beta2 values for both the channels
  plot_illustrative(fiber, 
                    wdm, 
                    cf,
                    recompute=True)
if 2 in fig_to_generate:
  ### Manually set the DGD and beta2 values for both the channels
  # plot_illustrative()
  pass