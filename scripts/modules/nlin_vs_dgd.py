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
import json
rc('text', usetex=True)
logging.basicConfig(filename='MMF_optimizer.log', encoding='utf-8', level=logging.INFO)
log = logging.getLogger(__name__)
log.debug("starting to load sim_config.json")
f = open("./scripts/sim_config.json")

data = json.load(f)
dispersion = data["dispersion"]
effective_area = data["effective_area"]
baud_rate = data["baud_rate"]
fiber_lengths = data["fiber_length"]
num_channels = data["num_channels"]
interfering_grid_index = data["interfering_grid_index"]
channel_spacing = data["channel_spacing"]
center_frequency = data["center_frequency"]
store_true = data["store_true"]
pulse_shape = data["pulse_shape"]
partial_collision_margin = data["partial_collision_margin"]
num_co = data["num_co"]
num_ct = data["num_ct"]
wavelength = data["wavelength"]
special = data["special"]
pump_direction = data["pump_direction"]
num_only_co_pumps = data['num_only_co_pumps']
num_only_ct_pumps = data['num_only_ct_pumps']
gain_dB_setup = data['gain_dB_list']
gain_dB_list = np.linspace(gain_dB_setup[0], gain_dB_setup[1], gain_dB_setup[2])
power_dBm_setup = data['power_dBm_list']
power_dBm_list = np.linspace(power_dBm_setup[0], power_dBm_setup[1], power_dBm_setup[2])
num_modes = data['num_modes']
oi_fit = np.load('oi_fit.npy')
oi_avg = np.load('oi_avg.npy')
use_avg_oi = False


def get_space_integrals(m, z, I):
    '''
      Read the time integral file and compute the space integrals 
    '''
    X0mm = np.zeros_like(m)
    X0mm = X0mm_space_integral(z, I, amplification_function=None)
    return X0mm


beta1_params = load_group_delay()

dpi = 300
grid = False
wdm = pynlin.wdm.WDM(
    spacing=channel_spacing,
    num_channels=num_channels,
    center_frequency=center_frequency
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

dgd = 1e-50
gvd = -1e-30

print(f"Computing the channel-pair NLIN coefficient. DGD = {dgd*1e12:.1e} ps/m, GVD = {gvd*1e27:.1e} ps^2/km ")

if False:
  pulse = GaussianPulse(
      baud_rate=baud_rate,
      num_symbols=1e2,
      samples_per_symbol=2**5,
      rolloff=0.0,
  )
else:
  pulse = NyquistPulse(
      baud_rate=baud_rate,
      num_symbols=1e3,
      samples_per_symbol=2**5,
      rolloff=0.0,
  )

zwL = 1 / (pulse.baud_rate * dgd * dummy_fiber.length)


def nlin_numerics(dgd, gvd):
  a_chan = (-1, -10)
  b_chan = (-1, -100)
  z, I, m = compute_all_collisions_time_integrals(a_chan, b_chan, dummy_fiber, wdm, pulse, dgd, gvd)
  X0mm = get_space_integrals(m, z, I) 
  print(np.abs(X0mm)**2)
  return np.sum(np.abs(X0mm)**2)

T = pulse.T0
L = dummy_fiber.length
# LD = - T**2 / gvd

def antonio(dgd, gvd):
 return L / (T * dgd)

def marco(dgd, gvd):
  return 0.406 * 1e30

def fra_lesser(dgd, gvd):
  if gvd != 0:
    LD = - T**2 / gvd
    return (LD/ (T * np.sqrt(2 * np.pi))* np.arcsinh(L / LD))**2
  else:
    return np.sqrt(np.pi) * (L/(np.sqrt(2*np.pi)*T))**2
    # return (L/(T)*np.sum(np.exp(-np.linspace(-100, 100, 201)**2)/2))**2
    
def fra_novel(dgd, gvd):
  return (0.877/(np.sqrt(2* np.pi)) * L / T)**2
  
def fra_greater(dgd, gvd):
  return fra_lesser(dgd, gvd) * 6.3

if gvd != 0:
    LD = - T**2 / gvd
    print(f"L/LD = zL = {L/LD:.3e}")
print(f"DGD = 0. num = {nlin_numerics(1e-20, gvd):.3e}, \
      fra < = {fra_lesser(0, gvd):.3e}, \
      fra novel = {fra_novel(0, gvd):.3e}")