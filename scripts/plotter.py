# THIS SCRIPT PLOTS
# NLIN vs Power
# OSNR vs Power
# ASE vs Power
# NLIN ASE comparison
# NLIN vs channel
# OSNR vs channel
# ASE vs channel

import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import h5py
import math
import os
from scipy.interpolate import interp1d
import tqdm
import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
from pynlin.fiber import Fiber
from pynlin.utils import dBm2watt, watt2dBm
from pynlin.wdm import WDM
import pynlin.constellations
from scipy import optimize
from scipy.special import erfc
import json

f = open("/home/lorenzi/Scrivania/progetti/NLIN/PyNLIN/scripts/sim_config.json")
data = json.load(f)
print(data)
dispersion=data["dispersion"] 
effective_area=data["effective_area"] 
baud_rate=data["baud_rate"] 
fiber_length=data["fiber_length"] 
channel_count=data["channel_count"] 
channel_spacing=data["channel_spacing"] 
center_frequency=data["center_frequency"] 
store_true=data["store_true"] 
pulse_shape=data["pulse_shape"] 
partial_collision_margin=data["partial_collision_margin"] 
num_co= data["num_co"] 
num_cnt=data["num_cnt"]
wavelength=data["wavelength"]

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.weight'] = '500'
plt.rcParams['font.size'] = '24'

length_setup = int(fiber_length*1e-3) 
plot_save_path = "/home/lorenzi/Scrivania/progetti/NLIN/plots_"+str(length_setup)+'/'+str(num_co)+'_co_'+str(num_cnt)+'_cnt/'
#
if not os.path.exists(plot_save_path):
    os.makedirs(plot_save_path)
#
results_path = '../results_'+str(length_setup)+'/'
results_path_bi = '../results_'+str(length_setup)+'/'+str(num_co)+'_co_'+str(num_cnt)+'_cnt/'
#
time_integrals_results_path = '../results/'

def H(n):
    s = 0
    n = int(n)
    for i in range(n):
        s += 1/(i+1)
    return s

def NLIN(n, a, b):
    return [a*(2*H(np.min([xxx, 50-xxx+1])-1)+H(50) - H(2*np.min([xxx, 50-xxx+1])))+b for xxx in n]

def OSNR_to_EVM(osnr):
    osnr = 10**(osnr/10)
    M =64
    i_range = [1+item for item in range(int(np.floor(np.sqrt(M))))]
    beta = [2*ii - 1 for ii in i_range]
    alpha = [3*ii*osnr/(2*(M-1)) for ii in beta]
    gamma = [1-ii/np.sqrt(M) for ii in i_range]
    sum1 = np.sum(np.multiply(gamma, list(map(np.exp, [-1*aa for aa in alpha])))  )
    sum2 = np.sum(np.multiply(np.multiply(gamma,beta), [erfc(np.sqrt(aa)) for aa in alpha]))
    

    return np.sqrt(np.divide(1, osnr) - np.sqrt(np.divide(96/np.pi/(M-1), osnr))*sum1 + sum2 )

def EVM_to_BER(evm):
    M = 64
    L = 8
    return (1-1/L)/np.log2(L) * erfc( np.sqrt((3*np.log2(L)*np.sqrt(2)) / ((L**2-1) * np.power(evm, 2) * np.log2(M))))

# PLOTTING PARAMETERS
interfering_grid_index = 1
#power_dBm_list = [-20, -10, -5, 0]
power_dBm_list = np.linspace(-20, 0, 11)
arity_list =[64]
coi_list = [0, 9, 19, 29, 39, 49]

wavelength = 1550

beta2 = -pynlin.utils.dispersion_to_beta2(
    dispersion * 1e-12 / (1e-9 * 1e3), wavelength * 1e-9
)
fiber = pynlin.fiber.Fiber(
    effective_area=80e-12,
    beta2=beta2
)
wdm = pynlin.wdm.WDM(
    spacing=channel_spacing * 1e-9,
    num_channels=num_channels,
    center_frequency=190
)
points_per_collision = 10

print("beta2: ", fiber.beta2)
print("gamma: ", fiber.gamma)
print(wdm.frequency_grid())
Delta_theta_2_co = np.zeros_like(
    np.ndarray(shape=(len(coi_list), len(power_dBm_list), len(arity_list)))
)
Delta_theta_2_cnt = np.zeros_like(Delta_theta_2_co)
Delta_theta_2_bi = np.zeros_like(Delta_theta_2_co)
Delta_theta_2_none =  np.zeros_like(Delta_theta_2_co)

show_flag = False
compute_X0mm_space_integrals = True

if input("\nX0mm and noise variance plotter: \n\t>Length= "+str(fiber_length*1e-3)+"km \n\t>power list= "+str(power_dBm_list)+" \n\t>coi_list= "+str(coi_list)+"\n\t>compute_X0mm_space_integrals= "+str(compute_X0mm_space_integrals)+"\nAre you sure? (y/[n])") != "y":
    exit()

f_0_9 = h5py.File(time_integrals_results_path + '0_9_results.h5', 'r')
f_19_29 = h5py.File(time_integrals_results_path + '19_29_results.h5', 'r')
f_39_49 = h5py.File(time_integrals_results_path + '39_49_results.h5', 'r')
# print(np.array(f_39_49['/time_integrals/channel_1/interfering_channel_2/m']))
# print(np.array(f_19_29['/time_integrals/channel_1/interfering_channel_1/m']))
# print(np.array(f_39_49['/time_integrals/channel_1/interfering_channel_1/integrals']))

# f = h5py.File('merged_time_integrals.h5', 'w')
# f.create_group('time_integrals')
# h5py.copy(f_0_9['time_integrals'], f)
# h5py.copy(f_19_29['time_integrals'], f)
# h5py.copy(f_39_49['time_integrals'], f)
if compute_X0mm_space_integrals:
    # sum of all the varianced (variances are the sum of all the X0mm over m)
    X_co = np.zeros_like(
        np.ndarray(shape=(len(coi_list), len(power_dBm_list)))
    )
    X_cnt = np.zeros_like(X_co)
    X_bi = np.zeros_like(X_co)
    X_none = np.zeros_like(X_co)
    ase_co = np.zeros_like(X_co)
    ase_cnt = np.zeros_like(X_co)
    ase_bi = np.zeros_like(X_co)

    for pow_idx, power_dBm in enumerate(power_dBm_list):
        print("Computing power ", power_dBm)
        average_power = dBm2watt(power_dBm)
        # SIMULATION DATA LOAD =================================

        pump_solution_co = np.load(
            results_path + 'pump_solution_co_' + str(power_dBm) + '.npy')
        signal_solution_co = np.load(
            results_path + 'signal_solution_co_' + str(power_dBm) + '.npy')
        pump_solution_cnt = np.load(
            results_path + 'pump_solution_cnt_' + str(power_dBm) + '.npy')
        signal_solution_cnt = np.load(
            results_path + 'signal_solution_cnt_' + str(power_dBm) + '.npy')
        pump_solution_bi = np.load(
            results_path_bi + 'pump_solution_bi_' + str(power_dBm) + '.npy')
        signal_solution_bi = np.load(
            results_path_bi + 'signal_solution_bi_' + str(power_dBm) + '.npy')
    
        # ASE power evolution
        ase_solution_co = np.load(
            results_path + 'ase_solution_co_' + str(power_dBm) + '.npy')
        ase_solution_cnt = np.load(
            results_path + 'ase_solution_cnt_' + str(power_dBm) + '.npy')
        ase_solution_bi = np.load(
            results_path_bi + 'ase_solution_bi_' + str(power_dBm) + '.npy')

        # compute fB squaring
        pump_solution_co =    np.divide(pump_solution_co, pump_solution_co[0, :])
        signal_solution_co =  np.divide(signal_solution_co, signal_solution_co[0, :])
        pump_solution_cnt =   np.divide(pump_solution_cnt, pump_solution_cnt[0, :])
        signal_solution_cnt = np.divide(signal_solution_cnt, signal_solution_cnt[0, :])
        pump_solution_bi =    np.divide(pump_solution_bi, pump_solution_bi[0, :])
        signal_solution_bi =  np.divide(signal_solution_bi, signal_solution_bi[0, :])

        #z_max = np.load(results_path + 'z_max.npy')
        #f = h5py.File(results_path + 'results_multi.h5', 'r')
        z_max = np.linspace(0, fiber_length, np.shape(pump_solution_cnt)[0])

        # compute the X0mm coefficients given the precompute time integrals
        # FULL X0mm EVALUATION FOR EVERY m =======================
        for coi_idx, coi in enumerate(coi_list):

            print("Computing Channel Of Interest ", coi + 1)

            # compute the first num_channels interferents (assume the WDM grid is identical)
            interfering_frequencies = pynlin.nlin.get_interfering_frequencies(
                coi, wdm.frequency_grid())
            pbar_description = "Computing space integrals"
            collisions_pbar = tqdm.tqdm(range(np.shape(signal_solution_co)[1])[
                                        0:num_channels - 1], leave=False)
            collisions_pbar.set_description(pbar_description)
            for incremental, interf_index in enumerate(collisions_pbar):
                #print("interfering channel : ", incremental)
                if coi == 0:
                    m = np.array(
                        f_0_9['/time_integrals/channel_0/interfering_channel_' + str(incremental) + '/m'])
                    z = np.array(
                        f_0_9['/time_integrals/channel_0/interfering_channel_' + str(incremental) + '/z'])
                    I = np.array(
                        f_0_9['/time_integrals/channel_0/interfering_channel_' + str(incremental) + '/integrals'])
                elif coi == 9:
                    m = np.array(
                        f_0_9['/time_integrals/channel_1/interfering_channel_' + str(incremental) + '/m'])
                    z = np.array(
                        f_0_9['/time_integrals/channel_1/interfering_channel_' + str(incremental) + '/z'])
                    I = np.array(
                        f_0_9['/time_integrals/channel_1/interfering_channel_' + str(incremental) + '/integrals'])
                elif coi == 19:
                    m = np.array(
                        f_19_29['/time_integrals/channel_0/interfering_channel_' + str(incremental) + '/m'])
                    z = np.array(
                        f_19_29['/time_integrals/channel_0/interfering_channel_' + str(incremental) + '/z'])
                    I = np.array(
                        f_19_29['/time_integrals/channel_0/interfering_channel_' + str(incremental) + '/integrals'])
                elif coi == 29:
                    m = np.array(
                        f_19_29['/time_integrals/channel_1/interfering_channel_' + str(incremental) + '/m'])
                    z = np.array(
                        f_19_29['/time_integrals/channel_1/interfering_channel_' + str(incremental) + '/z'])
                    I = np.array(
                        f_19_29['/time_integrals/channel_1/interfering_channel_' + str(incremental) + '/integrals'])
                elif coi == 39:
                    m = np.array(
                        f_39_49['/time_integrals/channel_0/interfering_channel_' + str(incremental) + '/m'])
                    z = np.array(
                        f_39_49['/time_integrals/channel_0/interfering_channel_' + str(incremental) + '/z'])
                    I = np.array(
                        f_39_49['/time_integrals/channel_0/interfering_channel_' + str(incremental) + '/integrals'])
                elif coi == 49:
                    m = np.array(
                        f_39_49['/time_integrals/channel_1/interfering_channel_' + str(incremental) + '/m'])
                    z = np.array(
                        f_39_49['/time_integrals/channel_1/interfering_channel_' + str(incremental) + '/z'])
                    I = np.array(
                        f_39_49['/time_integrals/channel_1/interfering_channel_' + str(incremental) + '/integrals'])

                # upper cut z
                z = np.array(list(filter(lambda x: x<=fiber_length, z)))
                I = I[:int(len(m)*(fiber_length/80e3)), :len(z)]
                m = m[:int(len(m)*(fiber_length/80e3))]

                fB_co = interp1d(
                    z_max, signal_solution_co[:, incremental], kind='linear')
                X0mm_co = pynlin.nlin.Xhkm_precomputed(
                    z, I, amplification_function=fB_co(z))
                X_co[coi_idx, pow_idx] += (np.sum(np.abs(X0mm_co)**2))

                fB_cnt = interp1d(
                    z_max, signal_solution_cnt[:, incremental], kind='linear')
                X0mm_cnt = pynlin.nlin.Xhkm_precomputed(
                    z, I, amplification_function=fB_cnt(z))
                X_cnt[coi_idx, pow_idx] += (np.sum(np.abs(X0mm_cnt)**2))

                fB_bi = interp1d(
                    z_max, signal_solution_bi[:, incremental], kind='linear')
                X0mm_bi = pynlin.nlin.Xhkm_precomputed(
                    z, I, amplification_function=fB_bi(z))
                X_bi[coi_idx, pow_idx] += (np.sum(np.abs(X0mm_bi)**2))

                X0mm_none = pynlin.nlin.Xhkm_precomputed(
                    z, I, amplification_function=None)
                X_none[coi_idx, pow_idx] += (np.sum(np.abs(X0mm_none)**2))
                #print(X_co)
                #print(X_cnt)
                #print(X_none)
            print("\ncomputing channel: ", coi_idx, "\n\n")
            print(ase_solution_co[-1, coi_idx])
            ase_co[coi_idx,pow_idx] = ase_solution_co[-1, coi_idx]
            ase_cnt[coi_idx,pow_idx] = ase_solution_cnt[-1, coi_idx]
            ase_bi[coi_idx,pow_idx] = ase_solution_bi[-1, coi_idx]
            print(ase_solution_co[-1, coi_idx])
            print(ase_solution_cnt[-1, coi_idx])
            print(ase_solution_bi[-1, coi_idx])


    print(ase_co)
    np.save("X_co.npy", X_co)
    np.save("X_cnt.npy", X_cnt)
    np.save("X_bi.npy", X_bi)
    np.save("X_none.npy", X_none)
    np.save("ase_co.npy", ase_co)
    np.save("ase_cnt.npy", ase_cnt)
    np.save("ase_bi.npy", ase_bi)

else:
    X_co = np.load("X_co.npy")
    X_cnt = np.load("X_cnt.npy")
    X_bi = np.load("X_bi.npy")
    X_none = np.load("X_none.npy")
    ase_co = np.load("ase_co.npy")
    ase_cnt = np.load("ase_cnt.npy")
    ase_bi = np.load("ase_bi.npy")
ar_idx = 0  # 16-QAM
M = 64
for pow_idx, power_dBm in enumerate(power_dBm_list):
    average_power = dBm2watt(power_dBm)
    qam = pynlin.constellations.QAM(M)
    qam_symbols = qam.symbols()

    # assign specific average optical energy
    qam_symbols = qam_symbols / np.sqrt(np.mean(np.abs(qam_symbols)**2)) * np.sqrt(average_power / baud_rate)
    constellation_variance = (np.mean(np.abs(qam_symbols)**4) - np.mean(np.abs(qam_symbols)**2) ** 2)

    # print("X co: ", np.sum(np.abs(X_co)))
    # print("X cnt: ", np.sum(np.abs(X_cnt)))
    # print("X bi: ", np.sum(np.abs(X_bi)))
    # print("X none: ", np.sum(np.abs(X_none)))

    for coi_idx, coi in enumerate(coi_list):
        Delta_theta_2_co[coi_idx, pow_idx, ar_idx] = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_co[coi_idx, pow_idx])
        Delta_theta_2_cnt[coi_idx, pow_idx, ar_idx] = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_cnt[coi_idx, pow_idx])
        Delta_theta_2_bi[coi_idx, pow_idx, ar_idx] = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_bi[coi_idx, pow_idx])
        Delta_theta_2_none[coi_idx, pow_idx, ar_idx] = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_none[coi_idx, pow_idx])

    # print("delta co: ", Delta_theta_2_co)
    # print("delta cnt: ", Delta_theta_2_cnt)
    # print("delta none: ", Delta_theta_2_none)



##############################
## NOISE VS POWER
##############################
markers = ["x", "+", "o", "o", "x", "+"]

fig_power, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(14,10))

plt.plot(show=True)
coi_selection = [0, 19, 49]
coi_selection_idx = [0, 2, 5]
for scan in range(len(coi_selection)):
    ax1.plot(power_dBm_list, 10* np.log10(Delta_theta_2_co[coi_selection_idx[scan], :, ar_idx])+power_dBm_list, marker=markers[scan],
                markersize=10, color='green', label="ch." + str(coi_selection[scan]) + " co.")
    ax2.plot(power_dBm_list, 10* np.log10(Delta_theta_2_cnt[coi_selection_idx[scan], :, ar_idx])+power_dBm_list, marker=markers[scan],
                markersize=10, color='blue', label="ch." + str(coi_selection[scan]) + " count.")
    ax3.plot(power_dBm_list, 10* np.log10(Delta_theta_2_bi[coi_selection_idx[scan], :, ar_idx])+power_dBm_list, marker=markers[scan],
                markersize=10, color='orange', label="ch." + str(coi_selection[scan]+1))
    ax4.plot(power_dBm_list, 10* np.log10(Delta_theta_2_none[coi_selection_idx[scan], :, ar_idx])+power_dBm_list, marker=markers[scan],
                markersize=10, color='grey', label="ch." + str(coi_selection[scan]+1))
ax1.grid(which="both")
#plt.annotate("ciao", (0, 0))
ax2.grid(which="both")
ax3.grid(which="both")
ax4.grid(which="both")

ax1.set_ylabel(r"NLIN [dBm]")
ax2.set_ylabel(r"NLIN [dBm]")
ax3.set_ylabel(r"NLIN [dBm]")
ax4.set_ylabel(r"NLIN [dBm]")
ax3.set_xlabel(r"Power [dBm]")
ax4.set_xlabel(r"Power [dBm]")

ax1.text(-15, -20, 'CO', bbox={'facecolor': 'white', 'alpha': 0.8})
ax2.text(-15, -40, 'CNT', bbox={'facecolor': 'white', 'alpha': 0.8})
ax3.text(-15, -40, 'BI', bbox={'facecolor': 'white', 'alpha': 0.8})
ax4.text(-5, -80, 'perf.', bbox={'facecolor': 'white', 'alpha': 0.8})
ax4.legend()
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()
plt.subplots_adjust(wspace=0.0, hspace=0, right = 8.5/10, top=9.9/10)
fig_power.savefig(plot_save_path+"noise_power.pdf")

#############################
##   ASE vd POWER
#############################
print(ase_bi)
fig_ase, (ax1, ax2, ax3) = plt.subplots(nrows= 3, sharex = True, figsize=(10, 10))
plt.plot(show=True)
coi_selection = [0, 19, 49]
coi_selection_idx = [0, 2, 5]
for scan in range(len(coi_selection)):
    ax1.plot(power_dBm_list, 10*np.log10(ase_co[coi_selection_idx[scan], :])+30, marker=markers[scan],
                markersize=10, color='green', label="ch." + str(coi_selection[scan]) + " co.")
    ax2.plot(power_dBm_list, 10*np.log10(ase_cnt[coi_selection_idx[scan], :])+30, marker=markers[scan],
                markersize=10, color='blue', label="ch." + str(coi_selection[scan]) + " count.")
    ax3.plot(power_dBm_list,10*np.log10(ase_bi[coi_selection_idx[scan], :])+30, marker=markers[scan],
                markersize=10, color='orange', label="ch." + str(coi_selection[scan]+1))
ax1.grid(which="both")
ax3.grid(which="both")
ax3.grid(which="both")
ax4.grid(which="both")

ax3.set_xlabel(r"Power [dBm]")
ax4.set_xlabel(r"Power [dBm]")

ax1.set_ylabel(r"ASE noise [dBm]")
ax2.set_ylabel(r"ASE noise [dBm]")
ax3.set_ylabel(r"ASE noise [dBm]")
ax3.legend()
plt.minorticks_on()
plt.subplots_adjust(wspace=0.0, hspace=0, right = 9.8/10, top=9.9/10)
fig_ase.savefig(plot_save_path+"ase_noise_vs_power.pdf")


########################################################
##  NLIN AND ASE COMPARISON (average over the channels)
########################################################
fig_comparison, (ax1, ax2, ax3) = plt.subplots(nrows= 3, sharex = True, figsize=(10, 7))
plt.plot(show=True)
coi_selection = [0, 19, 49]
coi_selection_idx = [0, 2, 5]
axis_num = 0
ax1.plot(power_dBm_list, power_dBm_list+10*np.log10(np.average([Delta_theta_2_co[coi_idx, :, ar_idx] for coi_idx in coi_selection_idx], axis=axis_num)) , marker=markers[0],
            markersize=10, color='green', label="NLIN")
ax2.plot(power_dBm_list, power_dBm_list+10*np.log10(np.average([Delta_theta_2_cnt[coi_idx, :, ar_idx] for coi_idx in coi_selection_idx], axis=axis_num)), marker=markers[0],
            markersize=10, color='blue', label="NLIN")
ax3.plot(power_dBm_list, power_dBm_list+10*np.log10(np.average([Delta_theta_2_bi[coi_idx, :, ar_idx] for coi_idx in coi_selection_idx], axis=axis_num)), marker=markers[0],
            markersize=10, color='orange', label="NLIN")
ax1.plot(power_dBm_list, 30+10*np.log10(np.average([ase_co[coi_idx, :] for coi_idx in coi_selection_idx], axis=axis_num)), marker=markers[2],
            markersize=10, color='green', label="ASE")
ax2.plot(power_dBm_list, 30+10*np.log10(np.average([ase_cnt[coi_idx, :] for coi_idx in coi_selection_idx], axis=axis_num)), marker=markers[2],
            markersize=10, color='blue', label="ASE")
ax3.plot(power_dBm_list, 30+10*np.log10(np.average([ase_bi[coi_idx, :] for coi_idx in coi_selection_idx], axis=axis_num)), marker=markers[2],
            markersize=10, color='orange', label="ASE")
ax1.grid(which="both")
#plt.annotate("ciao", (0, 0))
ax2.grid(which="both")
ax3.grid(which="both")

plt.xlabel(r"Power [dBm]")
plt.minorticks_on()
ax2.set_ylabel(r"Noise power [dBm]")
ax1.minorticks_on()
ax2.minorticks_on()
ax3.minorticks_on()
ax1.text(-20, -20, 'CO', bbox={'facecolor': 'white', 'alpha': 0.8})
ax2.text(-20, -55, 'CNT', bbox={'facecolor': 'white', 'alpha': 0.8})
ax3.text(-20, -55, 'BI', bbox={'facecolor': 'white', 'alpha': 0.8})

ax3.legend()
leg = ax3.get_legend()
leg.legendHandles[0].set_color('grey')
leg.legendHandles[1].set_color('grey')
plt.subplots_adjust(wspace=0.0, hspace=0, right = 9.8/10, top=9.9/10)
fig_comparison.savefig(plot_save_path+"comparison.pdf")


#####################################
## CHANNEL POWER AND OSNR
#####################################


selected_power = -10
pow_idx = np.where(power_dBm_list==selected_power)[0]
P_B = 10**(selected_power/10) # average power of the constellation in mW
T = (1/10e9)
# Delta_theta is the average 
xdata = [1, 10, 20, 30, 40, 50]
ydata= Delta_theta_2_none[:, pow_idx, ar_idx][:, 0]*P_B
print(xdata)
print(ydata)
alpha, beta = optimize.curve_fit(NLIN, xdata=xdata , ydata=ydata)[0]
print(f'alpha={alpha}, beta={beta}')
print("1/beta_2 Omega_0", 1/beta2/(channel_spacing*1e9))
full_coi = [i+1 for i in range(50)]
fig_channel, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(14,10))
plt.plot(show=True)
ax1.plot(coi_list, 10* np.log10(Delta_theta_2_co[:, pow_idx, ar_idx]*P_B), marker='x', markersize=15, color='green', label="ch." + str(coi) + "CO")
plt.grid(which="both")
ax2.plot(coi_list, 10*np.log10(Delta_theta_2_cnt[:, pow_idx, ar_idx]*P_B), marker='x', markersize=15, color='blue', label="ch." + str(coi) + "CNT.")
plt.grid(which="both")
ax3.plot(coi_list, 10*np.log10(Delta_theta_2_bi[:, pow_idx, ar_idx]*P_B), marker='x', markersize=15, color='orange', label="ch." + str(coi) + "BI")
plt.grid(which="both")
ax4.plot(coi_list, 10*np.log10(Delta_theta_2_none[:, pow_idx, ar_idx]*P_B), marker='x', markersize=15, color='grey', label="ch." + str(coi) + "perf.")
ax4.plot(range(50), 10*np.log10(NLIN(full_coi, alpha, beta)), color='red', linestyle = "dashed", linewidth = 2)
plt.grid(which="both")
# ax1.yaxis.set_major_locator(plt.MaxNLocator(5))

ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()
plt.xticks(ticks=coi_list, labels=[k+1 for k in coi_list])
ax1.grid(which="both")
ax2.grid(which="both")
ax3.grid(which="both")
ax4.set_xlabel(r"Channel index")
ax3.set_xlabel(r"Channel index")

ax3.xaxis.set_label(r"Channel index")
plt.xlabel(r"Channel index")
ax4.grid(which="both")
ax1.text(30, -46, 'CO', bbox={'facecolor': 'white', 'alpha': 0.8})
ax2.text(30, -59.5, 'CNT', bbox={'facecolor': 'white', 'alpha': 0.8})
ax3.text(30, -56, 'BI', bbox={'facecolor': 'white', 'alpha': 0.8})
ax4.text(30, -52, 'perf.', bbox={'facecolor': 'white', 'alpha': 0.8})
ax1.set_ylabel(r"NLIN [dBm]")
ax2.set_ylabel(r"NLIN [dBm]")
ax4.set_ylabel(r"NLIN [dBm]")
ax3.set_ylabel(r"NLIN [dBm]")
plt.autoscale(True)
plt.subplots_adjust(left = 0.15, wspace=0.0, hspace=0, right = 8.5/10, top=9.5/10)

fig_channel.savefig(plot_save_path+"noise_channel.pdf")


fig_channel, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(14,10))
plt.plot(show=True)
ax1.plot(coi_list, selected_power - 10* np.log10(Delta_theta_2_co[:, pow_idx, ar_idx]*P_B), marker='x', markersize=15, color='green', label="ch." + str(coi) + "CO")
plt.grid(which="both")
ax2.plot(coi_list, selected_power -10*np.log10(Delta_theta_2_cnt[:, pow_idx, ar_idx]*P_B), marker='x', markersize=15, color='blue', label="ch." + str(coi) + "CNT.")
plt.grid(which="both")
ax3.plot(coi_list, selected_power -10*np.log10(Delta_theta_2_bi[:, pow_idx, ar_idx]*P_B), marker='x', markersize=15, color='orange', label="ch." + str(coi) + "BI")
plt.grid(which="both")
ax4.plot(coi_list, selected_power -10*np.log10(Delta_theta_2_none[:, pow_idx, ar_idx]*P_B), marker='x', markersize=15, color='grey',  label="ch." + str(coi) + "perf.")
ax4.plot(range(50), selected_power-10*np.log10(NLIN(full_coi, alpha, beta)), color='red', linestyle = "dashed", linewidth = 2)
plt.grid(which="both")
# ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
# ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
# ax3.yaxis.set_major_locator(plt.MaxNLocator(5))
# ax4.yaxis.set_major_locator(plt.MaxNLocator(5))

# ax1.yaxis.set_minor_locator(plt.MaxNLocator(2))
# ax2.yaxis.set_minor_locator(plt.MaxNLocator(2))
# ax3.yaxis.set_minor_locator(plt.MaxNLocator(2))
# ax4.yaxis.set_minor_locator(plt.MaxNLocator(2))
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()
plt.xticks(ticks=coi_list, labels=[k+1 for k in coi_list])
ax1.grid(which="both")
ax2.grid(which="both")
ax3.grid(which="both")
ax3.xaxis.set_label(r"Channel index")
ax4.xaxis.set_label(r"Channel index")

ax4.grid(which="both")

ax1.set_ylabel(r"OSNR$_{NLIN}$ [dB]")
ax2.set_ylabel(r"OSNR$_{NLIN}$ [dB]")
ax4.set_ylabel(r"OSNR$_{NLIN}$ [dB]")
ax3.set_ylabel(r"OSNR$_{NLIN}$ [dB]")
ax1.text(30,  36, 'CO', bbox={'facecolor': 'white', 'alpha': 0.8})
ax2.text(30, 49.5, 'CNT', bbox={'facecolor': 'white', 'alpha': 0.8})
ax3.text(30, 46, 'BI', bbox={'facecolor': 'white', 'alpha': 0.8})
ax4.text(30, 42, 'perf.', bbox={'facecolor': 'white', 'alpha': 0.8})
plt.autoscale(True)
plt.subplots_adjust(left = 0.15, wspace=0.0, hspace=0, right = 8.5/10, top=9.5/10)

fig_channel.savefig(plot_save_path+"channel_osnr.pdf")

##########################
## ASE NOISE VS CHANNEL
##########################
fig_ase_channel, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10,6))

plt.plot(show=True)
ax1.plot(coi_list, 10* np.log10(ase_co[:, pow_idx]), marker='s', markersize=10, color='green', label="ch." + str(coi) + "CO")
plt.grid(which="both")
ax2.plot(coi_list, 10* np.log10(ase_cnt[:, pow_idx]), marker='s', markersize=10, color='blue', label="ch." + str(coi) + "count.")
plt.grid(which="both")
ax3.plot(coi_list, 10* np.log10(ase_bi[:, pow_idx]), marker='s', markersize=10, color='orange', label="ch." + str(coi) + "bi.")
plt.grid(which="both")
# ax3.plot(coi_list, 10* np.log10(ase_none[:, pow_idx]*P_B), marker='s', markersize=10, color='orange', label="ch." + str(coi) + "bi.")
# plt.grid(which="both")
plt.xlabel(r"Channel index")
plt.xticks(ticks=coi_list, labels=[k+1 for k in coi_list])
ax1.grid(which="both")
ax2.grid(which="both")
ax3.grid(which="both")

ax1.set_ylabel(r"$\Delta \theta^2$")
ax2.set_ylabel(r"$\Delta \theta^2$")
ax3.set_ylabel(r"$\Delta \theta^2$")
plt.subplots_adjust(left = 0.2, wspace=0.0, hspace=0, right = 9.8/10, top=9.9/10)
fig_ase_channel.savefig(plot_save_path+"ase_channel_noise.pdf")




###################################
## OSNR vs POWER
###################################
fig_powsnr, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(14,10))

plt.plot(show=True)
coi_selection = [0, 19, 49]
coi_selection_idx = [0, 2, 5]
power_list = list(map(dBm2watt, power_dBm_list))

for scan in range(len(coi_selection)):
    osnr_co =   power_dBm_list -10*np.log10(power_list*Delta_theta_2_co[coi_selection_idx[scan], :, ar_idx]   + ase_co[coi_selection_idx[scan], :])-30
    osnr_cnt =  power_dBm_list -10*np.log10(power_list*Delta_theta_2_cnt[coi_selection_idx[scan], :, ar_idx]  + ase_cnt[coi_selection_idx[scan],:])-30
    osnr_bi =   power_dBm_list -10*np.log10(power_list*Delta_theta_2_bi[coi_selection_idx[scan], :, ar_idx]   + ase_bi[coi_selection_idx[scan], :])-30
    osnr_none = power_dBm_list -10*np.log10(power_list*Delta_theta_2_none[coi_selection_idx[scan], :, ar_idx])-30
    ax1.plot(power_dBm_list, osnr_co, marker=markers[scan],
                markersize=10, color='green', label="ch." + str(coi_selection[scan]) + " co.")
    ax2.plot(power_dBm_list,osnr_cnt, marker=markers[scan],
                markersize=10, color='blue', label="ch." + str(coi_selection[scan]) + " count.")
    ax3.plot(power_dBm_list, osnr_bi, marker=markers[scan],
                markersize=10, color='orange', label="ch." + str(coi_selection[scan]+1))
    ax4.plot(power_dBm_list,osnr_none , marker=markers[scan],
                markersize=10, color='grey', label="ch." + str(coi_selection[scan]+1))
ax1.grid(which="both")
#plt.annotate("ciao", (0, 0))
ax2.grid(which="both")
ax3.grid(which="both")
ax4.grid(which="both")

ax1.set_ylabel(r"$OSNR$ [dB]")
ax2.set_ylabel(r"$OSNR$ [dB]")
ax3.set_ylabel(r"$OSNR$ [dB]")
ax4.set_ylabel(r"$OSNR$ [dB]")
ax3.set_xlabel(r"Power [dBm]")
ax4.set_xlabel(r"Power [dBm]")

ax1.text(-15,  25, 'CO', bbox={'facecolor': 'white', 'alpha': 0.8})
ax2.text(-15, 30, 'CNT', bbox={'facecolor': 'white', 'alpha': 0.8})
ax3.text(-15, 25, 'BI', bbox={'facecolor': 'white', 'alpha': 0.8})
ax4.text(-15, 25, 'perf.', bbox={'facecolor': 'white', 'alpha': 0.8})
ax4.legend()
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()

plt.subplots_adjust(wspace=0.0, hspace=0, right = 8.5/10, top=9.9/10)
fig_powsnr.savefig(plot_save_path+"osnr_vs_power.pdf")

####################################
## EVM AND BER
####################################


fig_ber, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(16,8))
coi_selection = [0, 19, 49]
coi_selection_idx = [0, 2, 5]
power_list = list(map(dBm2watt, power_dBm_list))
for scan in range(len(coi_selection)):
    osnr_co =   power_dBm_list -10*np.log10(power_list*Delta_theta_2_co[coi_selection_idx[scan], :, ar_idx]   + ase_co[coi_selection_idx[scan], :])-30
    osnr_cnt =  power_dBm_list -10*np.log10(power_list*Delta_theta_2_cnt[coi_selection_idx[scan], :, ar_idx]  + ase_cnt[coi_selection_idx[scan],:])-30
    osnr_bi =   power_dBm_list -10*np.log10(power_list*Delta_theta_2_bi[coi_selection_idx[scan], :, ar_idx]   + ase_bi[coi_selection_idx[scan], :])-30
    osnr_none = power_dBm_list -10*np.log10(power_list*Delta_theta_2_none[coi_selection_idx[scan], :, ar_idx])-30
    osnr_list = [osnr_co, osnr_cnt, osnr_bi, osnr_none]
    evms = []
    bers = []
    for osnr in osnr_list:
        evms.append(list(map(OSNR_to_EVM, osnr)))
        bers.append(list(map(EVM_to_BER, map(OSNR_to_EVM, osnr))))
    
    ax1.semilogy(power_dBm_list, bers[0], marker=markers[scan],
                markersize=10, color='green', label="ch." + str(coi_selection[scan]) + " co.")
    ax2.semilogy(power_dBm_list, bers[1], marker=markers[scan],
                markersize=10, color='blue', label="ch." + str(coi_selection[scan]) + " count.")
    ax3.semilogy(power_dBm_list, bers[2], marker=markers[scan],
                markersize=10, color='orange', label="ch." + str(coi_selection[scan]+1))
    ax4.semilogy(power_dBm_list, bers[3], marker=markers[scan],
                markersize=10, color='grey', label="ch." + str(coi_selection[scan]+1))
ax1.grid(which="both")
#plt.annotate("ciao", (0, 0))
ax2.grid(which="both")
ax3.grid(which="both")
ax4.grid(which="both")

ax1.set_ylabel(r"BER")
ax2.set_ylabel(r"BER")
ax3.set_ylabel(r"BER")
ax4.set_ylabel(r"BER")
ax3.set_xlabel(r"Power [dBm]")
ax4.set_xlabel(r"Power [dBm]")
ax1.set_ylim([10**-50,1])
ax2.set_ylim([10**-50,1])
ax3.set_ylim([10**-50,1])
ax4.set_ylim([10**-50,1])

ax1.text(-15, 10**(-6), 'CO', bbox={'facecolor': 'white', 'alpha': 0.8})
ax2.text(-15, 10**(-6), 'CNT', bbox={'facecolor': 'white', 'alpha': 0.8})
ax3.text(-15, 10**(-6), 'BI', bbox={'facecolor': 'white', 'alpha': 0.8})
ax4.text(-10, 10**(-6), 'perf.', bbox={'facecolor': 'white', 'alpha': 0.8})
ax4.legend()
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()
plt.subplots_adjust(wspace=0.0, hspace=0, right = 8.5/10, top=9.9/10)
fig_ber.savefig(plot_save_path+"BER_power.pdf")



fig_ber, ax1 = plt.subplots(nrows=1, sharex=True, figsize=(8,6))

coi_selection = [0, 19, 49]
coi_selection_idx = [0, 2, 5]
power_list = list(map(dBm2watt, power_dBm_list))
for scan in range(len(coi_selection)):
    osnr_co =   power_dBm_list -10*np.log10(power_list*Delta_theta_2_co[coi_selection_idx[scan], :, ar_idx]   + ase_co[coi_selection_idx[scan], :])-30
    osnr_list = [osnr_co]
    evms = []
    bers = []
    for osnr in osnr_list:
        evms.append(list(map(OSNR_to_EVM, osnr)))
        bers.append(list(map(EVM_to_BER, map(OSNR_to_EVM, osnr))))
    
    ax1.semilogy(power_dBm_list[-3:], bers[0][-3:], marker=markers[scan],
                markersize=10, color='green', label="ch." + str(coi_selection[scan]+1) + " co.")
ax1.grid(which="both")
ax1.set_ylabel(r"BER")
ax1.set_xlabel(r"Power [dBm]")
plt.legend()

ax1.text(-15, 10**(-6), 'CO', bbox={'facecolor': 'white', 'alpha': 0.8})
plt.subplots_adjust(left=0.2, bottom=0.15, wspace=0.0, hspace=0, right = 9.5/10, top=9.8/10)
plt.tight_layout()
fig_ber.savefig(plot_save_path+"BER_power_zoom.pdf")