import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import h5py
import os
from scipy.interpolate import interp1d
from scipy.integrate import quad
import tqdm
import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
from pynlin.utils import dBm2watt
import pynlin.constellations
import json

f = open("./scripts/sim_config.json")
data = json.load(f)
print(data)
dispersion=data["dispersion"] 
effective_area=data["effective_area"] 
baud_rate=data["baud_rate"] 
fiber_lengths=data["fiber_length"] 
num_channels=data["num_channels"] 
channel_spacing=data["channel_spacing"] 
center_frequency=data["center_frequency"] 
store_true=data["store_true"] 
pulse_shape=data["pulse_shape"] 
partial_collision_margin=data["partial_collision_margin"] 
num_co= data["num_co"] 
num_ct=data["num_ct"]
wavelength=data["wavelength"]
interfering_grid_index=data["interfering_grid_index"]
special = data['special']
num_only_co_pumps=data['num_only_co_pumps']
num_only_ct_pumps=data['num_only_ct_pumps']

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.weight'] = '500'
plt.rcParams['font.size'] = '24'

# PLOTTING PARAMETERS1
#power_dBm_list = [-20, -10, -5, 0]
#power_dBm_list = np.linspace(-20, 0, 3)
power_dBm_list = [-6.0]

# if input("\nSingle channel collision plotter: \n\t>Length= "+str(fibear_lengths)+"km \n\t>power list= "+str(power_dBm_list)+" \n\t>interfering_grid_index= "+str(interfering_grid_index)+"\nAre you sure? (y/[n])") != "y":
#         exit()

for fiber_length in fiber_lengths:
		length_setup = int(fiber_length*1e-3) 
		plot_save_path = "/home/lorenzi/Scrivania/progetti/NLIN/plots_"+str(length_setup)+'/'+str(num_co)+'_co_'+str(num_ct)+'_ct_'+special+'/'
		#
		if not os.path.exists(plot_save_path):
				os.makedirs(plot_save_path)
		#
		results_path = '../results_'+str(length_setup)+'/'
		results_path_bi = '../results_'+str(length_setup)+'/'+str(num_co)+'_co_'+str(num_ct)+'_ct_'+special+'/'
		#
		time_integrals_results_path = '../results/'

		arity_list = [16]


		beta2 = pynlin.utils.dispersion_to_beta2(
				dispersion, wavelength 
		)
		fiber = pynlin.fiber.Fiber(
				effective_area=80e-12,
				beta2=beta2
		)
		wdm = pynlin.wdm.WDM(
				spacing=channel_spacing,
				num_channels=num_channels,
				center_frequency=190e12
		)
		points_per_collision = 10 

		print("beta2: ", fiber.beta2)
		print("gamma: ", fiber.gamma)

		Delta_theta_2_co = np.zeros_like(
				np.ndarray(shape=(len(power_dBm_list), len(arity_list)))
				)
		Delta_theta_2_ct = np.zeros_like(Delta_theta_2_co)
		Delta_theta_2_bi = np.zeros_like(Delta_theta_2_co)



		interfering_list = range(num_channels-1)
		Noise_spacing_co = np.zeros_like(
				np.ndarray(shape=(len(power_dBm_list), len(interfering_list)))
				)
		Noise_spacing_ct = np.zeros_like(Noise_spacing_co)
		Noise_spacing_bi = np.zeros_like(Noise_spacing_co)
		Noise_spacing_none = np.zeros_like(Noise_spacing_co)

		show_flag = False

		for idx, power_dBm in enumerate(power_dBm_list):
				average_power =  dBm2watt(power_dBm)

				# SIMULATION DATA LOAD =================================

				pump_solution_co =     np.load(results_path + 'pump_solution_co_' + str(power_dBm) + '.npy')
				signal_solution_co =   np.load(results_path + 'signal_solution_co_' + str(power_dBm) + '.npy')
				pump_solution_ct =    np.load(results_path + 'pump_solution_ct_' + str(power_dBm) + '.npy')
				signal_solution_ct =  np.load(results_path + 'signal_solution_ct_' + str(power_dBm) + '.npy')
				pump_solution_bi =     np.load(results_path_bi + 'pump_solution_bi_' + str(power_dBm) + '.npy')
				signal_solution_bi =   np.load(results_path_bi + 'signal_solution_bi_' + str(power_dBm) + '.npy')

				#z_max = np.load(results_path + 'z_max.npy')
				f = h5py.File(time_integrals_results_path + '0_9_results.h5', 'r')
				z_max = np.linspace(0, fiber_length, np.shape(pump_solution_ct)[0])

				# COMPUTE THE fB =================================
				pump_solution_co =   np.divide(pump_solution_co, pump_solution_co[0, :])
				signal_solution_co = np.divide(signal_solution_co, signal_solution_co[0, :])
				pump_solution_ct =  np.divide(pump_solution_ct, pump_solution_ct[0, :])
				signal_solution_ct =np.divide(signal_solution_ct, signal_solution_ct[0, :])
				pump_solution_bi =   np.divide(pump_solution_bi, pump_solution_bi[0, :])
				signal_solution_bi = np.divide(signal_solution_bi, signal_solution_bi[0, :])

				# XPM COEFFICIENT EVALUATION, single m =================================
				# compute the collisions between the two furthest WDM channels
				frequency_of_interest = wdm.frequency_grid()[0]
				interfering_frequency = wdm.frequency_grid()[interfering_grid_index]
				single_interference_channel_spacing = interfering_frequency - frequency_of_interest
				print("single_interference_channel_spacing= ", single_interference_channel_spacing)



				# compute the X0mm coefficients given the precompute time integrals
				m = np.array(f['/time_integrals/channel_0/interfering_channel_'+str(interfering_grid_index-1)+'/m'])
				z = np.array(f['/time_integrals/channel_0/interfering_channel_'+str(interfering_grid_index-1)+'/z'])
				I = np.array(f['/time_integrals/channel_0/interfering_channel_'+str(interfering_grid_index-1)+'/integrals'])

				# upper cut z
				z = np.array(list(filter(lambda x: x<=fiber_length, z)))
				print(z)
				I = I[:int(len(m)*(fiber_length/80e3)), :len(z)]
				m = m[:int(len(m)*(fiber_length/80e3))]

				# PLOT A COUPLE OF CHANNEL fB
				select_idx = [0, 49]
				fB_co_1 = interp1d(z_max, signal_solution_co[:, select_idx[0]], kind='linear')
				fB_ct_1 = interp1d(z_max, signal_solution_ct[:, select_idx[0]], kind='linear')
				fB_bi_1 = interp1d(z_max, signal_solution_bi[:, select_idx[0]], kind='linear')

				fB_co_2 = interp1d(z_max, signal_solution_co[:, select_idx[1]], kind='linear')
				fB_ct_2 = interp1d(z_max, signal_solution_ct[:, select_idx[1]], kind='linear')
				fB_bi_2 = interp1d(z_max, signal_solution_bi[:, select_idx[1]], kind='linear')


				##########################
				#### fB
				##########################
				if True:
						fig_fB = plt.figure(figsize=(10, 8))
						ax = fig_fB.add_subplot(1, 1, 1)
						plt.plot(show = show_flag)
						c = cm.viridis(np.linspace(0.1, 0.9, 4),1)
						# format(wdm.frequency_grid()[select_idx[1]]*1e-12, ".1f")
						ax.plot(z*1e-3, fB_co_1(z), color="green",label = "CO",  linewidth = 3, zorder=-1)        
						ax.plot(z*1e-3, fB_ct_1(z), color="blue",label = "CT",  linewidth = 3, zorder=-1)
						ax.plot(z*1e-3, fB_bi_1(z), color="orange",label = "BI",  linewidth = 3, zorder=-1)

		## arc design
						'''
						# Configure arc
						center_x = 50            # x coordinate
						center_y = 2.6          # y coordinate
						radius_1 = 5         # radius 1
						radius_2 = radius_1/ 3           # radius 2 >> for cicle: radius_2 = 2 x radius_1
						angle = 180             # orientation
						theta_1 = 0            # arc starts at this angle
						theta_2 = 272           # arc finishes at this angle
						arc = Arc([center_x, center_y],
										radius_1,
										radius_2,
										angle = angle,
										theta1 = theta_1,
										theta2=theta_2,
										capstyle = 'round',
										linestyle='-',
										lw=3,
										color = 'black', 
										zorder=0)
						# Add arc
						ax.add_patch(arc)
						# Add arrow
						x1 = center_x        # x coordinate
						y1 = center_y+radius_2/2              # y coordinate    
						length_x = 3     # length on the x axis (negative so the arrow points to the left)
						length_y = 0.5        # length on the y axis
						ax.arrow(x1,
										y1,
										length_x,
										length_y,
										head_width=0.1,
										head_length=0.5,
										fc='k',
										ec='k',
										linewidth = 3)
						ax.text(x1+length_x+2.1, y1+length_y, "Copropagating")


						# Configure arc
						center_x = 10            # x coordinate
						center_y = 0.5          # y coordinate
						radius_1 = 5         # radius 1
						radius_2 = radius_1/ 3           # radius 2 >> for cicle: radius_2 = 2 x radius_1
						angle = 180             # orientation
						theta_1 = 5            # arc starts at this angle
						theta_2 = 352           # arc finishes at this angle
						arc = Arc([center_x, center_y],
										radius_1,
										radius_2,
										angle = angle,
										theta1 = theta_1,
										theta2=theta_2,
										capstyle = 'round',
										linestyle='-',
										lw=3,
										color = 'black', 
										zorder=0)
						# Add arc
						ax.add_patch(arc)
						# Add arrow
						x1 = center_x        # x coordinate
						y1 = center_y+radius_2/2              # y coordinate    
						length_x = 3     # length on the x axis (negative so the arrow points to the left)
						length_y = 0.5        # length on the y axis
						ax.arrow(x1,
										y1,
										length_x,
										length_y,
										head_width=0.1,
										head_length=0.5,
										fc='k',
										ec='k',
										linewidth = 3)
						ax.text(x1+length_x+2.1, y1+length_y, "Counterpropagating")

						# Configure arc
						center_x = 10            # x coordinate
						center_y = 0.5          # y coordinate
						radius_1 = 5         # radius 1
						radius_2 = radius_1/ 3           # radius 2 >> for cicle: radius_2 = 2 x radius_1
						angle = 180             # orientation
						theta_1 = 5            # arc starts at this angle
						theta_2 = 352           # arc finishes at this angle
						arc = Arc([center_x, center_y],
										radius_1,
										radius_2,
										angle = angle,
										theta1 = theta_1,
										theta2=theta_2,
										capstyle = 'round',
										linestyle='-',
										lw=3,
										color = 'black', 
										zorder=0)
						# Add arc
						ax.add_patch(arc)
						# Add arrow
						x1 = center_x        # x coordinate
						y1 = center_y+radius_2/2              # y coordinate    
						length_x = 3     # length on the x axis (negative so the arrow points to the left)
						length_y = 0.5        # length on the y axis
						ax.arrow(x1,
										y1,
										length_x,
										length_y,
										head_width=0.1,
										head_length=0.5,
										fc='k',
										ec='k',
										linewidth = 3)
						ax.text(x1+length_x+2.1, y1+length_y, "Bidirectional")
						'''
		##
						plt.grid(which="both")
						plt.xlabel(r"$z$ [km]")
						plt.ylabel(r"$f_B$")
						plt.legend()
						plt.yticks(range(5))
						fig_fB.tight_layout()
						fig_fB.savefig(plot_save_path+'f_B_1'+str(power_dBm)+'.pdf')    


						fig_fB_2 = plt.figure(figsize=(10, 8))
						ax = fig_fB_2.add_subplot(1, 1, 1)
						plt.plot(show = show_flag)
						c = cm.viridis(np.linspace(0.1, 0.9, 4),1)
						# format(wdm.frequency_grid()[select_idx[1]]*1e-12, ".1f")
						ax.plot(z*1e-3, fB_co_2(z), color="green",label = "CO", linewidth = 3, zorder=-1)
						ax.plot(z*1e-3, fB_ct_2(z), color="blue",label = "CT", linewidth = 3, zorder=-1)
						ax.plot(z*1e-3, fB_bi_2(z), color="orange",label = "BI", linewidth = 3, zorder=-1)

		## arc design < insert here eventually
		##
						plt.grid(which="both")
						plt.xlabel("Position [km]")
						plt.ylabel(r"$f_B$")
						plt.legend()
						plt.yticks(range(5))
						fig_fB_2.tight_layout()
						fig_fB_2.savefig(plot_save_path+'f_B_50'+str(power_dBm)+'.pdf')   


				# interpolate the amplification function using optimization results
				fB_co = interp1d(z_max, signal_solution_co[:, interfering_grid_index], kind='linear')
				fB_ct = interp1d(z_max, signal_solution_ct[:, interfering_grid_index], kind='linear')
				fB_bi = interp1d(z_max, signal_solution_bi[:, interfering_grid_index], kind='linear')

				approx = np.ones_like(m) /(beta2 * 2 * np.pi * single_interference_channel_spacing)
				X0mm_co = pynlin.nlin.Xhkm_precomputed(
						z, I, amplification_function=fB_co(z))
				X0mm_ct = pynlin.nlin.Xhkm_precomputed(
						z, I, amplification_function=fB_ct(z))
				X0mm_bi = pynlin.nlin.Xhkm_precomputed(
						z, I, amplification_function=fB_bi(z))
				X0mm_none = pynlin.nlin.Xhkm_precomputed(
						z, I, amplification_function=None)


				# X0mm PLOTTING
				if True:
						locs = pynlin.nlin.get_collision_location(
								m, fiber, single_interference_channel_spacing, 1 / baud_rate)


						##########################
						#### COLLISION SHAPEs
						##########################
						fig1, (ax1) = plt.subplots(nrows=1,sharex=True, sharey=True, figsize=(10,10))
						plt.plot(show = show_flag)

						for i, m_ in enumerate(m[5:-5]):
								i = i+5
								max_I =np.max(np.abs(I[:]))
								ax1.plot(z*1e-3, np.abs(I[i])/max_I * fB_co(z), color=cm.viridis(i/(len(m)-10)/3*2))
								ax1.plot(z*1e-3, fB_co(z), color="purple")
								ax1.axvline(locs[i] * 1e-3, color="grey", linestyle="dashed")
								ax1.axhline(0.5, color="grey", linestyle="dotted")

								# ax2.plot(z*1e-3, np.abs(I[i])/max_I * fB_ct(z), color=cm.viridis(i/(len(m)-10)/3*2))
								# ax2.plot(z*1e-3, fB_ct(z), color="purple")
								# ax2.axvline(locs[i] * 1e-3, color="grey", linestyle="dashed")
								# ax2.axhline(0.5, color="grey", linestyle="dotted")

								# ax3.plot(z*1e-3, np.abs(I[i])/max_I * fB_bi(z), color=cm.viridis(i/(len(m)-10)/3*2))
								# ax3.plot(z*1e-3, fB_bi(z), color="purple")
								# ax3.axvline(locs[i] * 1e-3, color="grey", linestyle="dashed")
								# ax3.axhline(0.5, color="grey", linestyle="dotted")

								# ax4.plot(z*1e-3, np.abs(I[i])/max_I, color=cm.viridis(i/(len(m)-10)/3*2))
								# ax4.plot(z*1e-3, np.ones_like(z), color="purple")
								# ax4.axvline(locs[i] * 1e-3, color="grey", linestyle="dashed")

						# ax4.set_xlabel("Position [km]")
						ax1.set_xlabel(r"$z$ [km]")
						ax1.set_ylabel(r"$s^{-1}$")
						# ax3.set_ylabel(r"$s^{-1}$")
						# ax1.text(70, 2, 'CO', bbox={'facecolor': 'white', 'alpha': 0.8})
						# ax2.text(70, 2, 'CT', bbox={'facecolor': 'white', 'alpha': 0.8})
						# ax3.text(70, 2, 'BI', bbox={'facecolor': 'white', 'alpha': 0.8})
						# ax4.text(70, 2, 'perf.', bbox={'facecolor': 'white', 'alpha': 0.8})

						plt.subplots_adjust(left = 0.1, wspace=0.0, hspace=0.0, right = 9.8/10, top=9.9/10)
						fig1.savefig(plot_save_path+'collision_shape_single_'+str(power_dBm)+'.pdf')

						##########################
						#### COLLISION SHAPEs 50
						##########################
						if True:
								f2 = h5py.File(time_integrals_results_path + '39_49_results.h5', 'r')
								interfering_grid_index = 10

								# compute the X0mm coefficients given the precompute time integrals
								m = np.array(f['/time_integrals/channel_1/interfering_channel_'+str(interfering_grid_index-1)+'/m'])
								z = np.array(f['/time_integrals/channel_1/interfering_channel_'+str(interfering_grid_index-1)+'/z'])
								I = np.array(f['/time_integrals/channel_1/interfering_channel_'+str(interfering_grid_index-1)+'/integrals'])

								# upper cut z
								z = np.array(list(filter(lambda x: x<=fiber_length, z)))
								I = I[:int(len(m)*(fiber_length/80e3)), :len(z)]
								m = m[:int(len(m)*(fiber_length/80e3))]
								print("m=", m)
								#
								locs = pynlin.nlin.get_collision_location(m, fiber, 100e9, 1 / baud_rate)
								interfering_grid_index = 2
								# interpolate the amplification function using optimization results
								fB_co = interp1d(z_max, signal_solution_co[:, interfering_grid_index], kind='linear')
								fB_ct = interp1d(z_max, signal_solution_ct[:, interfering_grid_index], kind='linear')
								fB_bi = interp1d(z_max, signal_solution_bi[:, interfering_grid_index], kind='linear')
								
								fig1, ((ax1, ax2, ax3, ax4)) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,10))
								plt.plot(show = show_flag)

								for i, m_ in enumerate(m[5:-5]):
										i = i+5
										max_I =np.max(np.abs(I[:]))
										ax1.plot(z*1e-3, np.abs(I[i])/max_I * fB_co(z), color=cm.viridis(i/(len(m)-10)/3*2))
										ax1.plot(z*1e-3, fB_co(z), color="purple")
										ax1.axvline(locs[i] * 1e-3, color="grey", linestyle="dashed")
										#ax1.axhline(0.5, color="grey", linestyle="dotted")

										ax2.plot(z*1e-3, np.abs(I[i])/max_I * fB_ct(z), color=cm.viridis(i/(len(m)-10)/3*2))
										ax2.plot(z*1e-3, fB_ct(z), color="purple")
										ax2.axvline(locs[i] * 1e-3, color="grey", linestyle="dashed")
										#ax2.axhline(0.5, color="grey", linestyle="dotted")

										ax3.plot(z*1e-3, np.abs(I[i])/max_I * fB_bi(z), color=cm.viridis(i/(len(m)-10)/3*2))
										ax3.plot(z*1e-3, fB_bi(z), color="purple")
										ax3.axvline(locs[i] * 1e-3, color="grey", linestyle="dashed")
										#ax3.axhline(0.5, color="grey", linestyle="dotted")

										ax4.plot(z*1e-3, np.abs(I[i])/max_I, color=cm.viridis(i/(len(m)-10)/3*2))
										ax4.plot(z*1e-3, np.ones_like(z), color="purple")
										ax4.axvline(locs[i] * 1e-3, color="grey", linestyle="dashed")

								ax4.set_xlabel("Position [km]")
								ax3.set_xlabel("Position [km]")
								ax1.set_ylabel(r"$s^{-1}$")
								ax3.set_ylabel(r"$s^{-1}$")
								# ax1.set_ylim([0, 3])
								ax2.set_ylim([0, 1.5])
								# ax3.set_ylim([0, 1.5])
								# ax4.set_ylim([0, 1.5])
								# ax1.text(70, 3, 'CO', bbox={'facecolor': 'white', 'alpha': 0.8})
								# ax2.text(70, -2, 'CT', bbox={'facecolor': 'white', 'alpha': 0.8})
								# ax3.text(70, 2, 'BI', bbox={'facecolor': 'white', 'alpha': 0.8})
								# ax4.text(70, 2, 'perf.', bbox={'facecolor': 'white', 'alpha': 0.8})
								plt.autoscale()
								plt.subplots_adjust(left = 0.1, wspace=0.0, hspace=0.0, right = 9.8/10, top=9.9/10)
								fig1.savefig(plot_save_path+'collision_shape_last_'+str(power_dBm)+'.pdf')

						##########################
						#### X0mm
						##########################
						fig2 = plt.figure(figsize=(10, 6))
						plt.plot(show = show_flag)

						plt.semilogy(m, np.abs(X0mm_co), marker='x', markersize = 10, ls='dotted', color='green', label="CO")
						plt.semilogy(m, np.abs(X0mm_ct), marker='x', markersize = 10, ls='dashed', color='blue', label="CT")
						plt.semilogy(m, np.abs(X0mm_bi), marker='x', markersize = 10, color='orange', label="BI")

						# plt.semilogy(m, np.abs(X0mm_none), marker="s", color='grey', label="perfect ampl.")
						# plt.semilogy(m, np.abs(approx), marker="s", label="approximation")
						plt.minorticks_on()
						plt.grid(which="both")
						plt.xlabel(r"Collision index $m$")
						plt.ylabel(r"$X_{0,m,m}$ [m/s]")
						# plt.title(
						#     rf"$f_B(z)$, $D={args.dispersion}$ ps/(nm km), $L={args.fiber_length}$ km, $R={args.baud_rate}$ GHz"
						# )
						plt.legend()
						fig2.tight_layout()
						fig2.savefig(plot_save_path+'X0mm_'+str(power_dBm)+'.pdf')

						##########################
						#### X0mm channel 50
						##########################
						if True:
								f2 = h5py.File(time_integrals_results_path + '39_49_results.h5', 'r')
								interfering_grid_index = 10

								# compute the X0mm coefficients given the precompute time integrals
								m = np.array(f['/time_integrals/channel_1/interfering_channel_'+str(interfering_grid_index-1)+'/m'])
								z = np.array(f['/time_integrals/channel_1/interfering_channel_'+str(interfering_grid_index-1)+'/z'])
								I = np.array(f['/time_integrals/channel_1/interfering_channel_'+str(interfering_grid_index-1)+'/integrals'])
								interfering_grid_index = 49
								# interpolate the amplification function using optimization results
								fB_co = interp1d(z_max, signal_solution_co[:, interfering_grid_index], kind='linear')
								fB_ct = interp1d(z_max, signal_solution_ct[:, interfering_grid_index], kind='linear')
								fB_bi = interp1d(z_max, signal_solution_bi[:, interfering_grid_index], kind='linear')
								# upper cut z
								z = np.array(list(filter(lambda x: x<=fiber_length, z)))
								I = I[:int(len(m)*(fiber_length/80e3)), :len(z)]
								m = m[:int(len(m)*(fiber_length/80e3))]
								X0mm_co = pynlin.nlin.Xhkm_precomputed(
										z, I, amplification_function=fB_co(z))
								X0mm_ct = pynlin.nlin.Xhkm_precomputed(
										z, I, amplification_function=fB_ct(z))
								X0mm_bi = pynlin.nlin.Xhkm_precomputed(
										z, I, amplification_function=fB_bi(z))
								X0mm_none = pynlin.nlin.Xhkm_precomputed(
										z, I, amplification_function=None)

								fig2 = plt.figure(figsize=(10, 8))
								plt.plot(show = show_flag)

								plt.semilogy(m, np.abs(X0mm_co), marker='x', markersize = 10, color='green', label="CO")
								plt.semilogy(m, np.abs(X0mm_ct), marker='x', markersize = 10, color='blue', label="CT")
								plt.semilogy(m, np.abs(X0mm_bi), marker='x', markersize = 10, color='orange', label="BI")

								# plt.semilogy(m, np.abs(X0mm_none), marker="s", color='grey', label="perfect ampl.")
								# plt.semilogy(m, np.abs(approx), marker="s", label="approximation")
								plt.minorticks_on()
								plt.grid(which="both")
								plt.xlabel(r"Collision index $m$")
								plt.ylabel(r"$X_{0,m,m}$ [m/s]")
								# plt.title(
								#     rf"$f_B(z)$, $D={args.dispersion}$ ps/(nm km), $L={args.fiber_length}$ km, $R={args.baud_rate}$ GHz"
								# )
								plt.legend()
								fig2.tight_layout()
								fig2.savefig(plot_save_path+'X0mm_last_'+str(power_dBm)+'.pdf')
						
				# FULL X0mm EVALUATION FOR EVERY m and every channel =======================
				X_co = []
				X_co.append(0.0)
				X_ct = []
				X_ct.append(0.0)
				X_bi = []
				X_bi.append(0.0)
				X_none = []
				X_none.append(0.0)

				# compute the first num_channels interferents (assume the WDM grid is identical)
				pbar_description = "Computing space integrals over interfering channels"
				collisions_pbar = tqdm.tqdm(range(1, 50), leave=False)
				collisions_pbar.set_description(pbar_description)
				for interf_index in collisions_pbar:
						fB_co = interp1d(
								z_max, signal_solution_co[:, interf_index], kind='linear')
						X0mm_co = pynlin.nlin.Xhkm_precomputed(
								z, I, amplification_function=fB_co(z))
						X_co.append(np.sum(np.abs(X0mm_co)**2))

						fB_ct = interp1d(
								z_max, signal_solution_ct[:, interf_index], kind='linear')
						X0mm_ct = pynlin.nlin.Xhkm_precomputed(
								z, I, amplification_function=fB_ct(z))
						X_ct.append(np.sum(np.abs(X0mm_ct)**2))

						fB_bi = interp1d(
								z_max, signal_solution_bi[:, interf_index], kind='linear')
						X0mm_bi = pynlin.nlin.Xhkm_precomputed(
								z, I, amplification_function=fB_bi(z))
						X_bi.append(np.sum(np.abs(X0mm_bi)**2))

						fB_none = interp1d(
								z_max, signal_solution_bi[:, interf_index], kind='linear')
						X0mm_none = pynlin.nlin.Xhkm_precomputed(
								z, I, amplification_function=None)
						X_none.append(np.sum(np.abs(X0mm_none)**2))

				# PHASE NOISE COMPUTATION =======================
				# copropagating

				for ar_idx, M in enumerate(arity_list):
						qam = pynlin.constellations.QAM(M)

						qam_symbols = qam.symbols()
						cardinality = len(qam_symbols)

						# assign specific average optical energy
						qam_symbols = qam_symbols / np.sqrt(np.mean(np.abs(qam_symbols)**2)) * np.sqrt(average_power/baud_rate)

						fig4 = plt.figure(figsize=(10, 10))
						plt.plot(show = show_flag)
						plt.scatter(np.real(qam_symbols), np.imag(qam_symbols))
						constellation_variance = (np.mean(np.abs(qam_symbols)**4) - np.mean(np.abs(qam_symbols)**2) **2)
						#print("\tenergy variance      = ", constellation_variance)

						Delta_theta_ch_2_co = 0
						Delta_theta_ch_2_ct = 0
						Delta_theta_ch_2_bi = 0

						for i in range(1, num_channels):
								Delta_theta_ch_2_co = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_co[i])
								Delta_theta_2_co[idx, ar_idx] += Delta_theta_ch_2_co
								Delta_theta_ch_2_ct = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_ct[i])
								Delta_theta_2_ct[idx, ar_idx] += Delta_theta_ch_2_ct
								Delta_theta_ch_2_bi = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_bi[i])
								Delta_theta_2_bi[idx, ar_idx] += Delta_theta_ch_2_bi
						for i in interfering_list:
								Noise_spacing_co[idx, i] = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_co[i])
								Noise_spacing_ct[idx, i] = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_ct[i])
								Noise_spacing_bi[idx, i] = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_bi[i])
								Noise_spacing_none[idx, i] = 4 * fiber.gamma**2 * constellation_variance * np.abs(X_none[i])


						#print("Total phase variance (CO): ", Delta_theta_2_co[idx])
						#print("Total phase variance (CT): ", Delta_theta_2_ct[idx])


		ar_idx = 0 # 16-QAM
		# fig_power = plt.figure(figsize=(10, 5))
		# plt.plot(show = True)
		# plt.semilogy(power_dBm_list, Delta_theta_2_co[:, ar_idx], marker='x', markersize = 10, color='green', label="coprop.")
		# plt.semilogy(power_dBm_list, Delta_theta_2_ct[:, ar_idx], marker='x', markersize = 10,color='blue',label="counterprop.")
		# plt.semilogy(power_dBm_list, Delta_theta_2_ct[:, ar_idx], marker='x', markersize = 10,color='blue',label="bidirection.")
		# plt.minorticks_on()
		# plt.grid(which="both")
		# plt.xlabel(r"Power [dBm]")
		# plt.ylabel(r"$\Delta \theta^2$")
		# plt.legend()
		# fig_power.tight_layout()
		# fig_power.savefig(plot_save_path+"power_noise_strange.pdf")

		idx = 0
		# fig_arity, (ax1,ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10,10))

		# ax1.loglog(arity_list, Delta_theta_2_co[idx, :], marker='x', markersize = 10, color='green', label="coprop.")
		# ax2.loglog(arity_list, Delta_theta_2_ct[idx, :], marker='x', markersize = 10,color='blue',label="counterprop.")
		# plt.subplots_adjust(hspace=0.0)

		# ax1.grid()
		# ax2.grid()
		# ax1.set_ylabel(r"$\Delta \theta^2$")
		# ax2.set_ylabel(r"$\Delta \theta^2$")
		# ax2.legend()
		# ax1.legend()
		# ax2.set_xlabel("QAM modulation arity")
		# ax2.set_xticks(arity_list)

		# fig_arity.tight_layout()
		# fig_arity.savefig(plot_save_path+"arity_noise.pdf")

		fig_spacing, (ax1, ax2,ax3) = plt.subplots(nrows=3, sharex=True, figsize=(9,5))
		interfering_list = range(1, 51)
		fB_co = [interp1d(z_max, signal_solution_co[:, intx-1], kind='linear') for intx in interfering_list]
		fB_ct = [interp1d(z_max, signal_solution_ct[:, intx-1], kind='linear')for intx in interfering_list]
		fB_bi =  [interp1d(z_max, signal_solution_bi[:, intx-1], kind='linear')for intx in interfering_list]
		ax1.plot(interfering_list, [1e-3*quad(fB_intx, 0, fiber_length)[0] for fB_intx in fB_co], marker='x', markersize = 7, color='green', label="coprop.")
		ax2.plot(interfering_list,  [1e-3*quad(fB_intx, 0, fiber_length)[0] for fB_intx in fB_ct], marker='x', markersize = 7,color='blue', label="counterprop.")
		ax3.plot(interfering_list,  [1e-3*quad(fB_intx, 0, fiber_length)[0] for fB_intx in fB_bi], marker='x', markersize = 7,color='orange',label="bidir.")

		plt.subplots_adjust(hspace=0.0)

		ax1.grid()
		ax2.grid()
		ax3.grid()

		ax2.set_ylabel(r"$\int dz f_B(z)$ [km]")
		ax1.text(3, 140, 'CO', bbox={'facecolor': 'white', 'alpha': 0.8})
		ax2.text(3, 33, 'CT', bbox={'facecolor': 'white', 'alpha': 0.8})
		ax3.text(3, 65, 'BI', bbox={'facecolor': 'white', 'alpha': 0.8})

		ax3.set_xlabel("Channel index")
		ax2.set_xticks([1, 10, 20, 30, 40, 50])

		plt.subplots_adjust(left = 0.15, wspace=0.0, hspace=0, right = 9.8/10, top=9.8/10, bottom=0.2)
		fig_spacing.savefig(plot_save_path+"spacing_noise.pdf")
