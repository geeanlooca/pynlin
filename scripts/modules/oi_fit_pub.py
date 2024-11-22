import scipy.io
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.cm import viridis
import seaborn as sns
import os
from pynlin.utils import oi_law_fit, oi_law

# Ensure output directories exist
os.makedirs('media/oi', exist_ok=True)

rc('text', usetex=True)

# Load data
oi_file = 'oi.mat'
mat = scipy.io.loadmat(oi_file)
oi_full = mat['OI'] * 1e12
wl = mat['wavelenght_array'][0] * 1e-9

# Average over polarizations
oi = np.ndarray((21, 21, 4, 4))
oi_avg = np.ndarray((4, 4))
oi_max = np.zeros_like(oi_avg)
oi_min = np.zeros_like(oi_avg)


def expression(l1i, l1f, params):
    Lambda = (l1f - l1i)
    a1, a2, b1, b2, c, x = params
    integral = Lambda * ((a1 + a2) * (l1f**3 - l1i**3) / 3 + (b1 + b2) * (l1f**2 - l1i**2) / 2) \
        + x * (l1f**2 - l1i**2) * (l1f**2 - l1i**2) / 4 + c * Lambda**2
    return integral / Lambda**2


def polix(f):
    ll = [2, 4, 4, 2]
    st = [0, 2, 6, 10]
    return (st[f], st[f] + ll[f])


for i in range(4):
    for j in range(4):
        oi[:, :, i, j] = np.mean(oi_full[:, :, polix(i)[0]:polix(i)[
                                 1], polix(j)[0]:polix(j)[1]], axis=(2, 3))
        oi_avg[i, j] = np.mean(oi[:, :, i, j])
        oi_max[i, j] = np.max(oi[:, :, i, j])
        oi_min[i, j] = np.min(oi[:, :, i, j])

np.save('oi.npy', oi)
np.save('oi_avg.npy', oi_avg)
np.save('oi_max.npy', oi_max)
np.save('oi_min.npy', oi_min)

# Quadratic fit of the OI in frequency
oi_fit = np.ndarray((6, 4, 4))

x, y = np.meshgrid(wl, wl)
for i in range(4):
    for j in range(4):
        oi_fit[:, i, j] = curve_fit(
            oi_law_fit, (x, y), oi[:, :, i, j].ravel(), p0=[1e10, 1e10, 1e10, 1e10, 0, 1])[0].T

plot = True
skip = 5
wlplot = 1e6 * wl
if plot:
    modes = ["01", "11", "21", "02"]
    n_modes = 4
    for i in range(n_modes):
        for j in range(i, n_modes):
            oi_slice = oi[:, :, i, j] * 1e3 * 1e-12  # Scaling as per original contour plot
            print(oi_slice)
            fig, ax = plt.subplots(figsize=(3, 2.5))
            sns.heatmap(oi_slice[::-1, :],
                        cmap='viridis',
                        # cbar_kws={'label': r'$\times 10^{-3}$'},
                        annot=False, ax=ax)
            contours = ax.contour(oi_slice, levels=8, colors='black', linewidths=0.5,
                                  linestyles='dashed',
                                  extent=[wl.min(), wl.max(), wl.min(), wl.max()])
            ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")
            # plt.title(f"OI Heatmap for Mode {modes[i]} vs Mode {modes[j]}")
            xticks = np.round(wlplot[::skip], 2)
            yticks = np.round(wlplot[::-skip], 2)
            plt.xticks(ticks=range(len(wl))[::5], labels=xticks, rotation=45)
            plt.yticks(ticks=range(len(wl))[::5], labels=yticks, rotation=0)
            plt.xlabel(f"$\lambda_{{{modes[i][0]}{modes[i][1]}}}$ [$\mu m$]")
            plt.ylabel(f"$\lambda_{{{modes[j][0]}{modes[j][1]}}}$ [$\mu m$]")
            # plt.ylabel(r"$\lambda$"+f"({modes[j]}) "+r"[$\mu m$]")
            cbar = ax.collections[0].colorbar
            cbar.set_label(r'$\times 10^{-3}$')
            plt.tight_layout()
            plt.savefig(f'media/oi/freq_dep_{modes[i]}_{modes[j]}.png', format='png', dpi=600)
            plt.close()

print("Seaborn heatmap plots saved to media/oi folder.")
