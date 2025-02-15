import matplotlib.pyplot as plt
import numpy as np
import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
from pynlin.utils import dBm2watt
import pynlin.constellations

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.weight'] = '500'
plt.rcParams['font.size'] = '26'

def arity_coefficient():
  m_QAM = [4, 16, 64, 256, 1024]
  m_PSK = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] 
  arity_list=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
  fig_arity = fig_arity = plt.figure(figsize=(10, 5))
  var_QAM = []
  var_PSK = []
  for m in m_QAM:
    average_power = dBm2watt(0)
    qam = pynlin.constellations.QAM(m)
    qam_symbols = qam.symbols()

    qam_symbols = qam_symbols / np.sqrt(np.mean(np.abs(qam_symbols)**2))
    var_QAM.append(np.mean(np.abs(qam_symbols)**4)/np.mean(np.abs(qam_symbols)**2) ** 2 - 1)

  for m in m_PSK:
    average_power = dBm2watt(0)
    qam = pynlin.constellations.PSK(m)
    qam_symbols = qam.symbols()

    qam_symbols = qam_symbols / np.sqrt(np.mean(np.abs(qam_symbols)**2))
    var_PSK.append(np.mean(np.abs(qam_symbols)**4)/np.mean(np.abs(qam_symbols)**2) ** 2 - 1)

  # normalized to 16-QAM variance
  var_16_QAM = var_QAM[1]
  plt.semilogx(m_QAM, var_QAM/var_16_QAM, color='black', base=2, linestyle="none", marker="x", markersize=13, label="QAM")
  plt.semilogx(m_PSK, var_PSK/var_16_QAM, color='black', base=2, linestyle="none", marker="o", markersize=10, label="PSK")
  plt.annotate("{:1.3f}".format(var_QAM[1]/var_16_QAM), (16, var_QAM[1]/var_16_QAM-0.15))
  plt.annotate("{:1.3f}".format(var_QAM[2]/var_16_QAM), (64, var_QAM[2]/var_16_QAM-0.15))
  plt.annotate("{:1.3f}".format(var_QAM[3]/var_16_QAM), (256-70, var_QAM[3]/var_16_QAM-0.15))
  plt.annotate("{:1.3f}".format(var_QAM[4]/var_16_QAM), (1024-390, var_QAM[4]/var_16_QAM-0.15))


  plt.grid()
  plt.xlabel("Modulation order")
  plt.xticks(ticks=arity_list, labels=arity_list)
  plt.ylabel(r"$\mu$ normalized to 16-QAM")
  #plt.yticks(ticks=constellation_variance, labels=["1.0", "1.190", "1.235"])
  plt.subplots_adjust(wspace=0.0, hspace=0, right = 9.8/10, bottom=-1)
  plt.legend(loc="center right")
  plt.minorticks_on()

  fig_arity.tight_layout()
  fig_arity.savefig("modulation_order_noise.pdf")
  
def constellation_statistics():
  average_energy = 1
  power_dBm_list = np.linspace(-20, 0, 3)
  arity_list = [16, 64, 256]

  constellation_variance = []

  for ar_idx, M in enumerate(arity_list):
              qam = pynlin.constellations.QAM(M)

              qam_symbols = qam.symbols()
              cardinality = len(qam_symbols)

              # assign specific average optical energy
              qam_symbols = qam_symbols / \
                  np.sqrt(np.mean(np.abs(qam_symbols)**2)) * \
                  np.sqrt(average_energy)

              constellation_variance.append(
                  np.mean(np.abs(qam_symbols)**4) - np.mean(np.abs(qam_symbols)**2) ** 2)

  fig_arity = plt.figure(figsize=(8, 9))

  # normalized to 16-QAM variance
  plt.loglog(arity_list, constellation_variance/constellation_variance[0],
            marker='x', markersize=10, color='black')
  plt.minorticks_off()
  plt.grid()
  plt.xlabel("Modulation order")
  plt.xticks(ticks=arity_list, labels=arity_list)
  plt.ylabel(r"variance scale factor")
  plt.yticks(ticks=constellation_variance/constellation_variance[0], labels=["1.0", "1.190", "1.235"])
  plt.subplots_adjust(wspace=0.0, hspace=0, right = 9.8/10, bottom=-1)

  fig_arity.tight_layout()
  fig_arity.savefig("order_noise.pdf")