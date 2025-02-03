import numpy as np
from scipy.special import erf

# Parameters
beta2A = 0.005  # ps^2/km
beta2B = 0.005
beta2m = (beta2A + beta2B) / 2

dbeta1 = 1e-50  # ps/km
L = 100  # km
T = 100  # ps

LDA = T**2 / beta2A
LDB = T**2 / beta2B
LDm = T**2 / beta2m
LWAB = T / dbeta1
m_max = L / LWAB

# Predefined constants
Nomega = 1000
Nzeta = 4000

domega = 2 * np.pi / Nomega
dzeta = L / Nzeta

omega = np.arange(0, 2 * np.pi, domega)
z = np.arange(dzeta, L + dzeta, dzeta)
m = 0
# Vectorized computations
j32 = np.sqrt(1j)**3 * 0.5
invLp = 1 / LWAB + 2 * np.pi / LDm
invLm = 1 / LWAB - 2 * np.pi / LDm

sqrt_LDm_z = np.sqrt(LDm / z[:, None])  # Broadcast for vectorized operations

term_pos1 = j32 * sqrt_LDm_z * ((-omega / LDB + invLp) * z[:, None] + m)
term_pos2 = j32 * sqrt_LDm_z * ((omega / LDA + invLm) * z[:, None]  + m)
term_neg1 = j32 * sqrt_LDm_z * ((-omega / LDA + invLp) * z[:, None] + m)
term_neg2 = j32 * sqrt_LDm_z * ((omega / LDB + invLm) * z[:, None]  + m)

intpos = np.sum(abs(erf(term_pos1) - erf(term_pos2))**2, axis=1)
intneg = np.sum(abs(erf(term_neg1) - erf(term_neg2))**2, axis=1)

intpos *= LDm * np.pi / 4 / (2 * np.pi)**2 * domega / (2 * np.pi) / z / T
intneg *= LDm * np.pi / 4 / (2 * np.pi)**2 * domega / (2 * np.pi) / z / T

X000 = dzeta * np.sum(intpos + intneg)

# Output results
X000_squared = X000**2

print(f"X000^2: {X000_squared:.3e}")

# Plot results
import matplotlib.pyplot as plt

plt.figure()
plt.plot(z, intpos, label="intpos")
plt.plot(z, intneg, label="intneg")
plt.legend()
plt.xlabel("z")
plt.ylabel("Intensity")
plt.plot(z, intpos + intneg, label="intpos + intneg")
plt.legend()
plt.xlabel("z")
plt.grid()
plt.show()
