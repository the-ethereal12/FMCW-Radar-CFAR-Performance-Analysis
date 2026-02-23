import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Radar System Parameters
# -------------------------
fc = 77e9
c = 3e8
lam = c / fc

Pt = 1e3            # 1 kW transmit power
G = 30              # antenna gain (linear, not dB)
sigma = 1           # target RCS (1 m^2 typical human-sized)
Smin = 1e-13        # receiver sensitivity

# -------------------------
# Maximum Range Calculation
# -------------------------
Rmax = ((Pt * (G**2) * (lam**2) * sigma) /
        ((4*np.pi)**3 * Smin)) ** 0.25

print("Maximum Detection Range (meters):", Rmax)

# -------------------------
# Range vs RCS Analysis
# -------------------------
rcs_values = np.linspace(0.1, 10, 100)
ranges = ((Pt * (G**2) * (lam**2) * rcs_values) /
          ((4*np.pi)**3 * Smin)) ** 0.25

plt.figure()
plt.plot(rcs_values, ranges)
plt.xlabel("Radar Cross Section (m^2)")
plt.ylabel("Maximum Detection Range (m)")
plt.title("Detection Range vs Target RCS")
plt.grid()
plt.show()