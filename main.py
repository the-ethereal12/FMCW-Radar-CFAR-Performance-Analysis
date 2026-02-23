import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Radar Parameters
# -------------------------
fc = 77e9
B = 150e6
T = 5.5e-6
c = 3e8
S = B / T

Nr = 512
Nd = 32
t = np.linspace(0, T, Nr)

R_true = 100
v_true = 25
SNR_dB = 10

range_res = c / (2 * B)
range_axis = np.linspace(0, range_res * (Nr//2), Nr//2)

# -------------------------
# Signal Generation
# -------------------------
signal = np.zeros((Nd, Nr))

for i in range(Nd):

    tx = np.cos(2*np.pi*(fc*t + (S/2)*t**2))

    tau = 2*R_true/c
    fd = 2*v_true*fc/c

    rx = np.cos(2*np.pi*(fc*(t-tau) +
                         (S/2)*(t-tau)**2 +
                         fd*i*T))

    signal[i,:] = tx * rx

# -------------------------
# Add Noise
# -------------------------
signal_power = np.mean(signal**2)
SNR_linear = 10**(SNR_dB/10)
noise_power = signal_power/SNR_linear

noise = np.sqrt(noise_power)*np.random.randn(Nd,Nr)

# -------------------------
# Add Ground Clutter
# -------------------------
clutter = 0.5*np.cos(2*np.pi*0.1*np.outer(np.ones(Nd), t))
signal_clutter = signal + noise + clutter

# -------------------------
# 2D FFT
# -------------------------
RDM = np.abs(np.fft.fftshift(
             np.fft.fft(
             np.fft.fft(signal_clutter, axis=1)[:, :Nr//2],
             axis=0), axes=0))

# -------------------------
# Fixed Threshold Detection
# -------------------------
fixed_threshold = np.mean(RDM) * 5
fixed_detection = RDM > fixed_threshold

# -------------------------
# CA-CFAR Detection
# -------------------------
Tr, Td = 6, 4
Gr, Gd = 2, 2
offset = 6

CFAR = np.zeros_like(RDM)

for i in range(Td+Gd, Nd-(Td+Gd)):
    for j in range(Tr+Gr, (Nr//2)-(Tr+Gr)):

        noise_window = RDM[i-(Td+Gd):i+(Td+Gd)+1,
                           j-(Tr+Gr):j+(Tr+Gr)+1]

        guard_window = RDM[i-Gd:i+Gd+1,
                           j-Gr:j+Gr+1]

        total_noise = np.sum(noise_window) - np.sum(guard_window)
        num_train = noise_window.size - guard_window.size

        threshold = (total_noise/num_train) * (10**(offset/10))

        if RDM[i,j] > threshold:
            CFAR[i,j] = 1

# -------------------------
# Plot Comparison
# -------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(RDM, aspect='auto', origin='lower')
plt.title("Range Doppler Map")

plt.subplot(1,3,2)
plt.imshow(fixed_detection, aspect='auto', origin='lower')
plt.title("Fixed Threshold Detection")

plt.subplot(1,3,3)
plt.imshow(CFAR, aspect='auto', origin='lower')
plt.title("CA-CFAR Detection")

plt.tight_layout()
plt.show()