# emg_plot.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
emg = np.loadtxt('../data/1_raw_data_13-12_22.03.16.txt')
# Visualize raw EMG signal
plt.figure(figsize=(10, 4))
plt.plot(emg)
plt.title('Raw EMG Signal')
plt.xlabel('time')
plt.ylabel('EMG (mV)')
plt.show()

# Filtering: Remove 50 Hz noise using notch filter
fs = 1000  #sampling frequency
b, a = signal.iirnotch(50, 30, fs)
filtered = signal.filtfilt(b, a, emg)

# Bandpass filtering (20-450 Hz)
b2, a2 = signal.butter(4, [20/(fs/2), 450/(fs/2)], btype='band')
filtered = signal.filtfilt(b2, a2, filtered)


# FFT before and after filtering
def plot_fft(sig, fs, title):
    N = len(sig)
    f = np.fft.rfftfreq(N, 1/fs)
    fft_vals = np.abs(np.fft.rfft(sig))
    plt.figure(figsize=(10, 4))
    plt.plot(f, fft_vals)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()

plot_fft(emg, fs, "FFT: Raw EMG")
plot_fft(filtered, fs, "FFT: Filtered EMG")

rms1 = np.sqrt(np.mean(emg**2))
print("RMS before filtering: %.4f" % rms1)

rms2 = np.sqrt(np.mean(filtered**2))
print("RMS after filtering: %.4f" % rms2)

