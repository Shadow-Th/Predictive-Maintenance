import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.fft import fft
WINDOW_SIZE = 2048
STEP_SIZE = 1024
healthy_data = np.loadtxt("healthy_data.csv", delimiter=",")
moderate_data = np.loadtxt("moderate_data.csv", delimiter=",")
unhealthy_data = np.loadtxt("unhealthy_data.csv", delimiter=",")
def window_and_label(signal_array, label, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    windows = []
    labels = []
    x = signal_array.flatten()
    for start in range(0, len(x) - window_size + 1, step_size):
        window = x[start:start + window_size]
        mean = np.mean(window)
        std = np.std(window)
        if std == 0:
            std = 1
        window_norm = (window - mean) / std
        windows.append(window_norm)
        labels.append(label)
    return np.array(windows), np.array(labels)
healthy_windows, healthy_labels = window_and_label(healthy_data, 0)
moderate_windows, moderate_labels = window_and_label(moderate_data, 1)
unhealthy_windows, unhealthy_labels = window_and_label(unhealthy_data, 2)
X = np.vstack([healthy_windows, moderate_windows, unhealthy_windows])
y = np.hstack([healthy_labels, moderate_labels, unhealthy_labels])
features = []
for win, lbl in zip(X, y):
    rms = np.sqrt(np.mean(win**2))
    std = np.std(win)
    mean = np.mean(win)
    peak2peak = np.ptp(win)
    kurt = kurtosis(win)
    sk = skew(win)
    fft_vals = np.abs(fft(win))
    freqs = np.fft.fftfreq(len(win))
    dominant_freq = freqs[np.argmax(fft_vals)]
    if np.sum(fft_vals) > 0:
        spec_centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals)
        spec_bandwidth = np.sqrt(np.sum(((freqs - spec_centroid) ** 2) * fft_vals) / np.sum(fft_vals))
    else:
        spec_centroid = 0.0
        spec_bandwidth = 0.0
    features.append([
        rms, std, mean, peak2peak, kurt, sk,
        dominant_freq, spec_centroid, spec_bandwidth, lbl
    ])
columns = [
    "rms", "std", "mean", "peak2peak", "kurtosis", "skewness",
    "dominant_freq", "spectral_centroid", "spectral_bandwidth", "label"
]
features_df = pd.DataFrame(features, columns=columns)
features_df.to_csv("features_extracted.csv", index=False)
print("Feature extraction complete. Saved as features_extracted.csv.")
