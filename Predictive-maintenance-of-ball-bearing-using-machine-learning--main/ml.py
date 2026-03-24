import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.fft import fft

def extract_features_multichannel(window):
    features = []
    for ch in range(window.shape[1]):
        sig = window[:, ch]
        rms = np.sqrt(np.mean(sig**2))
        std = np.std(sig)
        mean = np.mean(sig)
        peak2peak = np.ptp(sig)
        kurt = kurtosis(sig)
        sk = skew(sig)
        fft_vals = np.abs(fft(sig))
        freqs = np.fft.fftfreq(len(sig))
        dominant_freq = freqs[np.argmax(fft_vals)]
        if np.sum(fft_vals) > 0:
            spec_centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals)
            spec_bandwidth = np.sqrt(np.sum(((freqs - spec_centroid) ** 2) * fft_vals) / np.sum(fft_vals))
        else:
            spec_centroid = 0.0
            spec_bandwidth = 0.0
        features.extend([
            rms, std, mean, peak2peak, kurt, sk,
            dominant_freq, spec_centroid, spec_bandwidth
        ])
    return features

WINDOW_SIZE = 2048
STEP_SIZE = 1024

def window_and_label_multichannel(signal_array, label, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    windows = []
    labels = []
    num_samples, num_channels = signal_array.shape
    for start in range(0, num_samples - window_size + 1, step_size):
        window = signal_array[start:start + window_size, :]
        mean = np.mean(window, axis=0)
        std = np.std(window, axis=0)
        std[std == 0] = 1
        window_norm = (window - mean) / std
        windows.append(window_norm)
        labels.append(label)
    return np.array(windows), np.array(labels)
healthy_data = np.loadtxt("healthy_data.csv", delimiter=",")
moderate_data = np.loadtxt("moderate_data.csv", delimiter=",")
unhealthy_data = np.loadtxt("unhealthy_data.csv", delimiter=",")

healthy_windows, healthy_labels = window_and_label_multichannel(healthy_data, 0)
moderate_windows, moderate_labels = window_and_label_multichannel(moderate_data, 1)
unhealthy_windows, unhealthy_labels = window_and_label_multichannel(unhealthy_data, 2)
X_windows = np.vstack([healthy_windows, moderate_windows, unhealthy_windows])
y_labels = np.hstack([healthy_labels, moderate_labels, unhealthy_labels])
features_list = [extract_features_multichannel(window) for window in X_windows]
columns = []
num_channels = X_windows.shape[2]
stats = ["rms", "std", "mean", "peak2peak", "kurtosis", "skewness", "dominant_freq", "spectral_centroid", "spectral_bandwidth"]
for ch in range(num_channels):
    for stat in stats:
        columns.append(f"ch{ch+1}_{stat}")

columns.append("label")

features_df = pd.DataFrame(features_list, columns=columns[:-1])
features_df["label"] = y_labels
features_df.to_csv("ims_features_extracted.csv", index=False)
print("Feature extraction complete. Saved to ims_features_extracted.csv")
