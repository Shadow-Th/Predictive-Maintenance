import numpy as np

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

print("Windowed data shape:", X.shape)   
print("Labels shape:", y.shape)          
