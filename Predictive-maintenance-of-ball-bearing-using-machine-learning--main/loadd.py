import os
import numpy as np
import pandas as pd
folder_path = r"C:\Users\ASUS\Downloads\dataset"
def load_nasa_ims_data(folder_path):
    all_data = []
    print(f"Loading files from: {folder_path}")
    for idx, filename in enumerate(sorted(os.listdir(folder_path))):
        print(f"[{idx}] Reading: {filename}")
        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path, sep=r'\s+', header=None)
        all_data.append(data.values)
    print("Finished loading all files, stacking data...")
    if all_data:
        return np.vstack(all_data)
    else:
        print("No data loaded!")
        return np.array([])
healthy_data = load_nasa_ims_data(r"C:\Users\ASUS\Downloads\dataset\1st_test\1st_test")
moderate_data = load_nasa_ims_data(r"C:\Users\ASUS\Downloads\dataset\2nd_test\2nd_test")
unhealthy_data = load_nasa_ims_data(r"C:\Users\ASUS\Downloads\dataset\3rd_test\4th_test\txt")

np.savetxt("healthy_data.csv", healthy_data, delimiter=",")
np.savetxt("moderate_data.csv", moderate_data, delimiter=",")
np.savetxt("unhealthy_data.csv", unhealthy_data, delimiter=",")
