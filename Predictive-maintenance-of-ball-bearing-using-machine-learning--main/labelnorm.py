import os
import numpy as np
import pandas as pd

parent_folder = r"C:\Users\ASUS\Downloads\dataset"
test_folders = ["1st_test", "2nd_test", "3rd_test/4th_test/txt"]

all_dfs = []

for test_folder in test_folders:
    folder_path_txt = os.path.join(parent_folder, test_folder, "txt")
    folder_path = folder_path_txt if os.path.exists(folder_path_txt) else os.path.join(parent_folder, test_folder)
    print(f"Processing: {folder_path}")
    for file_num, filename in enumerate(sorted(os.listdir(folder_path))):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path) or filename.startswith('.'): continue
        data = pd.read_csv(file_path, sep=r'\s+', header=None)
        data['unit'] = f"{test_folder}_{file_num}"
        data['cycle'] = np.arange(len(data))
        all_dfs.append(data)

if not all_dfs:
    raise RuntimeError("No data loaded! Check the folder paths and file locations.")

df = pd.concat(all_dfs, ignore_index=True)

df['RUL'] = df.groupby('unit')['cycle'].transform(lambda x: x.max() - x)

def label_rul(rul):
    if rul > 50: return "healthy"
    elif rul > 20: return "moderate"
    else: return "unhealthy"
df["label"] = df["RUL"].apply(label_rul)

sensor_cols = [col for col in df.columns if isinstance(col, int) or str(col).isdigit()]
for col in sensor_cols:
    df[f"{col}_zscore"] = (df[col] - df[col].mean()) / df[col].std()

df.to_csv(r"C:\Users\ASUS\Downloads\ims_all_processed.csv", index=False)
print("Done! Combined processed file saved as ims_all_processed.csv")



