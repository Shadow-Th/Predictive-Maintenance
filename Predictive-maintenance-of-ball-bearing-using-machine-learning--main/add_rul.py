import pandas as pd
import numpy as np
df = pd.read_csv('features_extracted.csv')
df['RUL'] = len(df) - df.index - 1
df.to_csv('features_extracted_with_RUL.csv', index=False)

print("RUL values calculated and saved in 'features_extracted_with_RUL.csv'")
