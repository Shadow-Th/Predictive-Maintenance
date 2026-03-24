import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb

df = pd.read_csv('features_extracted_with_RUL.csv')
X = df.drop(['label', 'RUL'], axis=1).values
y = df['RUL'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMRegressor(objective='regression', n_estimators=100, random_state=42)
model.fit(X_train, y_train)

model.booster_.save_model('rul_lightgbm_model.txt')
print("Model trained and saved")
