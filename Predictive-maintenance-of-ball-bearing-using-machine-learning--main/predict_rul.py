import numpy as np
import lightgbm as lgb
model = lgb.Booster(model_file='rul_lightgbm_model.txt')
new_features = np.array([1.0,1.0,-3.859759734048396e-17,7.531956823831241,0.3063178571804146,0.106292365058768,0.119140625,-0.0001470161087086,0.2888257077960481])
predicted_rul = model.predict(new_features.reshape(1, -1))[0]
sample_time_seconds = 60 
rul_seconds = predicted_rul * sample_time_seconds
rul_days = rul_seconds // (24 * 3600)
rul_hours = (rul_seconds % (24 * 3600)) // 3600
rul_minutes = (rul_seconds % 3600) // 60

print(f"Predicted RUL: {predicted_rul:.2f} samples")
print(f"Estimated remaining life: {int(rul_days)} days, {int(rul_hours)} hours, {int(rul_minutes)} minutes")

