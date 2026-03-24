import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv(r'features_extracted.csv')
df['label'] = df['label'].astype(int)
X = df.drop('label', axis=1).values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(y_train)}")
print(f"Testing samples: {len(y_test)}")
print(f"Training class distribution: {np.bincount(y_train)}")
print(f"Testing class distribution: {np.bincount(y_test)}")
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'verbose': -1,
    'seed': 42
}
model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = np.argmax(y_pred_proba, axis=1)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification report:\n", classification_report(y_test, y_pred))
