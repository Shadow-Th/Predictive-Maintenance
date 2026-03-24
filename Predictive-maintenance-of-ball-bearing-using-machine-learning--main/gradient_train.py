import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
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
clf = GradientBoostingClassifier(
    n_estimators=100,       
    learning_rate=0.1,      
    max_depth=3,            
    random_state=42
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification report:\n", classification_report(y_test, y_pred))
