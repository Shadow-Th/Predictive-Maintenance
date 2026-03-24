import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
data = pd.read_csv("features_extracted.csv")
X = data.drop(columns=["label"])
y = data["label"]
label_map = {'healthy':0, 'moderate':1, 'unhealthy':2}
if y.dtype == 'object':
    y = y.map(label_map)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
