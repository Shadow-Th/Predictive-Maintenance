import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
df = pd.read_csv('features_extracted.csv')
df['label'] = df['label'].astype(int)
X = df.drop('label', axis=1).values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
class LGBMWrapper:
    def __init__(self):
        self.params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'verbose': -1,
            'seed': 42
        }
        self.num_boost_round = 100
        self.model = None

    def fit(self, X_train, y_train):
        train_data = lgb.Dataset(X_train, label=y_train)
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round
        )
    def predict(self, X_test):
        y_pred_proba = self.model.predict(X_test)
        return np.argmax(y_pred_proba, axis=1)

models = [
    ("XGBoost", XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=100, random_state=42, eval_metric='mlogloss')),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ("ANN", MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)),
    ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
    ("LightGBM", LGBMWrapper())
]

acc, prec, rec, f1, model_names = [], [], [], [], []
classes_numeric = ['0', '1', '2']

for name, clf in models:
    clf.fit(X_train, y_train)
    if name == "XGBoost":
        y_pred_raw = clf.predict(X_test)
        y_pred = np.argmax(y_pred_raw, axis=1) if y_pred_raw.ndim > 1 else y_pred_raw
    else:
        y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    acc.append(accuracy_score(y_test, y_pred))
    prec.append([report[c]['precision'] for c in classes_numeric])
    rec.append([report[c]['recall'] for c in classes_numeric])
    f1.append([report[c]['f1-score'] for c in classes_numeric])
    model_names.append(name)
acc = [round(a, 4) for a in acc]
prec = [[round(p, 4) for p in row] for row in prec]
rec = [[round(r, 4) for r in row] for row in rec]
f1 = [[round(f, 4) for f in row] for row in f1]

classes = ['Class 0', 'Class 1', 'Class 2']
x = np.arange(len(classes))
width = 0.12   
colors = ['dodgerblue', 'gold', 'mediumorchid', 'limegreen', 'orchid', 'cyan']

plt.figure(figsize=(11, 6.5))
bars = plt.bar(model_names, acc, color=colors, zorder=2)
plt.ylim(0.75, 1.01)
plt.ylabel('Test Accuracy')
plt.title('Model Test Accuracy Comparison')
for i, bar in enumerate(bars):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.006, f'{yval:.4f}', ha='center', va='bottom', fontsize=11, rotation=0)
plt.grid(visible=True, which='major', axis='y', linestyle='--', alpha=0.5, zorder=1)
plt.tight_layout(pad=1.2)
plt.show()

def plot_metric_per_class(metric, metric_name):
    plt.figure(figsize=(12, 7.5))
    for idx, alg in enumerate(model_names):
        plt.bar(x + width * idx, metric[idx], width, color=colors[idx], label=alg, zorder=2)
        for k in range(len(classes)):
            val = metric[idx][k]
            if val > 0.5:
                plt.text(x[k] + width * idx, val + 0.01,
                         f"{val:.4f}",
                         ha='center', va='bottom', fontsize=10, rotation=0, fontweight='bold')
    plt.xticks(x + width * (len(model_names) / 2 - 0.5), classes, fontsize=14)
    plt.ylim(0.5, 1.05)
    plt.ylabel('Score', fontsize=14)
    plt.title(f"{metric_name} Per Class", fontsize=15)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=12, frameon=True, fancybox=True)
    plt.grid(visible=True, which='major', axis='y', linestyle='--', alpha=0.45, zorder=1)
    plt.tight_layout(rect=[0, 0.05, 1, 1]) 
    plt.show()

plot_metric_per_class(prec, "Precision")
plot_metric_per_class(rec, "Recall")
plot_metric_per_class(f1, "F1-score")