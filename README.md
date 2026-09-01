# Predictive Maintenance & Remaining Useful Life (RUL) Estimation for Ball Bearings

An end-to-end machine learning and signal processing pipeline designed to monitor real-time health degradation, extract hybrid time-frequency features, and predict the Remaining Useful Life (RUL) of industrial machinery using high-frequency multivariate sensor data (NASA IMS Bearing Dataset).

---

## **Current Development & Validation Status**

> **Active Development Phase:** Feature extraction pipelines, signal preprocessing modules, and LightGBM regression models are actively undergoing optimization, hyperparameter tuning, and cross-validation against synthetic and historical degradation logs.

---

## **System Architecture & Core Mechanisms**

Industrial machinery failures often stem from gradual, sub-surface fatigue in rolling element bearings. Traditional threshold-based monitoring fails to capture non-linear wear patterns early enough. This system replaces static heuristics with a robust machine learning framework powered by **LightGBM** to map complex vibration signals to predictive health metrics.

### **1. Signal Processing & Hybrid Feature Extraction**

* **Time-Domain Metrics:** Computes statistical indicators from raw accelerometer streams—including **RMS, Kurtosis, and Crest Factor**—to capture sudden vibration spikes and impact shocks indicative of micro-cracks.
* **Frequency-Domain Analysis:** Applies the **Fast Fourier Transform (FFT)** to isolate spectral power distributions, mapping energy shifts to specific bearing component failure frequencies (inner/outer race and ball defects).

### **2. Dimensionality Reduction & Health Index (HI) Construction**

* **Feature Aggregation:** Merges multi-sensor time-frequency features into a consolidated feature matrix.
* **Normalization via PCA:** Implements **Principal Component Analysis (PCA)** to distill high-dimensional multi-sensor streams into a single, continuous, and normalized **Health Index (HI)** ranging from 1 ( pristine state) to 0 (complete failure).

### **3. Predictive Modeling via LightGBM**

* **Gradient Boosting Regression:** Utilizes **LightGBM** (Light Gradient Boosting Machine) to model the non-linear degradation trajectory and predict the Remaining Useful Life (RUL). LightGBM is chosen for its superior training speed, efficiency with large tabular feature spaces, and ability to handle complex gradient interactions without overfitting.
* **Threshold-Based Alerting:** Establishes quantitative degradation thresholds on the model's output to trigger proactive maintenance routines before catastrophic failure occurs.

---

## **Current Implementation & Testing Roadmap**

* **Pipeline Optimization:** Refining automated rolling-window feature extraction scripts to handle massive multivariate time-series sets without memory bottlenecks.
* **Model Benchmarking:** Fine-tuning LightGBM hyperparameters (learning rate, max depth, feature fraction) and evaluating performance using RMSE and Mean Absolute Error (MAE) metrics.
* **Robustness Testing:** Running validation suites via Pytest to ensure data integrity across missing sensor inputs or anomalous spikes.

---

## **Technical Stack**

* **Core Language:** Python (3.10+)
* **Machine Learning & Modeling:** LightGBM, Scikit-learn, NumPy, Pandas
* **Signal Processing & Math:** SciPy (FFT, filtering), statsmodels
* **Testing & Workflow:** Pytest, Git/GitHub version control
