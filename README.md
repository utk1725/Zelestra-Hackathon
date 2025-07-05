---

# Solar Panel Efficiency Prediction

This project focuses on predicting solar panel **efficiency** using real-world panel performance and environmental data. It implements end-to-end data processing, feature engineering, and a model ensemble using **CatBoost**, **LightGBM**, and **Ridge Regression** stacking for high-quality predictions.

---

## 📁 Dataset Overview

Three main files are used:

* `train.csv`: Contains 20,000 rows of training data with 17 columns.
* `test.csv`: Contains 12,000 rows of test data without the target.
* `sample_submission.csv`: Format reference for predictions.

Key target variable: `efficiency`

---

## 📊 Project Workflow

### 1. Data Upload & Exploration

* Upload files using Google Colab's `files` module.
* Load data using `pandas`, inspect with `.info()`, `.describe()`, and `.head()`.
* Check data distribution and correlation with seaborn plots.

### 2. Data Cleaning

* Convert all numerical columns to proper types, coercing invalid entries to `NaN`.
* Handle missing data naturally through robust model training and feature engineering.

### 3. Feature Engineering

Two stages of feature engineering:

**Basic:**

```python
- power_output = voltage * current
- temp_diff = module_temperature - temperature
- cleanliness_effect = (1 - soiling_ratio) * irradiance
- panel_degradation = panel_age * soiling_ratio
```

**Advanced:**

```python
- irradiance_per_cloud = irradiance / (cloud_coverage + 1)
- cooling_factor = wind_speed / (module_temperature + 1)
- stress_index = temp_diff / (wind_speed + 1)
- soiling_humidity = soiling_ratio * humidity
- maintenance_effect = maintenance_count / (panel_age + 1)
- instability_index = wind_speed / (pressure + 1)
```

---

### 4. Label Encoding

Categorical columns encoded using `LabelEncoder`:

* `string_id`
* `error_code`
* `installation_type`

---

### 5. Modeling

**Models used:**

* [CatBoostRegressor](https://catboost.ai/)
* [LightGBM Regressor](https://lightgbm.readthedocs.io/)
* `Ridge` Regression for stacked ensemble

**Training Strategy:**

* 5-Fold Cross-Validation for CatBoost and LightGBM
* Out-of-fold predictions used to train the final `Ridge` meta-model
* Final predictions clipped between 0 and 1 for realistic efficiency values

---

### 6. Submission

The final ensemble output is saved to:

```plaintext
stacking_meta_submission.csv
```

and downloaded using:

```python
files.download("stacking_meta_submission.csv")
```

---

## 🛠️ Dependencies

Install these packages before running the notebook:

```bash
pip install catboost lightgbm
```

Also uses:

* pandas
* numpy
* seaborn
* matplotlib
* scikit-learn
* Google Colab (for I/O)

---

## 📈 Evaluation Metric

**Root Mean Squared Error (RMSE)** is used to measure model performance:

```python
from sklearn.metrics import mean_squared_error
```

---

## 📌 Project Highlights

* 🔍 Exploratory Data Analysis
* 🧪 Feature Engineering for Insight & Performance
* 🧠 Model Ensemble with Cross-Validation
* ✅ Submission-ready Output

---

## 📤 Output Example

| id  | efficiency |
| --- | ---------- |
| 0   | 0.5621     |
| 1   | 0.3964     |
| ... | ...        |

---

## 📬 Contact

**Utkarsh Singh**

📧 Email: [utkarshthakur17022002@gmail.com](mailto:utkarshthakur17022002@gmail.com)

🔗 LinkedIn: [linkedin.com/in/utkarshsingh1702](https://www.linkedin.com/in/utkarshsingh1702)

---
