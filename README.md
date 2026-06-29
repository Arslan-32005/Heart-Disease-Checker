# Heart Disease Risk Predictor

## Overview
Binary classification project to predict heart disease risk based on patient medical data. Built with Random Forest and hyperparameter tuning via GridSearchCV with Stratified K-Fold cross-validation.

## Dataset
- **Source:** Kaggle — Heart Disease UCI
- **Size:** 1,025 rows, 14 features
- **Target:** 0 = No Disease, 1 = Disease
- **Class Distribution:** Balanced — 500 vs 525

## Features Used
| Feature | Description |
|---------|-------------|
| age | Patient age |
| sex | Gender (0=Female, 1=Male) |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure |
| chol | Cholesterol level |
| fbs | Fasting blood sugar > 120 |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression |
| slope | ST slope |
| ca | Major vessels colored by fluoroscopy |
| thal | Thal type |

## Approach

### 1. Preprocessing
- Loaded and explored dataset — no missing values
- Separated features and target
- Applied `StandardScaler` on all 13 features — scaling zaroori tha kyunki features alag alag ranges mein the (age: 20-80, chol: 100-600)
- Train-test split: 80/20 with `stratify=y`

### 2. Models Compared
| Model | Accuracy | F1 (Disease) |
|-------|----------|--------------|
| Logistic Regression | 81% | 0.83 |
| Random Forest (default) | 100% | 1.00 — Overfit |
| XGBoost (default) | 100% | 1.00 — Overfit |
| **Random Forest (tuned)** | **94%** | **0.94** |

### 3. Hyperparameter Tuning — GridSearchCV
- Applied GridSearchCV with StratifiedKFold (5 splits) on Random Forest
- Initial params included `max_depth: [3,5,10]` — depth 10 caused overfitting (100% accuracy)
- Reduced params to `max_depth: [3,5]`, `min_samples_leaf: [2,4]` — overfitting resolved
- **Best params:** `max_depth=5, min_samples_leaf=2, n_estimators=200`
- **Best CV F1:** 0.91

### 4. Final Model Results
precision  recall  f1-score
No Disease       0.97    0.91    0.94

Disease          0.92    0.97    0.94

Accuracy                         0.94
## Key Learnings
- Default RF and XGBoost both overfit on small dataset (1025 rows) — 100% accuracy was memorization not learning
- GridSearchCV with depth limiting resolved overfitting — F1 dropped from 1.00 to realistic 0.94
- StandardScaler zaroori tha — features alag ranges mein the, model biased ho jaata bina scaling ke
- StratifiedKFold use kiya GridSearch mein — class distribution har fold mein same rakha

## Deployment
- Live app: [Heart Disease Checker](https://heart-disease-checker-fc3xzsxs4n2fuvafpjhlsd.streamlit.app/)
- User Enter 13 medical values — model predict real-time heart attack risk

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (RandomForestClassifier, LogisticRegression, StandardScaler, GridSearchCV, StratifiedKFold)
- XGBoost
- Streamlit
- Joblib
