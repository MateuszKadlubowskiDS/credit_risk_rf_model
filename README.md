# Credit Risk Prediction with Random Forest

This project builds a machine learning model to predict **loan default risk** using LendingClub data.

## Problem

Binary classification task:

* **1 (Default)** – borrower failed to repay
* **0 (Non-default)** – loan repaid

The dataset is **highly imbalanced**, making default detection challenging.

---

## Data

* Source: LendingClub public dataset
* File: `loan.csv` (~430 MB, not included)
* Each row = one loan
* Target: `is_default` (derived from `loan_status`)

Download: https://www.kaggle.com/datasets/wordsforthewise/lending-club

---

## Pipeline

* Data cleaning and preprocessing
* Removal of **data leakage features** (post-loan variables)
* Feature engineering (e.g. grade encoding, employment length)
* One-hot encoding of categorical variables
* Train/test split with stratification
* Model: **Random Forest (class_weight='balanced')** -- Handling class imbalance
* Hyperparameter tuning using GridSearchCV
* Threshold tuning for recall/precision trade-off

---

## Results

* **ROC-AUC:** ~0.71
* **Recall (default):** ~0.65
* **Precision (default):** ~0.12

The model prioritizes recall to better identify high-risk borrowers, accepting a higher number of false positives.

---

## Key Insight

Most important feature:

* `grade` (internal credit rating)

Additional important factors:

* debt-to-income ratio (`dti`)
* credit utilization (`revol_util`)
* total balances (`tot_cur_bal`, `total_rev_hi_lim`)

---
## Tech Stack

* Python
* pandas
* scikit-learn
* matplotlib / seaborn

