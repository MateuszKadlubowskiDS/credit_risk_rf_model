# ============================================
# 1. Imports
# ============================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)

pd.set_option('display.max_columns', 100)

# ============================================
# 2. Load data
# ============================================

df = pd.read_csv("Data/loan.csv", low_memory=False)
df_clean = df.copy()

# ============================================
# 3. Target + basic cleaning
# ============================================

default_statuses = [
    'Charged Off', 'Default', 'Late (31-120 days)',
    'Late (16-30 days)', 'Does not meet the credit policy. Status:Charged Off'
]

df_clean['is_default'] = df_clean['loan_status'].apply(
    lambda x: 1 if x in default_statuses else 0
)

df_clean['purpose'] = df_clean['purpose'].apply(
    lambda x: x if x in ['credit_card', 'debt_consolidation'] else 'other'
)

df_clean[['mths_since_last_delinq',
          'mths_since_last_record',
          'mths_since_last_major_derog']] = df_clean[[
    'mths_since_last_delinq',
    'mths_since_last_record',
    'mths_since_last_major_derog'
]].fillna(0)

# ============================================
# 4. Remove leakage / useless columns
# ============================================

useless_vars = [
    'id','member_id','loan_amnt','funded_amnt','funded_amnt_inv','term',
    'application_type','loan_status',

    # leakage
    'total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int',
    'total_rec_late_fee','out_prncp','out_prncp_inv','last_pymnt_amnt',
    'recoveries','collection_recovery_fee',

    # future info
    'last_pymnt_d','next_pymnt_d','last_credit_pull_d',

    # noisy
    'url','title','emp_title','desc',
    'zip_code','addr_state',

    # other
    'verification_status_joint','dti_joint','annual_inc_joint',
    'int_rate','installment','issue_d','pymnt_plan',
    'earliest_cr_line','initial_list_status','policy_code',
    'mths_since_last_record','sub_grade'
]

df_clean = df_clean.drop(columns=useless_vars, errors='ignore')

# ============================================
# 5. Encoding
# ============================================

# grade → numeric
df_clean['grade'] = df_clean['grade'].astype('category').cat.codes + 1

# one-hot
categorical_vars = ['verification_status', 'purpose', 'home_ownership']
categorical_vars = [c for c in categorical_vars if c in df_clean.columns]

ohe = OneHotEncoder(sparse_output=False, drop='first')
encoded = ohe.fit_transform(df_clean[categorical_vars])

encoded_df = pd.DataFrame(
    encoded,
    columns=ohe.get_feature_names_out(categorical_vars)
)

df_clean = pd.concat([df_clean.reset_index(drop=True),
                      encoded_df.reset_index(drop=True)], axis=1)

df_clean = df_clean.drop(columns=categorical_vars)

# emp_length
def convert_emp_length(x):
    x = str(x)
    if "10+" in x:
        return 10
    if "< 1" in x:
        return 0
    digits = "".join(filter(str.isdigit, x))
    return int(digits) if digits else 0

if 'emp_length' in df_clean.columns:
    df_clean['emp_length_numeric'] = df_clean['emp_length'].apply(convert_emp_length)
    df_clean = df_clean.drop(columns=['emp_length'])

# ============================================
# 6. Missing values
# ============================================

df_clean = df_clean.dropna()

# ============================================
# 7. Split
# ============================================

X = df_clean.drop(columns=['is_default'])
y = df_clean['is_default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ============================================
# 8. Baseline model
# ============================================

rf_model = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# ============================================
# 9. Feature selection
# ============================================

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

selected_features = feature_importances[
    feature_importances['Importance'] >= 0.01
]['Feature']

X_train = X_train[selected_features]
X_test = X_test[selected_features]

# ============================================
# 10. GridSearch
# ============================================

param_grid = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

grid = GridSearchCV(
    RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    param_grid,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

# ============================================
# 11. Final model
# ============================================

final_model = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',
    n_jobs=-1,
    **grid.best_params_
)

final_model.fit(X_train, y_train)

# ============================================
# 12. Threshold tuning
# ============================================

y_proba = final_model.predict_proba(X_test)[:, 1]

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

results = []

for t in thresholds:
    y_pred = (y_proba > t).astype(int)
    
    results.append({
        "threshold": t,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
print(results_df)

# ============================================
# 13. Final evaluation
# ============================================

best_threshold = 0.5
y_pred = (y_proba > best_threshold).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("AUC:", roc_auc_score(y_test, y_proba))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()

# ============================================
# 14. Feature importance
# ============================================

fi = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': final_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(fi)