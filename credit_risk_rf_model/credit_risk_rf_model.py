# ============================================
# 1. Import libraries
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
    roc_auc_score, accuracy_score
)
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Display settings
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


# ============================================
# 2. Data loading
# ============================================

# Load dataset (use relative path)
df = pd.read_csv("data/loan.csv", low_memory=False)
risk = df.copy()


# ============================================
# 3. Feature engineering
# ============================================

# Define default loan statuses
default_statuses = [
    'Charged Off', 'Default', 'Late (31-120 days)',
    'Late (16-30 days)', 'Does not meet the credit policy. Status:Charged Off'
]

# Create binary target variable
risk['is_default'] = risk['loan_status'].apply(lambda x: 1 if x in default_statuses else 0)

# Simplify loan purpose categories
risk['purpose'] = risk['purpose'].apply(
    lambda x: x if x in ['credit_card', 'debt_consolidation'] else 'other'
)

# Fill missing credit-related values with 0
risk[['mths_since_last_delinq','mths_since_last_record','mths_since_last_major_derog']] = \
    risk[['mths_since_last_delinq','mths_since_last_record','mths_since_last_major_derog']].fillna(0)


# ============================================
# 4. Drop irrelevant or high-null columns
# ============================================

useless_vars = [
    'id','member_id','loan_amnt','funded_amnt','funded_amnt_inv','term',
    'application_type','loan_status','int_rate','installment','issue_d','pymnt_plan',
    'url','title','zip_code','addr_state','verification_status_joint','earliest_cr_line',
    'initial_list_status','recoveries','collection_recovery_fee','last_pymnt_d',
    'last_credit_pull_d','policy_code','emp_title','next_pymnt_d','desc',
    'mths_since_last_record','sub_grade','dti_joint','annual_inc_joint'
]

risk = risk.drop(columns=useless_vars, errors='ignore').fillna(0)


# ============================================
# 5. Feature selection and encoding
# ============================================

selected_features = [
    'grade','total_rec_int','inq_last_6mths','revol_util','verification_status',
    'total_rec_late_fee','purpose','home_ownership','emp_length','dti',
    'acc_now_delinq','collections_12_mths_ex_med','delinq_2yrs','total_cu_tl',
    'inq_fi','open_il_12m','total_pymnt','open_acc_6m','open_rv_12m','inq_last_12m',
    'total_pymnt_inv','open_il_24m','pub_rec','il_util','open_rv_24m','open_il_6m',
    'total_bal_il','mths_since_rcnt_il','max_bal_bc','all_util','mths_since_last_delinq',
    'tot_coll_amt','open_acc','mths_since_last_major_derog','total_acc','revol_bal',
    'annual_inc','total_rec_prncp','tot_cur_bal','last_pymnt_amnt','total_rev_hi_lim',
    'out_prncp_inv','out_prncp','is_default'
]

risk = risk[selected_features].copy()

# Map grade to numeric values
grade_mapping = {grade: i + 1 for i, grade in enumerate(sorted(risk['grade'].unique()))}
risk['grade'] = risk['grade'].map(grade_mapping)

# One-hot encode categorical variables
categorical_vars = ['verification_status', 'purpose', 'home_ownership']
ohe = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = ohe.fit_transform(risk[categorical_vars])
encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(categorical_vars))

risk = pd.concat([risk.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
risk = risk.drop(columns=categorical_vars)

# Convert emp_length to numeric
def convert_emp_length(emp_length):
    if pd.isnull(emp_length):
        return 0
    if "10+" in emp_length:
        return 10
    if "< 1" in emp_length:
        return 0
    return int("".join(filter(str.isdigit, str(emp_length))))

risk['emp_length_numeric'] = risk['emp_length'].apply(convert_emp_length)
risk = risk.drop(columns=['emp_length'], errors='ignore')


# ============================================
# 6. Handle missing data
# ============================================

risk = risk.dropna()
print(f"Rows after removing missing data: {risk.shape[0]}")


# ============================================
# 7. Split data
# ============================================

X = risk.drop(columns=['is_default'])
y = risk['is_default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ============================================
# 8. Balance dataset
# ============================================

majority_class = risk[risk['is_default'] == 0]
minority_class = risk[risk['is_default'] == 1]

majority_downsampled = resample(
    majority_class,
    replace=False,
    n_samples=len(minority_class),
    random_state=42
)

balanced_data = pd.concat([majority_downsampled, minority_class])
X_balanced = balanced_data.drop(columns=['is_default'])
y_balanced = balanced_data['is_default']


# ============================================
# 9. Random Forest model + GridSearchCV
# ============================================

param_grid = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_balanced, y_balanced)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best AUC during GridSearchCV: {grid_search.best_score_:.4f}")


# ============================================
# 10. Evaluate model
# ============================================

best_params = grid_search.best_params_
final_rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', **best_params)
final_rf_model.fit(X_train, y_train)

def evaluate_model(model, X, y, dataset_name=""):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred_proba)

    print(f"\nClassification Report ({dataset_name}):")
    print(classification_report(y, y_pred))
    print(f"AUC ({dataset_name}): {auc:.4f}")

    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Default', 'Default'],
                yticklabels=['Non-Default', 'Default'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({dataset_name})')
    plt.show()

evaluate_model(final_rf_model, X_test, y_test, dataset_name="Test Set")


# ============================================
# 11. Feature importance
# ============================================

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': final_rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in Final Random Forest Model')
plt.gca().invert_yaxis()
plt.show()
