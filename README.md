# Credit Risk Prediction with Random Forest

This project predicts whether a loan will default (1) or not default (0) based on customer and loan attributes.

## Data
- Source: LendingClub public loan dataset
- File: `loan.csv` (not included in the repository due to large size ~430 MB)
- Each row represents one loan issued on the LendingClub platform
- Target variable: `is_default` (constructed from the original `loan_status` column)

### Access
The dataset can be obtained from public LendingClub archives, for example via Kaggle:
https://www.kaggle.com/datasets/wordsforthewise/lending-club

## Main steps in the pipeline
- Cleaning and preprocessing (missing values, categorical encoding)
- Feature engineering (grade mapping, employment length parsing)
- Handling class imbalance (downsampling majority class)
- Hyperparameter tuning using GridSearchCV
- Final Random Forest model training
- Evaluation: ROC AUC, classification report, confusion matrix
- Feature importance plot

## How to run
pip install -r requirements.txt  
python credit_risk_rf_model.py

## Technologies
Python  
pandas  
scikit-learn  
imbalanced-learn  
seaborn  
matplotlib  

## My role
- preprocessing and feature engineering  
- balancing the dataset  
- Random Forest with hyperparameter tuning  
- model evaluation and interpretation  

