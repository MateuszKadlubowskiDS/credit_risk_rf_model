# Credit Risk Prediction with Random Forest

This project predicts whether a loan will **default (1)** or **not default (0)** based on customer and loan attributes.

# Credit Risk Prediction with Random Forest

This project predicts whether a loan will default (1) or not default (0) based on customer and loan attributes.

## Data
- File: data/loan.csv
- Target variable: is_default (constructed from loan_status)

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

