# Credit Risk Prediction with Random Forest

This project predicts whether a loan will default (1) or not default (0) based on customer and loan attributes.

## Data
- Dataset is not included in the repository due to its large size (~430 MB).
- To run the project locally, place the original `loan.csv` file in a local `data/` folder (which is ignored by git).

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

