# Customer_Churn_Prediction
End-to-end ML pipeline for customer churn prediction built with Python.

# Customer Churn Prediction

A machine learning project that predicts customer churn using an XGBoost classifier with a custom feature engineering pipeline.

## Features

- XGBoost model with Optuna hyperparameter tuning
- Custom preprocessing and feature engineering
- Streamlit dashboard for:
  - Single prediction
  - Batch CSV prediction
  - Model evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
- FastAPI implementation for REST API inference

## Tech Stack

Python, Pandas, Scikit-learn, XGBoost, Optuna, Streamlit, FastAPI, Joblib

## Run

```bash
streamlit run streamlit_app.py
```

or

```bash
uvicorn app:app --reload
```
