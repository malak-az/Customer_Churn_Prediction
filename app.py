from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Customer Churn Prediction API")

pipeline = joblib.load("models/churn_model_pipeline.pkl")

model = pipeline["model"]
scaler = pipeline["scaler"]
threshold = pipeline["threshold"]
feature_columns = pipeline["feature_columns"]

class Customer(BaseModel):
    CreditScore: int
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Geography: str
 
def get_age_group(age):
    if age <= 30:
        return "18-30"
    elif age <= 40:
        return "31-40"
    elif age <= 50:
        return "41-50"
    elif age <= 60:
        return "51-60"
    return "60+"


def get_credit_category(score):
    if score < 580:
        return "Poor"
    elif score < 670:
        return "Fair"
    elif score < 740:
        return "Good"
    elif score < 800:
        return "Very Good"
    return "Excellent"
   
@app.post("/predict")
def predict(customer: Customer):

    gender = 0 if customer.Gender.lower() == "male" else 1

    df = pd.DataFrame([{
        "CreditScore": customer.CreditScore,
        "Gender": gender,
        "Tenure": customer.Tenure,
        "Balance": customer.Balance,
        "NumOfProducts": customer.NumOfProducts,
        "HasCrCard": customer.HasCrCard,
        "IsActiveMember": customer.IsActiveMember,
        "EstimatedSalary": customer.EstimatedSalary,
        "Geography": customer.Geography,
        "AgeGroup": get_age_group(customer.Age),
        "CreditCategory": get_credit_category(customer.CreditScore),
        "BalancePerProduct":
            customer.Balance / max(customer.NumOfProducts, 1)
    }])

    df = pd.get_dummies(
        df,
        columns=["Geography", "AgeGroup", "CreditCategory"]
    )

    df = df.reindex(columns=feature_columns, fill_value=0)

    scaled = scaler.transform(df)

    probability = model.predict_proba(scaled)[0][1]

    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "probability": round(float(probability), 4),
        "threshold": round(float(threshold), 4),
        "result": "Customer will churn" if prediction else "Customer will stay"
    }