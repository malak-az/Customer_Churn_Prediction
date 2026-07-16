"""
Customer Churn Prediction API — single-file version.

Run locally:
    uvicorn app:app --reload

Endpoints:
    GET  /health   -> confirms the model loaded
    POST /predict  -> takes raw customer fields, returns churn probability + decision
"""
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse
from io import StringIO
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ---------------------------------------------------------------------------
# Feature engineering — must stay in lockstep with notebooks/feateng.ipynb
# ---------------------------------------------------------------------------

CREDIT_BINS = [300, 580, 670, 740, 800, 900]
CREDIT_LABELS = ["Poor", "Fair", "Good", "Very Good", "Excellent"]

AGE_BINS = [18, 30, 40, 50, 60, 100]
AGE_LABELS = ["18-30", "31-40", "41-50", "51-60", "60+"]


def engineer_features(raw: dict, feature_columns: list) -> pd.DataFrame:
    """
    raw: dict with keys CreditScore, Geography, Gender, Age, Tenure, Balance,
         NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
    feature_columns: the exact column list saved alongside the model (from the
         pickle), used to guarantee correct column order/presence.
    """
    df = pd.DataFrame([raw])

    # Gender -> binary, same mapping as training
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

    # Geography -> one-hot (France/Germany/Spain), same as training
    for geo in ["France", "Germany", "Spain"]:
        df[f"Geography_{geo}"] = (df["Geography"] == geo).astype(int)
    df.drop(columns=["Geography"], inplace=True)

    # CreditScore -> CreditCategory one-hot
    df["CreditCategory"] = pd.cut(df["CreditScore"], bins=CREDIT_BINS, labels=CREDIT_LABELS)
    for label in CREDIT_LABELS:
        df[f"CreditCategory_{label}"] = (df["CreditCategory"] == label).astype(int)
    df.drop(columns=["CreditCategory"], inplace=True)

    # Age -> AgeGroup one-hot, then drop raw Age (matches notebook)
    df["AgeGroup"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True)
    for label in AGE_LABELS:
        df[f"AgeGroup_{label}"] = (df["AgeGroup"] == label).astype(int)
    df.drop(columns=["AgeGroup", "Age"], inplace=True)

    # Engineered ratio feature
    df["BalancePerProduct"] = df["Balance"] / df["NumOfProducts"].replace(0, 1)

    # Guarantee exact column set + order the model was trained on
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    return df


# ---------------------------------------------------------------------------
# App + model loading
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "churn_model_pipeline.pkl")

app = FastAPI(title="Customer Churn Prediction API", version="1.0")

pipeline = None


@app.on_event("startup")
def load_model():
    global pipeline
    pipeline = joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CustomerInput(BaseModel):
    CreditScore: int = Field(..., ge=300, le=900, example=650)
    Geography: Literal["France", "Germany", "Spain"] = "France"
    Gender: Literal["Male", "Female"] = "Female"
    Age: int = Field(..., ge=18, le=100, example=40)
    Tenure: int = Field(..., ge=0, le=10, example=3)
    Balance: float = Field(..., ge=0, example=60000.0)
    NumOfProducts: int = Field(..., ge=1, le=4, example=2)
    HasCrCard: Literal[0, 1] = 1
    IsActiveMember: Literal[0, 1] = 1
    EstimatedSalary: float = Field(..., ge=0, example=50000.0)


class PredictionOutput(BaseModel):
    churn_probability: float
    will_churn: bool
    threshold_used: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model = pipeline["model"]
    threshold = float(pipeline["threshold"])
    feature_columns = pipeline["feature_columns"]

    X = engineer_features(customer.dict(), feature_columns)

    proba = float(model.predict_proba(X)[:, 1][0])
    will_churn = proba >= threshold

    return PredictionOutput(
        churn_probability=round(proba, 4),
        will_churn=will_churn,
        threshold_used=threshold
    )
    
    
@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    try:
        # Read uploaded CSV
        df = pd.read_csv(file.file)

        # Required columns
        required_columns = [
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary"
        ]

        # Check for missing columns
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}"
            )

        model = pipeline["model"]
        threshold = float(pipeline["threshold"])
        feature_columns = pipeline["feature_columns"]

        probabilities = []
        predictions = []

        # Predict each row
        for _, row in df.iterrows():

            features = engineer_features(
                row.to_dict(),
                feature_columns
            )

            probability = float(model.predict_proba(features)[0][1])

            probabilities.append(round(probability, 4))
            predictions.append(probability >= threshold)

        # Add results
        df["ChurnProbability"] = probabilities
        df["WillChurn"] = predictions

        # Convert to CSV
        output = StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={
                "Content-Disposition":
                "attachment; filename=predictions.csv"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
        
@app.post("/evaluate-csv")
async def evaluate_csv(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    try:
        df = pd.read_csv(file.file)

        required_columns = [
            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            "Exited"
        ]

        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}"
            )

        y_true = df["Exited"]

        X = df.drop(columns=["Exited"])

        model = pipeline["model"]
        threshold = float(pipeline["threshold"])
        feature_columns = pipeline["feature_columns"]

        probabilities = []
        predictions = []

        for _, row in X.iterrows():

            features = engineer_features(
                row.to_dict(),
                feature_columns
            )

            prob = float(model.predict_proba(features)[0][1])

            probabilities.append(prob)
            predictions.append(int(prob >= threshold))

        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions)
        recall = recall_score(y_true, predictions)
        f1 = f1_score(y_true, predictions)
        roc_auc = roc_auc_score(y_true, probabilities)

        cm = confusion_matrix(y_true, predictions)

        return {
            "samples": len(df),
            "threshold": round(threshold, 4),

            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),

            "confusion_matrix": cm.tolist()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )