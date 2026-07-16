"""
Customer Churn Prediction API — single-file version.

Run locally:
    uvicorn app:app --reload

Endpoints:
    GET  /health         -> confirms the model loaded
    POST /predict        -> takes raw customer fields, returns churn probability + decision
    POST /predict-csv    -> batch predictions from an uploaded CSV
    POST /evaluate-csv   -> batch predictions + metrics against a labeled CSV (needs "Exited")

Feature engineering is no longer duplicated here. The pickled artifact is a full
sklearn Pipeline (ChurnFeatureEngineer -> XGBClassifier) trained in
notebooks/pipeline_churn.ipynb. This module only ever calls pipeline.predict_proba()
on raw fields — see feature_engineering.py for the one place that logic lives.
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

# Imported so joblib can unpickle the Pipeline's custom transformer step.
# Not called directly anywhere below - the import itself is what matters.
from feature_engineering import ChurnFeatureEngineer  # noqa: F401

# ---------------------------------------------------------------------------
# App + pipeline loading
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "churn_model_pipeline.pkl")

REQUIRED_COLUMNS = [
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
]

app = FastAPI(title="Customer Churn Prediction API", version="2.0")

pipeline = None
threshold = None


@app.on_event("startup")
def load_model():
    global pipeline, threshold
    saved = joblib.load(MODEL_PATH)
    pipeline = saved["pipeline"]
    threshold = float(saved["threshold"])


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

    X = pd.DataFrame([customer.dict()])
    proba = float(pipeline.predict_proba(X)[:, 1][0])
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

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    try:
        df = pd.read_csv(file.file)

        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}"
            )

        # Whole batch through the pipeline at once - no per-row loop needed
        # now that feature engineering is vectorized inside the transformer.
        probabilities = pipeline.predict_proba(df[REQUIRED_COLUMNS])[:, 1]

        df["ChurnProbability"] = probabilities.round(4)
        df["WillChurn"] = probabilities >= threshold

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

    except HTTPException:
        raise
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

        required_with_label = REQUIRED_COLUMNS + ["Exited"]
        missing = [col for col in required_with_label if col not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}"
            )

        y_true = df["Exited"]

        probabilities = pipeline.predict_proba(df[REQUIRED_COLUMNS])[:, 1]
        predictions = (probabilities >= threshold).astype(int)

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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
