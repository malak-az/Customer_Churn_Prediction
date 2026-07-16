# Deploying the Churn API (FastAPI + Render)

## What's here

```
app/
├── __init__.py
├── main.py         # FastAPI app: /health and /predict
└── preprocess.py    # rebuilds the 22 trained feature columns from raw input
requirements-api.txt  # lean, pinned deps for the API only (not the full notebook env)
Dockerfile
```

This has already been tested locally against your actual `models/churn_model_pipeline.pkl`
(XGBoost, threshold 0.309) — confirmed the preprocessing produces the exact 22-column
feature order the model expects, and a real churned customer from your raw data
correctly predicts above threshold.

## Step 1 — Test locally

```bash
pip install -r requirements-api.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs`, try `/predict` with a sample customer. Compare
against a prediction from your notebook for the same row to double check.

## Step 2 — Commit and push

```bash
git add app/ requirements-api.txt Dockerfile DEPLOYMENT.md
git commit -m "Add FastAPI deployment for churn model"
git push
```

## Step 3 — Create the Render service

1. render.com → **New +** → **Web Service**
2. Connect the `Customer_Churn_Prediction` repo
3. Render should detect the `Dockerfile` — choose **Docker** as environment if asked
4. Free tier is fine for a portfolio demo
5. **Create Web Service** and wait for the build

## Step 4 — Test the live URL

```bash
curl https://<your-service>.onrender.com/health

curl -X POST https://<your-service>.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 619, "Geography": "France", "Gender": "Female", "Age": 42,
    "Tenure": 2, "Balance": 0.0, "NumOfProducts": 1, "HasCrCard": 1,
    "IsActiveMember": 1, "EstimatedSalary": 101348.88
  }'
```

Expected: `churn_probability` ≈ 0.397, `will_churn: true`, `threshold_used` ≈ 0.309
(same numbers verified locally above).

## Notes

- Free-tier Render spins down when idle — first request after a while takes ~30-50s.
- `requirements-api.txt` is a deliberately trimmed subset of your full dev environment
  (no streamlit/torch/transformers/etc.) so the image stays small and builds fast.
- If you retrain the model, just overwrite `models/churn_model_pipeline.pkl` and push —
  Render redeploys automatically on push to `main`.
