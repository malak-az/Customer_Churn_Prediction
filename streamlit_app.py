import streamlit as st
import pandas as pd
import joblib

saved = joblib.load("models/churn_model_pipeline.pkl")

pipeline = saved["pipeline"]
threshold = saved["threshold"]

st.title("🏦 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")

credit_score = st.number_input(
    "Credit Score",
    300,
    900,
    650
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    18,
    100,
    40
)

tenure = st.number_input(
    "Tenure",
    0,
    10,
    3
)

balance = st.number_input(
    "Balance",
    0.0,
    value=60000.0
)

products = st.number_input(
    "Number of Products",
    1,
    4,
    2
)

card = st.selectbox(
    "Has Credit Card",
    [0,1]
)

active = st.selectbox(
    "Is Active Member",
    [0,1]
)

salary = st.number_input(
    "Estimated Salary",
    0.0,
    value=50000.0
)
if st.button("Predict"):

    customer = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": card,
        "IsActiveMember": active,
        "EstimatedSalary": salary
    }])

    probability = pipeline.predict_proba(customer)[0][1]

    prediction = probability >= threshold

    st.metric(
        "Churn Probability",
        f"{probability*100:.2f}%"
    )

    if prediction:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")
        
uploaded = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)
if uploaded is not None:

    df = pd.read_csv(uploaded)

    probs = pipeline.predict_proba(df)[:,1]

    preds = probs >= threshold

    df["Churn Probability"] = probs

    df["Prediction"] = preds

    st.dataframe(df)

    st.download_button(
        "Download Results",
        df.to_csv(index=False),
        "predictions.csv",
        "text/csv"
    )
    
