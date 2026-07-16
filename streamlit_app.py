import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

#######################################################
# PAGE CONFIG
#######################################################

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

#######################################################
# LOAD MODEL
#######################################################

@st.cache_resource
def load_pipeline():
    saved = joblib.load("models/churn_model_pipeline.pkl")
    return saved["pipeline"], saved["threshold"]

pipeline, threshold = load_pipeline()

#######################################################
# SIDEBAR
#######################################################

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose Page",
    [
        " Home",
        " Single Prediction",
        " Batch Prediction",
        " Model Evaluation"
    ]
)

#######################################################
# HOME
#######################################################

if page == " Home":

    st.title(" Customer Churn Prediction Dashboard")

    st.markdown("""
This dashboard predicts whether a customer is likely to leave the bank.

### Features

-  Predict one customer
-  Upload CSV for batch prediction
-  Evaluate model performance
-  Download prediction results

Model:

- XGBoost
- Custom Feature Engineering Pipeline
- Threshold Optimization
""")

#######################################################
# SINGLE PREDICTION
#######################################################

elif page == " Single Prediction":

    st.title(" Single Customer Prediction")

    col1, col2 = st.columns(2)

    with col1:

        credit = st.number_input("Credit Score",300,900,650)

        geography = st.selectbox(
            "Geography",
            ["France","Germany","Spain"]
        )

        gender = st.selectbox(
            "Gender",
            ["Male","Female"]
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

    with col2:

        balance = st.number_input(
            "Balance",
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
            "Active Member",
            [0,1]
        )

        salary = st.number_input(
            "Estimated Salary",
            value=50000.0
        )

    if st.button("Predict"):

        df = pd.DataFrame([{

            "CreditScore":credit,
            "Geography":geography,
            "Gender":gender,
            "Age":age,
            "Tenure":tenure,
            "Balance":balance,
            "NumOfProducts":products,
            "HasCrCard":card,
            "IsActiveMember":active,
            "EstimatedSalary":salary

        }])

        prob = pipeline.predict_proba(df)[0][1]

        pred = prob >= threshold

        st.metric(
            "Churn Probability",
            f"{prob*100:.2f}%"
        )

        st.progress(float(prob))

        if pred:
            st.error("⚠ Customer is likely to churn.")
        else:
            st.success(" Customer is likely to stay.")

#######################################################
# CSV PREDICTION
#######################################################

elif page == " Batch Prediction":

    st.title(" Batch Prediction")

    uploaded = st.file_uploader(
        "Upload CSV",
        type="csv"
    ) 

    if uploaded is not None:

        df = pd.read_csv(uploaded)

        probs = pipeline.predict_proba(df)[:, 1]

        preds = (probs >= threshold).astype(int)

        output = df.copy()

        output["Probability"] = probs.round(4)

        output["Prediction"] = preds

        # Optional: make prediction more readable
        output["Prediction Label"] = output["Prediction"].map({
            0: "Stay",
            1: "Churn"
        })

        st.dataframe(output, use_container_width=True)

        st.download_button(
            "Download Predictions",
            output.to_csv(index=False),
            "predictions.csv",
            "text/csv"
    )

#######################################################
# MODEL EVALUATION
#######################################################

elif page == " Model Evaluation":

    st.title(" Model Evaluation")

    uploaded = st.file_uploader(
        "Upload labeled CSV (must contain Exited)",
        type="csv"
    )

    if uploaded:

        df = pd.read_csv(uploaded)

        if "Exited" not in df.columns:

            st.error("CSV must contain Exited column.")

        else:

            y_true = df["Exited"]

            X = df.drop(columns=["Exited"])

            probs = pipeline.predict_proba(X)[:,1]

            preds = (probs>=threshold).astype(int)

            accuracy = accuracy_score(y_true,preds)
            precision = precision_score(y_true,preds)
            recall = recall_score(y_true,preds)
            f1 = f1_score(y_true,preds)
            auc = roc_auc_score(y_true,probs)

            c1,c2,c3,c4,c5 = st.columns(5)

            c1.metric("Accuracy",f"{accuracy:.3f}")
            c2.metric("Precision",f"{precision:.3f}")
            c3.metric("Recall",f"{recall:.3f}")
            c4.metric("F1 Score",f"{f1:.3f}")
            c5.metric("ROC AUC",f"{auc:.3f}")

            st.divider()

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_true,preds)

            fig, ax = plt.subplots(figsize=(5,4))

            sns.heatmap(

                cm,

                annot=True,

                fmt="d",

                cmap="Blues",

                xticklabels=["Stay","Churn"],

                yticklabels=["Stay","Churn"],

                ax=ax

            )

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            st.pyplot(fig)

            st.divider()

            st.subheader("Classification Report")

            report = classification_report(
                y_true,
                preds,
                output_dict=True
            )

            st.dataframe(
                pd.DataFrame(report).transpose(),
                use_container_width=True
            )

            st.divider()

            output = df.copy()

            output["Probability"] = probs

            output["Prediction"] = preds

            st.download_button(

                "📥 Download Predictions",

                output.to_csv(index=False),

                "evaluation_results.csv",

                "text/csv"

            )