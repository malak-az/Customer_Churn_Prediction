"""
Shared feature-engineering transformer for the customer-churn model.

This module must be importable from BOTH:
  - the training notebook (pipeline_churn.ipynb), which fits and pickles the
    full sklearn Pipeline (this transformer + the classifier) as one object
  - app.py, which unpickles that same Pipeline to serve predictions

That's the whole point: this is now the ONLY place the feature engineering
logic lives. Nothing needs to be reimplemented in the API anymore.

Place this file at your project root, next to app.py:

    project/
      app.py
      feature_engineering.py   <- this file
      models/
        churn_model_pipeline.pkl
      notebooks/
        pipeline_churn.ipynb
      data/
        raw/
        processed/
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Turns raw customer fields into the engineered feature set the model
    was trained on:
      - Gender -> binary
      - Geography -> one-hot (France/Germany/Spain)
      - CreditScore -> binned CreditCategory -> one-hot
      - Age -> binned AgeGroup -> one-hot (raw Age is dropped)
      - BalancePerProduct = Balance / NumOfProducts

    This is stateless (bins/labels/mappings are fixed by design, not learned
    from data), so fit() doesn't need to compute anything from X/y — but it's
    still implemented as a proper transformer so it can live inside a
    sklearn Pipeline and be saved/loaded as a single artifact.

    Expects a DataFrame with columns:
      CreditScore, Geography, Gender, Age, Tenure, Balance,
      NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
    """

    CREDIT_BINS = [300, 580, 670, 740, 800, 900]
    CREDIT_LABELS = ["Poor", "Fair", "Good", "Very Good", "Excellent"]

    AGE_BINS = [18, 30, 40, 50, 60, 100]
    AGE_LABELS = ["18-30", "31-40", "41-50", "51-60", "60+"]

    GEOGRAPHIES = ["France", "Germany", "Spain"]

    def fit(self, X, y=None):
        # Nothing is learned from data - bins/labels/mappings are fixed.
        # We still run a transform here so get_feature_names_out() and any
        # downstream introspection (e.g. feature importance plots) work.
        self.feature_names_out_ = list(self._engineer(X).columns)
        return self

    def transform(self, X):
        return self._engineer(X)

    def get_feature_names_out(self, input_features=None):
        return getattr(self, "feature_names_out_", None)

    def _engineer(self, X):
        df = X.copy()

        # Gender -> binary
        df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

        # Geography -> one-hot
        for geo in self.GEOGRAPHIES:
            df[f"Geography_{geo}"] = (df["Geography"] == geo).astype(int)
        df.drop(columns=["Geography"], inplace=True)

        # CreditScore -> CreditCategory one-hot
        df["CreditCategory"] = pd.cut(
            df["CreditScore"], bins=self.CREDIT_BINS, labels=self.CREDIT_LABELS
        )
        for label in self.CREDIT_LABELS:
            df[f"CreditCategory_{label}"] = (df["CreditCategory"] == label).astype(int)
        df.drop(columns=["CreditCategory"], inplace=True)

        # Age -> AgeGroup one-hot, drop raw Age
        df["AgeGroup"] = pd.cut(
            df["Age"], bins=self.AGE_BINS, labels=self.AGE_LABELS, include_lowest=True
        )
        for label in self.AGE_LABELS:
            df[f"AgeGroup_{label}"] = (df["AgeGroup"] == label).astype(int)
        df.drop(columns=["AgeGroup", "Age"], inplace=True)

        # Engineered ratio feature
        df["BalancePerProduct"] = df["Balance"] / df["NumOfProducts"].replace(0, 1)

        return df
