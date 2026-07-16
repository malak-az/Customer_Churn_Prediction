import joblib

pipeline = joblib.load("models/churn_model_pipeline.pkl")

print("Keys:", pipeline.keys())
print()

print("Feature Columns:")
for col in pipeline["feature_columns"]:
    print(col)

print("\nThreshold:", pipeline["threshold"])
print("\nModel:", type(pipeline["model"]))
print("Scaler:", type(pipeline["scaler"]))