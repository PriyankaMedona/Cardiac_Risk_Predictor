import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set page config (MUST BE FIRST)
st.set_page_config(page_title="Cardiac Disease Risk Predictor", layout="wide")

# -------------------------
# Load Feature & Target Data
# -------------------------
X = pd.read_csv("/content/features.csv")
y = pd.read_csv("/content/target.csv")

# -------------------------
# Extract Selected Features (after feature selection)
# -------------------------
selected_features = X.columns.tolist()
disease_labels = y.columns.tolist()

# -------------------------
# Simulate Single User Input
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
user_input = pd.DataFrame(X_scaled[:1], columns=selected_features)

# -------------------------
# Load Top 5 Models
# -------------------------
model_files = {
    "LightGBM": "lightgbm_model.pkl",
    "MLP": "mlp_model.pkl",
    "XGBoost": "xgboost_model.pkl",
    "CatBoost": "catboost_model.pkl",
    "Random Forest": "random_forest_model.pkl"
}

predictions = []

for model_name, file_name in model_files.items():
    model = joblib.load(file_name)
    preds = model.predict_proba(user_input)

    if isinstance(preds, list):
        preds = np.array([p[:, 1] for p in preds]).T
    elif preds.shape[1] == 2 and len(disease_labels) == 1:
        preds = preds[:, 1].reshape(1, -1)

    predictions.append(preds)

# -------------------------
# Soft Voting
# -------------------------
avg_pred = np.mean(predictions, axis=0)
risk_scores = (avg_pred * 100).flatten()

# -------------------------
# Determine Risk
# -------------------------
risk_data = dict(zip(disease_labels, risk_scores))
highest_disease = max(risk_data, key=risk_data.get)
highest_score = risk_data[highest_disease]

if highest_score >= 75:
    level = "High Risk Chance"
elif 55 <= highest_score < 75:
    level = "Mid Level Risk Chance"
elif 35 <= highest_score < 55:
    level = "Low Level Risk Chance"
elif all(score < 35 for score in risk_scores):
    highest_disease = "Super Healthy 🎉"
    highest_score = 0
    level = "No possible chance for any disease"

# -------------------------
# Display Results
# -------------------------
st.write(f"\n✅ Risk Analysis Complete!")
st.write(f"Top Risk Disease: {highest_disease}")
if highest_disease != "Super Healthy 🎉":
    st.write(f"Risk Score: {highest_score:.2f}% → {level}")
else:
    st.write(level)

st.write("\nFull Risk Scores:")
for disease, score in risk_data.items():
    st.write(f"{disease}: {score:.2f}%")
