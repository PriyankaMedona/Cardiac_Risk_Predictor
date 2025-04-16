import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Load Feature & Target Info
# ------------------------------
@st.cache_data
def load_data():
    X = pd.read_csv("features.csv")
    y = pd.read_csv("target.csv")
    return X, y

@st.cache_data
def load_models():
    model_files = {
        "LightGBM": "lightgbm_model.pkl",
        "MLP": "mlp_model.pkl",
        "XGBoost": "xgboost_model.pkl",
        "CatBoost": "catboost_model.pkl",
        "Random Forest": "random_forest_model.pkl"
    }
    models = {name: joblib.load(path) for name, path in model_files.items()}
    return models

X, y = load_data()
selected_features = X.columns.tolist()
disease_labels = y.columns.tolist()
models = load_models()

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Cardiac Disease Risk Predictor", layout="wide")
st.title("💓 Cardiac Disease Risk Predictor with Soft Voting Ensemble")

st.markdown("""
Enter your health details below to assess your cardiac disease risk using five advanced models (LightGBM, XGBoost, CatBoost, MLP, Random Forest) combined via soft voting.
""")

# ------------------------------
# Collect Input from User
# ------------------------------
def user_input_form():
    input_data = {}
    with st.form("user_form"):
        for feature in selected_features:
            input_data[feature] = st.number_input(f"{feature}", step=0.01)
        submitted = st.form_submit_button("Predict Risk")
        if submitted:
            return pd.DataFrame([input_data])
    return None

user_df = user_input_form()

# ------------------------------
# Predict & Display Results
# ------------------------------
if user_df is not None:
    st.subheader("🔍 Prediction Results")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    user_scaled = scaler.transform(user_df)

    predictions = []
    for name, model in models.items():
        preds = model.predict_proba(user_scaled)
        if isinstance(preds, list):  # Some models return list of arrays
            preds = np.array([p[:, 1] for p in preds]).T
        elif preds.shape[1] == 2 and len(disease_labels) == 1:
            preds = preds[:, 1].reshape(1, -1)
        predictions.append(preds)

    avg_pred = np.mean(predictions, axis=0)
    risk_scores = (avg_pred * 100).flatten()
    risk_data = dict(zip(disease_labels, risk_scores))

    highest_disease = max(risk_data, key=risk_data.get)
    highest_score = risk_data[highest_disease]

    if highest_score >= 75:
        level = "🔴 High Risk Chance"
    elif 55 <= highest_score < 75:
        level = "🟠 Mid Level Risk Chance"
    elif 35 <= highest_score < 55:
        level = "🟡 Low Level Risk Chance"
    elif all(score < 35 for score in risk_scores):
        highest_disease = "🎉 Super Healthy"
        level = "🟢 No possible chance for any disease"
        highest_score = 0

    # Display Top Prediction
    st.success(f"**Top Risk Disease:** {highest_disease}")
    if highest_disease != "🎉 Super Healthy":
        st.write(f"**Risk Score:** `{highest_score:.2f}%` → {level}")
    else:
        st.write(level)

    # Display All Scores
    st.subheader("📊 Risk Scores for All Diseases")
    risk_df = pd.DataFrame(risk_data.items(), columns=["Disease", "Risk Score (%)"])
    st.dataframe(risk_df.style.format({"Risk Score (%)": "{:.2f}"}), use_container_width=True)

    # Save to CSV
    result_row = user_df.copy()
    for disease, score in risk_data.items():
        result_row[f"{disease}_score"] = score
    result_row["Top_Disease"] = highest_disease
    result_row["Risk_Level"] = level
    result_row.to_csv("predictions.csv", mode="a", header=False, index=False)
