from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT)) 
    
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from ml_model.train_model import MODEL_PATH
from rule_based_system.rules import assess_patient
from utils.data_processing import CLEANED_DATA_PATH, RAW_DATA_PATH, clean_dataset, load_dataset

st.set_page_config(page_title="Heart Disease Detection", page_icon="❤️", layout="wide")


@st.cache_data
def get_clean_data() -> pd.DataFrame:
    source_path = CLEANED_DATA_PATH if Path(CLEANED_DATA_PATH).exists() else RAW_DATA_PATH
    return clean_dataset(load_dataset(source_path))


@st.cache_resource
def load_model():
    if not Path(MODEL_PATH).exists():
        return None
    return joblib.load(MODEL_PATH)


def render_sidebar_inputs() -> dict:
    st.sidebar.header("Patient Inputs")
    return {
        "age": st.sidebar.slider("Age", 25, 80, 54),
        "sex": st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male"),
        "cp": st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3]),
        "trestbps": st.sidebar.slider("Resting Blood Pressure", 90, 200, 130),
        "chol": st.sidebar.slider("Cholesterol", 100, 400, 240),
        "fbs": st.sidebar.selectbox("Fasting Blood Sugar > 120", [0, 1]),
        "restecg": st.sidebar.selectbox("Resting ECG", [0, 1, 2]),
        "thalach": st.sidebar.slider("Max Heart Rate", 70, 210, 150),
        "exang": st.sidebar.selectbox("Exercise-Induced Angina", [0, 1]),
        "oldpeak": st.sidebar.slider("Oldpeak", 0.0, 6.5, 1.0, step=0.1),
        "slope": st.sidebar.selectbox("Slope", [0, 1, 2]),
        "ca": st.sidebar.selectbox("Major Vessels Colored", [0, 1, 2, 3, 4]),
        "thal": st.sidebar.selectbox("Thal", [0, 1, 2, 3]),
    }


def main() -> None:
    st.title("Heart Disease Detection Dashboard")
    st.caption("Compare a machine learning prediction with a transparent rule-based explanation.")

    clean_df = get_clean_data()
    model = load_model()
    patient = render_sidebar_inputs()
    patient_df = pd.DataFrame([patient])

    left, right = st.columns(2)

    with left:
        st.subheader("Decision Tree Result")
        if model is None:
            st.warning("Train the model first with `python ml_model/train_model.py` to enable ML predictions.")
        else:
            prediction = int(model.predict(patient_df)[0])
            probability = float(model.predict_proba(patient_df)[0][1])
            st.metric("Prediction", "Heart Disease" if prediction == 1 else "No Heart Disease")
            st.metric("Probability", f"{probability:.2%}")

    with right:
        st.subheader("Expert System Result")
        expert_result = assess_patient(patient)
        st.metric("Risk Label", expert_result["risk_label"].title())
        st.metric("Risk Score", expert_result["score"])
        st.write("Triggered rules:")
        for reason in expert_result["reasons"] or ["No rules fired"]:
            st.write(f"- {reason}")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Target Distribution")
        target_counts = clean_df["target"].value_counts().rename(index={0: "No Disease", 1: "Disease"})
        st.bar_chart(target_counts)

    with col_b:
        st.subheader("Age vs Cholesterol")
        fig, axis = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=clean_df, x="age", y="chol", hue="target", palette="Set2", ax=axis)
        axis.set_title("Age vs Cholesterol")
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    heatmap_fig, heatmap_ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(clean_df.corr(numeric_only=True), cmap="coolwarm", annot=False, ax=heatmap_ax)
    st.pyplot(heatmap_fig)


if __name__ == "__main__":
    main()
