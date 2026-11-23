
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Digital Addiction Risk Predictor",
    page_icon="📱",
    layout="wide"
)

# Load trained model and scaler
model = joblib.load("models/logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Header
st.title("📱 Digital Addiction Risk Predictor")
st.write(
    "Analyze smartphone usage patterns and estimate "
    "the risk of excessive smartphone usage."
)

st.info(
    "This is an educational machine-learning risk estimate, "
    "not a clinical diagnosis."
)

# Sidebar
st.sidebar.title("⚙️ Prediction Settings")

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.1
)

st.sidebar.caption(
    "Adjust the threshold to change how the model "
    "classifies high-risk usage patterns."
)

# Input section
st.header("Enter Smartphone Usage Details")

col1, col2 = st.columns(2)

with col1:
    screen_time = st.slider(
        "Daily Screen Time (hours)",
        0.0, 24.0, 6.0, 0.5
    )

    night_usage = st.slider(
        "Night-time Usage (hours)",
        0.0, 12.0, 2.0, 0.5
    )

with col2:
    app_sessions = st.number_input(
        "App Sessions per Day",
        min_value=0,
        max_value=500,
        value=80
    )

    social_media = st.slider(
        "Social Media Usage (hours)",
        0.0, 24.0, 3.0, 0.5
    )

# Prediction
if st.button("Predict Addiction Risk", type="primary"):

    input_data = pd.DataFrame([{
        "daily_screen_time": screen_time,
        "night_usage": night_usage,
        "app_sessions": app_sessions,
        "social_media_usage": social_media
    }])

    # Apply saved scaler
    scaled_data = scaler.transform(input_data)

    # Predict probability
    probability = model.predict_proba(scaled_data)[0][1]

    # Apply selected threshold
    prediction = int(probability >= threshold)

    st.divider()
    st.header("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "High Risk Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{threshold * 100:.0f}%"
        )

    st.progress(float(probability))

    if prediction == 1:
        st.error("⚠️ High Risk Usage Pattern")
    else:
        st.success("✅ Low Risk Usage Pattern")

    st.write("### Probability Interpretation")

    st.write(
        f"The model estimates a {probability * 100:.2f}% "
        "probability of the positive class."
    )

    st.write(
        f"The selected threshold is {threshold:.2f}. "
        "The classification changes when the threshold changes."
    )

    st.caption(
        "Risk estimates reflect the labels and patterns "
        "in the training dataset."
    )

# Model evaluation section
st.divider()
st.header("📊 Model Evaluation")

col1, col2 = st.columns(2)

with col1:
    if os.path.exists("outputs/roc_curve.png"):
        st.image(
            "outputs/roc_curve.png",
            caption="ROC Curve"
        )

with col2:
    if os.path.exists("outputs/sigmoid_curve.png"):
        st.image(
            "outputs/sigmoid_curve.png",
            caption="Sigmoid Function"
        )

if os.path.exists("outputs/threshold_comparison.csv"):
    st.subheader("Threshold Comparison")
    results = pd.read_csv(
        "outputs/threshold_comparison.csv"
    )
    st.dataframe(results, use_container_width=True)

if os.path.exists("outputs/regularization_comparison.csv"):
    st.subheader("L1 vs L2 Regularization")
    regularization = pd.read_csv(
        "outputs/regularization_comparison.csv"
    )
    st.dataframe(
        regularization,
        use_container_width=True
    )

st.divider()
st.caption(
    "RCA Unit 2 Project | Binary Classification "
    "using Logistic Regression"
)


st.divider()
st.header("📈 Additional RCA Analysis")

for file, title in [
    ("confusion_matrix.png", "Confusion Matrix"),
    ("decision_behaviour.png", "Decision Behaviour"),
]:
    path = f"outputs/{file}"

    if os.path.exists(path):
        st.subheader(title)
        st.image(path, use_container_width=True)
