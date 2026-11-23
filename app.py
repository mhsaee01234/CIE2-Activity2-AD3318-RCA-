
import streamlit as st
import pandas as pd
import joblib
import os

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Digital Addiction Risk Predictor",
    page_icon="📱",
    layout="wide"
)

# =====================================================
# 2. PROJECT PATHS
# =====================================================

MODEL_PATH = "models/logistic_model.pkl"
SCALER_PATH = "models/scaler.pkl"
OUTPUT_DIR = "outputs"

# =====================================================
# 3. LOAD TRAINED MODEL AND SCALER
# =====================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run train_model.py first."
        )

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Scaler not found. Check your prepare_data.py pipeline."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


try:
    model, scaler = load_model()

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# =====================================================
# 4. HEADER
# =====================================================

st.title("📱 Digital Addiction Risk Predictor")

st.write(
    "Analyze smartphone usage patterns and estimate "
    "the risk of excessive smartphone usage."
)

st.info(
    "This is an educational machine-learning risk estimate, "
    "not a clinical diagnosis."
)

st.divider()


# =====================================================
# 5. SIDEBAR SETTINGS
# =====================================================

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


# =====================================================
# 6. INPUT SECTION
# =====================================================

st.header("Enter Smartphone Usage Details")

col1, col2 = st.columns(2)

with col1:

    screen_time = st.slider(
        "Daily Screen Time (hours)",
        min_value=0.0,
        max_value=24.0,
        value=6.0,
        step=0.5
    )

    night_usage = st.slider(
        "Night-time Usage (hours)",
        min_value=0.0,
        max_value=12.0,
        value=2.0,
        step=0.5
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
        min_value=0.0,
        max_value=24.0,
        value=3.0,
        step=0.5
    )


# =====================================================
# 7. PREDICTION
# =====================================================

if st.button(
    "Predict Addiction Risk",
    type="primary",
    use_container_width=True
):

    # Prepare input features

    input_data = pd.DataFrame([{
        "daily_screen_time": screen_time,
        "night_usage": night_usage,
        "app_sessions": app_sessions,
        "social_media_usage": social_media
    }])

    try:

        # Ensure feature order matches training

        input_data = input_data[
            scaler.feature_names_in_
        ]

        # Apply saved scaler

        scaled_data = scaler.transform(input_data)

        # Calculate probability

        probability = model.predict_proba(
            scaled_data
        )[0][1]

        # Apply selected threshold

        prediction = int(
            probability >= threshold
        )

        # Save prediction in session state

        st.session_state["probability"] = probability
        st.session_state["prediction"] = prediction
        st.session_state["threshold"] = threshold

    except Exception as e:

        st.error(f"Prediction error: {e}")


# =====================================================
# 8. DISPLAY PREDICTION RESULTS
# =====================================================

if "probability" in st.session_state:

    probability = st.session_state["probability"]
    prediction = st.session_state["prediction"]
    selected_threshold = st.session_state["threshold"]

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
            f"{selected_threshold * 100:.0f}%"
        )

    st.progress(float(probability))

    if prediction == 1:

        st.error("⚠️ High Risk Usage Pattern")

        st.write(
            "Your entered smartphone usage pattern "
            "has been classified as high risk by the model."
        )

    else:

        st.success("✅ Low Risk Usage Pattern")

        st.write(
            "Your entered smartphone usage pattern "
            "has been classified as low risk by the model."
        )

    st.subheader("Probability Interpretation")

    st.write(
        f"The model estimates a "
        f"{probability * 100:.2f}% probability "
        "of the positive class."
    )

    st.write(
        f"The selected threshold is "
        f"{selected_threshold:.2f}. "
        "The classification changes when the threshold changes."
    )

    st.caption(
        "Risk estimates reflect the labels and patterns "
        "in the training dataset."
    )


# =====================================================
# 9. MODEL EVALUATION
# =====================================================

st.divider()

st.header("📊 Model Evaluation")

# ROC Curve and Sigmoid Curve

col1, col2 = st.columns(2)

with col1:

    st.subheader("ROC Curve")

    path = os.path.join(
        OUTPUT_DIR,
        "roc_curve.png"
    )

    if os.path.exists(path):

        st.image(
            path,
            use_container_width=True
        )

    else:
        st.warning("ROC curve not found.")


with col2:

    st.subheader("Sigmoid Function")

    path = os.path.join(
        OUTPUT_DIR,
        "sigmoid_curve.png"
    )

    if os.path.exists(path):

        st.image(
            path,
            use_container_width=True
        )

    else:
        st.warning("Sigmoid curve not found.")


# =====================================================
# 10. MODEL PERFORMANCE METRICS
# =====================================================

st.subheader("Model Performance Metrics")

metrics_path = os.path.join(
    OUTPUT_DIR,
    "model_metrics.csv"
)

if os.path.exists(metrics_path):

    metrics = pd.read_csv(metrics_path)

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning("Model metrics not found.")


# =====================================================
# 11. THRESHOLD COMPARISON
# =====================================================

st.subheader("Threshold Comparison")

threshold_path = os.path.join(
    OUTPUT_DIR,
    "threshold_comparison.csv"
)

if os.path.exists(threshold_path):

    threshold_results = pd.read_csv(
        threshold_path
    )

    st.dataframe(
        threshold_results,
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning("Threshold comparison not found.")


# =====================================================
# 12. REGULARIZATION COMPARISON
# =====================================================

st.subheader("L1 vs L2 Regularization")

regularization_path = os.path.join(
    OUTPUT_DIR,
    "regularization_comparison.csv"
)

if os.path.exists(regularization_path):

    regularization = pd.read_csv(
        regularization_path
    )

    st.dataframe(
        regularization,
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning("Regularization results not found.")


# =====================================================
# 13. CONFUSION MATRIX
# =====================================================

st.divider()

st.header("📈 Additional RCA Analysis")

confusion_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

if os.path.exists(confusion_path):

    st.subheader("Confusion Matrix")

    st.image(
        confusion_path,
        use_container_width=True
    )

else:
    st.warning("Confusion matrix not found.")


# =====================================================
# 14. DECISION BEHAVIOUR
# =====================================================

decision_path = os.path.join(
    OUTPUT_DIR,
    "decision_behaviour.png"
)

if os.path.exists(decision_path):

    st.subheader("Decision Behaviour")

    st.image(
        decision_path,
        use_container_width=True
    )

else:
    st.warning("Decision behaviour graph not found.")


# =====================================================
# 15. FOOTER
# =====================================================

st.divider()

st.caption(
    "RCA Unit 2 Project | Binary Classification "
    "using Logistic Regression"
)

st.caption(
    "Digital Addiction Risk Predictor | "
    "Educational Machine Learning Application"
)
