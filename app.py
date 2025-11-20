from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================================
# 1) Configuration
# =========================================
PACKAGE_FILENAME = "website_model_package.pkl"

st.set_page_config(
    page_title="Risk Prediction Web App",
    page_icon="🩺",
    layout="wide",
)

# =========================================
# 2) Load website model package (joblib)
# =========================================
@st.cache_resource
def load_website_package(filename: str = PACKAGE_FILENAME) -> Dict[str, Any]:
    pkg_path = Path(filename)
    if not pkg_path.exists():
        raise FileNotFoundError(
            f"Model package file '{filename}' not found in working directory."
        )

    # IMPORTANT: use joblib.load, NOT pickle.load
    pkg = joblib.load(pkg_path)

    required_keys = ["pipeline", "feature_names"]
    for k in required_keys:
        if k not in pkg:
            raise KeyError(
                f"Key '{k}' not found in website package. "
                f"Available keys: {list(pkg.keys())}"
            )
    return pkg


# =========================================
# 3) Small helpers
# =========================================
def predict_with_pipeline(pipeline, X: pd.DataFrame, threshold: float = 0.5):
    """
    Generic prediction for sklearn Pipeline:
      - If predict_proba exists: return (label, proba1)
      - Else: (label, None)
    """
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        proba_1 = float(proba[0, 1])
        label = int(proba_1 >= threshold)
        return label, proba_1
    else:
        y_pred = pipeline.predict(X)
        label = int(y_pred[0])
        return label, None


# =========================================
# 4) Load package and extract info
# =========================================
try:
    PKG = load_website_package()
except Exception as e:
    st.error(
        f"❌ Could not load model package.\n\n"
        f"Error: {type(e).__name__}: {e}\n\n"
        f"Make sure `{PACKAGE_FILENAME}` is in the same folder as `app.py`, "
        "and that it was saved with `joblib.dump`."
    )
    st.stop()

pipeline = PKG["pipeline"]
feature_names = list(PKG["feature_names"])
feature_ranges = PKG.get("feature_ranges", {})
performance = PKG.get("performance", {})
model_type = PKG.get("model_type", "Unknown model")
is_binary = PKG.get("is_binary", True)
classes = PKG.get("classes", [])

# =========================================
# 5) Sidebar: model info + logos + threshold
# =========================================

st.sidebar.markdown("**Powered by**")

LOGO_WIDTH = 180  # adjust as you like

# Make sure these filenames exist in your repo
st.sidebar.image("images.png", width=LOGO_WIDTH)
st.sidebar.image("images.jpeg", width=LOGO_WIDTH)
st.sidebar.image("qeeri_logo.png", width=LOGO_WIDTH)


st.sidebar.header("Model info")
st.sidebar.write(f"**Model type:** {model_type}")

if performance:
    st.sidebar.write(
        f"**Accuracy:** {performance.get('accuracy', float('nan')):.3f}"
    )
    st.sidebar.write(
        f"**F1-score:** {performance.get('f1_score', float('nan')):.3f}"
    )

st.sidebar.write(f"**Binary classification:** {is_binary}")
st.sidebar.write(f"**Classes:** {classes}")

# Decision threshold slider (only once!)
threshold = st.sidebar.slider(
    "Decision threshold (for probability-based models)",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
)

# st.sidebar.write("📊 Features used:")
# st.sidebar.write(feature_names)

# =========================================
# 6) Main title
# =========================================
st.title("🩺 Risk Prediction Web App")

st.markdown(
    """
This app uses your trained model from `website_model_package.pkl`  
to estimate risk based on clinical and hoof-related features.
"""
)

# =========================================
# 7) Input form
# =========================================
st.subheader("Enter input features")

feature_config = {
    "Age(years)":      dict(label="Age (years)", min=0.0, max=40.0, default=10.0, step=1.0),
    "Sex":             dict(label="Sex (encoded)", min=0.0, max=3.0, default=1.0, step=1.0),
    "HeartRate":       dict(label="Heart Rate", min=0.0, max=150.0, default=44.0, step=1.0),
    "Respiratoryrate": dict(label="Respiratory rate", min=0.0, max=120.0, default=20.0, step=1.0),
    "Rectaltemperature": dict(label="Rectal temperature (°C)", min=30.0, max=42.0, default=37.5, step=0.1),
    "Gutsounds":       dict(label="Gutsounds (encoded)", min=-1.0, max=3.0, default=0.0, step=1.0),
    "Digitalpulses":   dict(label="Digital pulses (encoded)", min=0.0, max=3.0, default=0.0, step=1.0),
    "Bodyweight(kg)":  dict(label="Body weight (kg)", min=50.0, max=900.0, default=450.0, step=10.0),
    "BodyConditionScoring(outof9)": dict(label="Body Condition Score (out of 9)", min=1.0, max=9.0, default=5.0, step=0.5),
    "LengthRF":        dict(label="LengthRF", min=0.0, max=50.0, default=10.0, step=0.1),
    "LengthLF":        dict(label="LengthLF", min=0.0, max=50.0, default=10.0, step=0.1),
    "LengthRH":        dict(label="LengthRH", min=0.0, max=50.0, default=10.0, step=0.1),
    "WidthRF":         dict(label="WidthRF", min=0.0, max=50.0, default=10.0, step=0.1),
    "WidthLF":         dict(label="WidthLF", min=0.0, max=50.0, default=10.0, step=0.1),
    "WidthRH":         dict(label="WidthRH", min=0.0, max=50.0, default=10.0, step=0.1),
    "HTRF":            dict(label="HTRF (encoded)", min=-1.0, max=3.0, default=0.0, step=1.0),
    "HTRH":            dict(label="HTRH (encoded)", min=-1.0, max=3.0, default=0.0, step=1.0),
    "LERF":            dict(label="LERF (encoded)", min=0.0, max=4.0, default=0.0, step=1.0),
}

with st.form("input_form"):
    col1, col2 = st.columns(2)
    values: Dict[str, float] = {}

    for i, feat in enumerate(feature_names):
        cfg = feature_config.get(feat, {})
        frange = feature_ranges.get(feat, {})

        # Fallbacks if ranges not provided
        min_val = cfg.get("min", float(frange.get("min", 0.0)))
        max_val = cfg.get("max", float(frange.get("max", 100.0)))
        if max_val <= min_val:
            max_val = min_val + 1.0

        default_val = cfg.get(
            "default",
            float((min_val + max_val) / 2.0)
        )
        step_val = cfg.get("step", 1.0)

        col = col1 if i % 2 == 0 else col2
        val = col.number_input(
            cfg.get("label", feat),
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            step=float(step_val),
        )
        values[feat] = float(val)

    submitted = st.form_submit_button("Predict")

# =========================================
# 8) Run prediction
# =========================================
if submitted:
    X = pd.DataFrame([[values[f] for f in feature_names]], columns=feature_names)

    label, proba = predict_with_pipeline(pipeline, X, threshold=threshold)

    risk_text = "High risk" if label == 1 else "Low / intermediate risk"

    st.markdown("### 🔍 Prediction result")
    st.write(f"**Predicted class:** `{label}`")
    st.write(f"**Risk interpretation:** **{risk_text}**")

    if proba is not None:
        st.write(f"**Probability of class 1:** `{proba:.3f}` (threshold = {threshold:.2f})")

    st.markdown("#### Input used")
    st.dataframe(X, use_container_width=True)
