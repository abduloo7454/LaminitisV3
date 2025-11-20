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
    page_title="Laminitis Risk Prediction",
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
# 3) Helper for prediction
# =========================================
def predict_with_pipeline(pipeline, X: pd.DataFrame, threshold: float = 0.5):
    """
    For a sklearn Pipeline:
      - If predict_proba exists: return (label, p(class=1))
      - Else: return (label, None)
    """
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        p1 = float(proba[0, 1])
        label = int(p1 >= threshold)
        return label, p1
    else:
        y_pred = pipeline.predict(X)
        label = int(y_pred[0])
        return label, None


# =========================================
# 4) Load model package
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
# 5) Sidebar: Powered by + navigation
# =========================================
# ---- Powered by logos at TOP ----
st.sidebar.markdown("**Powered by**")
LOGO_WIDTH = 160  # adjust to taste

# Make sure these filenames exist in your repo
st.sidebar.image("images.png",     width=LOGO_WIDTH)
st.sidebar.image("images.jpeg",    width=LOGO_WIDTH)
st.sidebar.image("qeeri_logo.png", width=LOGO_WIDTH)

st.sidebar.markdown("---")

# ---- Navigation between pages ----
page = st.sidebar.radio(
    "Go to",
    ["Risk Prediction", "About the Project", "Meet the Team"],
)


# =========================================
# 6) PAGE 1 – Risk Prediction
# =========================================
if page == "Risk Prediction":
    # Sidebar model info + threshold
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

    threshold = st.sidebar.slider(
        "Decision threshold (for probability-based models)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )

    st.title("🩺 Laminitis Risk Prediction Web App")

    st.markdown(
        """
This tool uses a machine-learning model trained on clinical and hoof-related
features to estimate the probability that a horse is at risk of laminitis.
Please enter the available information below.
"""
    )

    st.subheader("Enter input features")

    # ----- feature config -----
    feature_config = {
        "Age(years)":      dict(label="Age (years)", min=0.0, max=40.0, default=10.0, step=1.0),
        "Sex":             dict(label="Sex (encoded)", min=0.0, max=3.0, default=1.0, step=1.0),
        "HeartRate":       dict(label="Heart rate (beats/min)", min=0.0, max=150.0, default=44.0, step=1.0),
        "Respiratoryrate": dict(label="Respiratory rate (breaths/min)", min=0.0, max=120.0, default=20.0, step=1.0),
        "Rectaltemperature": dict(label="Rectal temperature (°C)", min=30.0, max=42.0, default=37.5, step=0.1),
        "Gutsounds":       dict(label="Gutsounds (encoded)", min=-1.0, max=3.0, default=0.0, step=1.0),
        "Digitalpulses":   dict(label="Digital pulses (encoded)", min=0.0, max=3.0, default=0.0, step=1.0),
        "Bodyweight(kg)":  dict(label="Body weight (kg)", min=50.0, max=900.0, default=450.0, step=10.0),
        "BodyConditionScoring(outof9)": dict(label="Body Condition Score (1–9)", min=1.0, max=9.0, default=5.0, step=0.5),
        "LengthRF":        dict(label="Length RF", min=0.0, max=50.0, default=10.0, step=0.1),
        "LengthLF":        dict(label="Length LF", min=0.0, max=50.0, default=10.0, step=0.1),
        "LengthRH":        dict(label="Length RH", min=0.0, max=50.0, default=10.0, step=0.1),
        "WidthRF":         dict(label="Width RF", min=0.0, max=50.0, default=10.0, step=0.1),
        "WidthLF":         dict(label="Width LF", min=0.0, max=50.0, default=10.0, step=0.1),
        "WidthRH":         dict(label="Width RH", min=0.0, max=50.0, default=10.0, step=0.1),
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

            min_val = cfg.get("min", float(frange.get("min", 0.0)))
            max_val = cfg.get("max", float(frange.get("max", 100.0)))
            if max_val <= min_val:
                max_val = min_val + 1.0

            default_val = cfg.get(
                "default",
                float((min_val + max_val) / 2.0),
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

    if submitted:
        X = pd.DataFrame([[values[f] for f in feature_names]], columns=feature_names)
        label, proba = predict_with_pipeline(pipeline, X, threshold=threshold)

        risk_text = "High risk" if label == 1 else "Low / intermediate risk"

        st.markdown("### 🔍 Prediction result")
        st.write(f"**Predicted class:** `{label}`")
        st.write(f"**Risk interpretation:** **{risk_text}**")

        if proba is not None:
            st.write(
                f"**Probability of class 1:** `{proba:.3f}` "
                f"(threshold = {threshold:.2f})"
            )

        st.markdown("#### Input used")
        st.dataframe(X, use_container_width=True)


# =========================================
# 7) PAGE 2 – About the Project
# =========================================
elif page == "About the Project":
    st.title("🔍 About the Laminitis Risk Project")

    st.markdown(
        """
This project aims to develop a practical, data-driven tool to support
early identification of horses at risk of laminitis.

The underlying model was trained on clinical examination findings and
hoof-measurement data collected from horses examined at the Equine
Veterinary Medical Center and collaborating institutions.

Key goals of the project:

- Translate machine-learning research into a simple web-based risk tool.
- Help clinicians combine multiple risk factors into a single probability estimate.
- Provide a transparent framework that can be improved as more data become available.

This online tool is intended to complement, not replace, the judgement
of experienced veterinarians. Final clinical decisions should always
consider the full history, examination findings, and imaging where relevant.
"""
    )


# =========================================
# 8) PAGE 3 – Team
# =========================================
elif page == "Meet the Team":
    st.title("👥 Project Team")

    st.markdown(
        """
This work is the result of a collaborative effort between veterinarians,
data scientists, and researchers.

**Clinical and Veterinary Contributors**

- Equine clinicians providing case assessment, data collection,
  and interpretation of laminitis risk.
- Specialists overseeing case selection, inclusion criteria,
  and validation of the risk categories.

**Data Science and Modelling**

- Researchers responsible for data preprocessing, feature engineering,
  model development, and evaluation.
- Team members implementing the web interface and deployment workflow.

**Institutional Support**

- Qatar Biomedical Research Institute / Hamad Bin Khalifa University  
- Equine Veterinary Medical Center  
- Partner institutes contributing expertise, infrastructure, and funding.

You can adapt this section with individual names, roles, and contact
details as needed for internal or public use.
"""
    )
