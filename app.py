import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st


# =====================================================
# 1) Configuration
# =====================================================

PACKAGE_FILENAME = "website_model_package.pkl"


# Default feature list (order matters!)
DEFAULT_FEATURE_NAMES: List[str] = [
    "Age(years)",
    "Sex",
    "HeartRate",
    "Respiratoryrate",
    "Rectaltemperature",
    "Gutsounds",
    "Digitalpulses",
    "Bodyweight(kg)",
    "BodyConditionScoring(outof9)",
    "LengthRF",
    "LengthLF",
    "LengthRH",
    "WidthRF",
    "WidthLF",
    "WidthRH",
    "HTRF",
    "HTRH",
    "LERF",
]


# =====================================================
# 2) Load package (models, scaler, feature_names)
# =====================================================

@st.cache_resource
def load_model_package(
    filename: str = PACKAGE_FILENAME,
) -> Tuple[Dict[str, Any], Optional[Any], List[str], Dict[str, Any]]:
    """
    Load the pickle package and extract:
      - models: dict[str, model-like object]
      - scaler: optional scaler / transformer
      - feature_names: list of feature names (if present; otherwise DEFAULT_FEATURE_NAMES)
      - raw_pkg: raw dictionary returned by pickle.load()
    """
    pkg_path = Path(filename)

    if not pkg_path.exists():
        raise FileNotFoundError(
            f"Model package file '{filename}' not found in working directory."
        )

    # Load pickle safely (assuming same environment as training)
    with pkg_path.open("rb") as f:
        pkg = pickle.load(f)

    if not isinstance(pkg, dict):
        # If user saved a single object, wrap it into a dict
        pkg = {"model": pkg}

    # ---- Extract models ----
    models: Dict[str, Any] = {}

    if "models" in pkg and isinstance(pkg["models"], dict):
        # user stored multiple models in a dict
        models = pkg["models"]
    elif "model" in pkg:
        # single model
        models = {"Main model": pkg["model"]}
    elif "pipeline" in pkg:
        # single sklearn Pipeline
        models = {"Pipeline model": pkg["pipeline"]}
    else:
        # last resort: any key that looks like a model
        for k, v in pkg.items():
            if hasattr(v, "predict"):
                models[k] = v

    if not models:
        raise KeyError(
            "No model found in the package. "
            "Expected keys like 'models', 'model', or 'pipeline'."
        )

    # ---- Extract scaler (optional) ----
    scaler = pkg.get("scaler", None)

    # ---- Extract feature names ----
    feature_names = pkg.get("feature_names", None)
    if feature_names is None:
        # fall back to default list
        feature_names = DEFAULT_FEATURE_NAMES

    # Ensure feature_names is a list of strings
    feature_names = [str(f) for f in feature_names]

    return models, scaler, feature_names, pkg


# =====================================================
# 3) Small helper functions
# =====================================================

def build_input_dataframe(
    feature_names: List[str],
    form_values: Dict[str, float],
) -> pd.DataFrame:
    """
    Build a single-row DataFrame matching the feature_names order.
    """
    row = {f: form_values[f] for f in feature_names}
    return pd.DataFrame([row])


def model_predict(
    model: Any,
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> Tuple[int, Optional[float]]:
    """
    Generic wrapper for predicting with a model.

    If model has `.predict_proba`, return:
        label (0/1), probability_of_class_1
    Otherwise return:
        label, None
    """
    # For sklearn Pipelines, just call them directly with X
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)  # type: ignore[attr-defined]
            proba_1 = float(proba[0, 1])
            label = int(proba_1 >= threshold)
            return label, proba_1
        else:
            # no predict_proba -> classification only
            y_pred = model.predict(X)
            label = int(y_pred[0])
            return label, None
    except Exception as e:
        # Fallback: try converting X to numpy
        X_np = X.values
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_np)  # type: ignore[attr-defined]
            proba_1 = float(proba[0, 1])
            label = int(proba_1 >= threshold)
            return label, proba_1
        else:
            y_pred = model.predict(X_np)
            label = int(y_pred[0])
            return label, None


# =====================================================
# 4) Streamlit layout
# =====================================================

st.set_page_config(
    page_title="Risk Prediction Website",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Risk Prediction Web App")

st.markdown(
    """
This app uses your trained models from `website_model_package.pkl`  
to estimate risk based on clinical and hoof-related features.
"""
)

# Try loading the model package
try:
    MODELS, SCALER, FEATURE_NAMES, RAW_PKG = load_model_package()
except Exception as e:
    st.error(
        f"❌ Could not load model package.\n\n"
        f"Error: `{type(e).__name__}: {e}`\n\n"
        "Make sure `website_model_package.pkl` is in the same folder as `app.py`."
    )
    st.stop()

# Sidebar: model selection & threshold
st.sidebar.header("Model settings")

model_names = list(MODELS.keys())
selected_models = st.sidebar.multiselect(
    "Select models to run",
    model_names,
    default=model_names,
)

threshold = st.sidebar.slider(
    "Decision threshold (for probability-based models)",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.write("📦 Package keys:", list(RAW_PKG.keys()))
st.sidebar.write("📊 Features used:", FEATURE_NAMES)


# =====================================================
# 5) Input form for all features
# =====================================================

st.subheader("Enter input features")

with st.form("input_form"):
    # Two-column layout for nicer UI
    col1, col2 = st.columns(2)

    # NOTE: ranges / defaults are generic — adjust for your data as needed.
    with col1:
        age = st.number_input("Age (years)", 0.0, 40.0, 10.0, 1.0)
        sex = st.number_input("Sex (encoded)", 0, 3, 1, 1)
        heart_rate = st.number_input("Heart Rate", 0.0, 150.0, 44.0, 1.0)
        resp_rate = st.number_input("Respiratory rate", 0.0, 120.0, 20.0, 1.0)
        rect_temp = st.number_input("Rectal temperature (°C)", 30.0, 42.0, 37.5, 0.1)
        gut_sounds = st.number_input("Gutsounds (encoded)", -1.0, 3.0, 0.0, 1.0)
        digital_pulses = st.number_input("Digital pulses (encoded)", 0.0, 3.0, 0.0, 1.0)
        body_weight = st.number_input("Body weight (kg)", 50.0, 900.0, 450.0, 10.0)
        bcs = st.number_input(
            "Body Condition Score (out of 9)", 1.0, 9.0, 5.0, 0.5
        )

    with col2:
        length_rf = st.number_input("LengthRF", 0.0, 50.0, 10.0, 0.1)
        length_lf = st.number_input("LengthLF", 0.0, 50.0, 10.0, 0.1)
        length_rh = st.number_input("LengthRH", 0.0, 50.0, 10.0, 0.1)
        width_rf = st.number_input("WidthRF", 0.0, 50.0, 10.0, 0.1)
        width_lf = st.number_input("WidthLF", 0.0, 50.0, 10.0, 0.1)
        width_rh = st.number_input("WidthRH", 0.0, 50.0, 10.0, 0.1)
        htrf = st.number_input("HTRF (encoded)", -1.0, 3.0, 0.0, 1.0)
        htrh = st.number_input("HTRH (encoded)", -1.0, 3.0, 0.0, 1.0)
        lerf = st.number_input("LERF (encoded)", 0.0, 4.0, 0.0, 1.0)

    submitted = st.form_submit_button("Predict")


# =====================================================
# 6) Run prediction when form is submitted
# =====================================================

if submitted:
    if not selected_models:
        st.warning("Please select at least one model from the sidebar.")
        st.stop()

    # Map all form values into a dict
    form_values: Dict[str, float] = {
        "Age(years)": age,
        "Sex": float(sex),
        "HeartRate": heart_rate,
        "Respiratoryrate": resp_rate,
        "Rectaltemperature": rect_temp,
        "Gutsounds": gut_sounds,
        "Digitalpulses": digital_pulses,
        "Bodyweight(kg)": body_weight,
        "BodyConditionScoring(outof9)": bcs,
        "LengthRF": length_rf,
        "LengthLF": length_lf,
        "LengthRH": length_rh,
        "WidthRF": width_rf,
        "WidthLF": width_lf,
        "WidthRH": width_rh,
        "HTRF": htrf,
        "HTRH": htrh,
        "LERF": lerf,
    }

    # Build input DataFrame in correct feature order
    X_input = build_input_dataframe(FEATURE_NAMES, form_values)

    # If a separate scaler exists (and is not inside pipelines), try to apply it
    # Otherwise, many users store full Pipelines that already include scaling.
    if SCALER is not None:
        try:
            X_scaled = SCALER.transform(X_input)
            X_for_model = pd.DataFrame(X_scaled, columns=FEATURE_NAMES)
        except Exception:
            # If scaler mismatches, just use original X_input
            X_for_model = X_input
    else:
        X_for_model = X_input

    st.markdown("### 🔍 Prediction results")
    results_rows = []

    for model_name in selected_models:
        model = MODELS[model_name]

        # If model is a Pipeline (with scaler inside), we can pass raw X_input.
        try:
            label, proba = model_predict(model, X_for_model)
        except Exception:
            label, proba = model_predict(model, X_input)

        risk_text = "High risk" if label == 1 else "Low / intermediate risk"

        results_rows.append(
            {
                "Model": model_name,
                "Predicted class": int(label),
                "Risk interpretation": risk_text,
                "Probability (class 1)": round(proba, 3) if proba is not None else None,
            }
        )

    results_df = pd.DataFrame(results_rows)
    st.dataframe(results_df, use_container_width=True)

    # Simple summary based on the first model in the list
    primary = results_rows[0]
    st.markdown(
        f"**Summary (based on `{primary['Model']}`):** "
        f"Predicted class = `{primary['Predicted class']}` → **{primary['Risk interpretation']}**"
    )

    st.markdown("#### Input used")
    st.dataframe(X_input, use_container_width=True)
