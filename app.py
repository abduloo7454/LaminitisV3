from pathlib import Path
from typing import Dict, Any

import joblib
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

# ---- Small CSS tuning for nicer UI ----
st.markdown(
    """
    <style>
    /* Tighter, centered main container */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    h1 {
        text-align: center;
        margin-bottom: 0.3rem;
    }
    h2, h3, h4, h5 {
        font-weight: 600;
    }
    label {
        font-weight: 500 !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        padding-top: 0.4rem;
        padding-bottom: 0.4rem;
    }
    /* Optional: hide default Streamlit menu/footer for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================
# 2) Load website model package (joblib)
# =========================================
@st.cache_resource
def load_website_package(filename: str = PACKAGE_FILENAME) -> Dict[str, Any]:
    """
    Load the website-ready package saved with joblib.

    Expected structure:
    {
        "pipeline": sklearn Pipeline,
        "feature_names": [...],
        "feature_ranges": {feature: {"min": ..., "max": ...}, ...},
        "model_type": str,
        "is_binary": bool,
        "classes": [...],
        "performance": {"accuracy": float, "f1_score": float}
    }
    """
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
    Predict using a sklearn Pipeline.

    If pipeline has predict_proba:
        returns (label, probability_of_class_1)
    Else:
        returns (label, None)
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
# 4) Load model package and extract info
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
# 5) Sidebar – Powered by (top) + model info
# =========================================
st.sidebar.markdown("**Powered by**")

LOGO_WIDTH = 160  # adjust to taste

# Make sure these files exist in your repo root:
#   images.png, images.jpeg, qeeri_logo.png
st.sidebar.image("images.png",     width=LOGO_WIDTH)
st.sidebar.image("images.jpeg",    width=LOGO_WIDTH)
st.sidebar.image("qeeri_logo.png", width=LOGO_WIDTH)

st.sidebar.markdown("---")

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

st.sidebar.markdown("---")


# =========================================
# 6) Main title + tabs (boxed navigation)
# =========================================
st.title("Laminitis Risk Prediction Web App 🩺")

st.markdown(
    """
This tool uses a machine-learning model trained on clinical and hoof-related
features to estimate the probability that a horse is at risk of laminitis.

Use the tabs below to switch between the prediction tool, project overview,
and team information.
"""
)

tab1, tab2, tab3 = st.tabs(
    ["🧮 Risk Prediction", "📄 About the Project", "👥 Meet the Team"]
)


# =========================================
# 7) TAB 1 – Risk Prediction
# =========================================
with tab1:
    st.subheader("Enter input features")

    # Nice labels, ranges, and defaults for each feature
    clinical_config = {
        "Age(years)": dict(
            label="Age (years)", min=0.0, max=40.0, default=10.0, step=1.0
        ),
        "Sex": dict(
            label="Sex (encoded)", min=0.0, max=3.0, default=1.0, step=1.0,
            # help could describe encoding if you want
        ),
        "HeartRate": dict(
            label="Heart rate (beats/min)", min=0.0, max=150.0, default=44.0, step=1.0
        ),
        "Respiratoryrate": dict(
            label="Respiratory rate (breaths/min)", min=0.0, max=120.0, default=20.0, step=1.0
        ),
        "Rectaltemperature": dict(
            label="Rectal temperature (°C)", min=30.0, max=42.0, default=37.5, step=0.1
        ),
        "Gutsounds": dict(
            label="Gutsounds (encoded)", min=-1.0, max=3.0, default=0.0, step=1.0
        ),
        "Digitalpulses": dict(
            label="Digital pulses (encoded)", min=0.0, max=3.0, default=0.0, step=1.0
        ),
        "Bodyweight(kg)": dict(
            label="Body weight (kg)", min=50.0, max=900.0, default=450.0, step=10.0
        ),
        "BodyConditionScoring(outof9)": dict(
            label="Body Condition Score (1–9)", min=1.0, max=9.0, default=5.0, step=0.5
        ),
    }

    hoof_config = {
        "LengthRF": dict(
            label="Length RF", min=0.0, max=50.0, default=10.0, step=0.1
        ),
        "LengthLF": dict(
            label="Length LF", min=0.0, max=50.0, default=10.0, step=0.1
        ),
        "LengthRH": dict(
            label="Length RH", min=0.0, max=50.0, default=10.0, step=0.1
        ),
        "WidthRF": dict(
            label="Width RF", min=0.0, max=50.0, default=10.0, step=0.1
        ),
        "WidthLF": dict(
            label="Width LF", min=0.0, max=50.0, default=10.0, step=0.1
        ),
        "WidthRH": dict(
            label="Width RH", min=0.0, max=50.0, default=10.0, step=0.1
        ),
        "HTRF": dict(
            label="HTRF (encoded)", min=-1.0, max=3.0, default=0.0, step=1.0
        ),
        "HTRH": dict(
            label="HTRH (encoded)", min=-1.0, max=3.0, default=0.0, step=1.0
        ),
        "LERF": dict(
            label="LERF (encoded)", min=0.0, max=4.0, default=0.0, step=1.0
        ),
    }

    # Combined config for easy mapping later
    feature_config: Dict[str, Dict[str, float]] = {}
    feature_config.update(clinical_config)
    feature_config.update(hoof_config)

    with st.form("input_form"):
        values: Dict[str, float] = {}

        # ---------- Clinical block ----------
        st.markdown("#### Clinical information")
        c1, c2, c3 = st.columns(3)

        # Assign clinical fields to 3 columns
        clinical_order = [
            "Age(years)",
            "Sex",
            "HeartRate",
            "Respiratoryrate",
            "Rectaltemperature",
            "Gutsounds",
            "Digitalpulses",
            "Bodyweight(kg)",
            "BodyConditionScoring(outof9)",
        ]
        clinical_cols = [c1, c2, c3]

        for i, feat in enumerate(clinical_order):
            cfg = clinical_config[feat]
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

            col = clinical_cols[i % 3]
            val = col.number_input(
                cfg.get("label", feat),
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                step=float(step_val),
            )
            values[feat] = float(val)

        # ---------- Hoof block ----------
        st.markdown("#### Hoof measurements")
        h1, h2, h3 = st.columns(3)
        hoof_order = [
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
        hoof_cols = [h1, h2, h3]

        for i, feat in enumerate(hoof_order):
            cfg = hoof_config[feat]
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

            col = hoof_cols[i % 3]
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
        # Build row in correct order of feature_names from the model
        feature_value_map = values  # keys match feature names
        X = pd.DataFrame(
            [[feature_value_map[f] for f in feature_names]],
            columns=feature_names,
        )

        label, proba = predict_with_pipeline(pipeline, X, threshold=threshold)

        # ---- Styled result card ----
        if label == 1:
            bg = "#ffe5e5"      # light red
            text_color = "#7a0000"  # dark red text
            title = "High risk"
        else:
            bg = "#e6ffe6"      # light green
            text_color = "#005000"  # dark green text
            title = "Low / intermediate risk"
        
        if proba is not None:
            detail = f"Probability (class 1): <b>{proba:.3f}</b> (threshold = {threshold:.2f})"
        else:
            detail = "Model does not expose predicted probabilities."
        
        st.markdown(
            f"""
            <div style="
                padding:1rem;
                border-radius:0.75rem;
                background-color:{bg};
                margin-top:1rem;
                color:{text_color};
            ">
                <h3 style="margin:0; color:{text_color};">{title}</h3>
                <p style="margin:0.4rem 0 0; color:{text_color};">
                    Predicted class: <b>{label}</b><br>
                    {detail}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


        st.markdown("#### Input used")
        st.dataframe(X, use_container_width=True)


# =========================================
# 8) TAB 2 – About the Project
# =========================================
with tab2:
    st.header("About the Laminitis Risk Project")

    st.markdown(
        """
This project aims to develop a practical, data-driven tool to support
early identification of horses at risk of laminitis.

The underlying model was trained on clinical examination findings and
hoof-measurement data collected from horses examined at the Equine
Veterinary Medical Center and collaborating institutions.

**Key goals**

- Translate machine-learning research into a simple web-based risk tool  
- Help clinicians combine multiple risk factors into a single probability estimate  
- Provide a framework that can be refined as new data are added  

This online tool is intended to complement, not replace, the judgement
of experienced veterinarians. Clinical decisions should always be made
by qualified clinicians considering the full clinical context.
"""
    )


# =========================================
# 9) TAB 3 – Meet the Team
# =========================================
with tab3:
    st.header("Meet the Team")

    st.markdown('Prof. Othmane Bouhali')
    st.markdown('Dr. Halima Bensmail')
    st.markdown('Dr. Jessica P Johnson')
    st.markdown('Abdullah')

    col_team1, col_team2 = st.columns(2)

    with col_team1:
        st.subheader("Clinical & Veterinary Leads")
        st.markdown(
            """
- Equine clinicians providing case assessment, data collection,
  and interpretation of laminitis risk  
- Specialists overseeing case selection, inclusion criteria,
  and validation of risk categories
"""
        )

    with col_team2:
        st.subheader("Data Science & Modelling")
        st.markdown(
            """
- Researchers responsible for data preprocessing, feature engineering,
  model development, and performance evaluation  
- Team members implementing the web interface and deployment workflow
"""
        )

    st.subheader("Institutional Support")
    st.markdown(
        """
# - Qatar Biomedical Research Institute / Hamad Bin Khalifa University  
# - Equine Veterinary Medical Center  
# - Partner institutes contributing expertise, infrastructure, and computing support
"""
    )


# =========================================
# 10) Footer disclaimer
# =========================================
st.markdown(
    """
    <hr>
    <p style="font-size:0.8rem; color:gray; text-align:center;">
    This tool is for research and educational purposes only and does not replace
    clinical judgement. Final decisions should always be made by qualified
    veterinarians based on full examination and context.
    </p>
    """,
    unsafe_allow_html=True,
)
