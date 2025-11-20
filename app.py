import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import os
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Equine Laminitis Risk Assessment",
    page_icon="🐎",
    layout="wide"
)

# Load model function
@st.cache_resource
def load_model():
    try:
        with open('website_model_package.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

def main():
    st.sidebar.title("🐎 Laminitis Risk Assessor")
    page = st.sidebar.radio("Navigate", ["Home", "Risk Assessment", "About"])
    
    if page == "Home":
        show_home()
    elif page == "Risk Assessment":
        show_assessment()
    else:
        show_about()

def show_home():
    st.title("Equine Laminitis Risk Assessment")
    st.markdown("---")
    
    st.markdown("""
    ## 🐎 Early Detection Saves Hooves
    
    This tool helps assess your horse's risk of developing laminitis using machine learning.
    
    ### Key Risk Factors:
    - **Obesity** and overfeeding
    - **Endocrine disorders** (PPID, EMS)
    - **Grain overload**
    - **Systemic inflammation**
    
    ### How to Use:
    1. Navigate to **Risk Assessment**
    2. Enter your horse's health parameters
    3. Get instant risk evaluation
    """)

def show_assessment():
    st.title("🎯 Laminitis Risk Assessment")
    
    # Load model
    model_data = load_model()
    if not model_data:
        st.error("Model not loaded. Please check deployment.")
        return
    
    st.markdown("Enter your horse's health parameters:")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", 0, 50, 10)
        sex = st.selectbox("Sex", ["Female", "Male", "Gelding", "Stallion"])
        heart_rate = st.number_input("Heart Rate (bpm)", 20, 100, 36)
        respiratory_rate = st.number_input("Respiratory Rate", 8, 40, 12)
        rectal_temp = st.number_input("Rectal Temp (°C)", 36.0, 40.0, 37.5)
    
    with col2:
        gut_sounds = st.number_input("Gut Sounds", 0.0, 20.0, 8.0)
        digital_pulses = st.number_input("Digital Pulses", 0.0, 10.0, 2.0)
        body_weight = st.number_input("Body Weight (kg)", 100.0, 1000.0, 500.0)
        body_condition = st.slider("Body Condition (1-9)", 1.0, 9.0, 5.0)
    
    # Additional measurements (simplified)
    st.subheader("Limb Measurements (cm)")
    col3, col4 = st.columns(2)
    with col3:
        length_rf = st.number_input("Right Fore Length", 30.0, 60.0, 45.0)
        length_lf = st.number_input("Left Fore Length", 30.0, 60.0, 45.0)
    with col4:
        width_rf = st.number_input("Right Fore Width", 10.0, 30.0, 15.0)
        width_lf = st.number_input("Left Fore Width", 10.0, 30.0, 15.0)
    
    if st.button("🔍 Assess Risk"):
        try:
            # Prepare input
            input_data = {
                'Age(years)': age,
                'Sex': 1 if sex in ["Male", "Stallion"] else 0,
                'HeartRate': heart_rate,
                'Respiratoryrate': respiratory_rate,
                'Rectaltemperature': rectal_temp,
                'Gutsounds': gut_sounds,
                'Digitalpulses': digital_pulses,
                'Bodyweight(kg)': body_weight,
                'BodyConditionScoring(outof9)': body_condition,
                'LengthRF': length_rf, 'LengthLF': length_lf, 'LengthRH': 45.0,
                'WidthRF': width_rf, 'WidthLF': width_lf, 'WidthRH': 15.0,
                'HTRF': 10.0, 'HTRH': 10.0, 'LERF': 10.0
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Predict
            prediction = model_data['pipeline'].predict(input_df)[0]
            probabilities = model_data['pipeline'].predict_proba(input_df)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Results")
            
            risk_prob = probabilities[1]
            
            if risk_prob < 0.3:
                st.success(f"✅ **LOW RISK** ({risk_prob:.1%})")
                st.info("Continue regular monitoring and preventive care.")
            elif risk_prob < 0.7:
                st.warning(f"⚠️ **MODERATE RISK** ({risk_prob:.1%})")
                st.info("Consider veterinary consultation and dietary review.")
            else:
                st.error(f"🚨 **HIGH RISK** ({risk_prob:.1%})")
                st.info("Immediate veterinary attention recommended.")
                
        except Exception as e:
            st.error(f"Prediction error: {e}")

def show_about():
    st.title("About This Tool")
    st.markdown("""
    ## Equine Laminitis Risk Assessment
    
    **Powered by Machine Learning**
    
    This application uses a Support Vector Machine (SVM) model trained on equine health data 
    to predict laminitis risk with high accuracy.
    
    ### Research Team:
    - Veterinary Researchers
    - Data Scientists
    - Equine Specialists
    
    *For educational and research purposes.*
    """)

if __name__ == "__main__":
    main()
