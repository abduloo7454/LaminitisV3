import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="What is My Risk? - Prediabetes Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load('website_model_package.pkl')
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Risk level calculation based on probabilities
def calculate_risk_level(probability):
    if probability < 0.2:
        return "Very Low Risk", "#2E8B57"  # Sea Green
    elif probability < 0.4:
        return "Low Risk", "#32CD32"  # Lime Green
    elif probability < 0.6:
        return "Moderate Risk", "#FFD700"  # Gold
    elif probability < 0.8:
        return "High Risk", "#FF8C00"  # Dark Orange
    else:
        return "Very High Risk", "#FF4500"  # Orange Red

def main():
    # Load model
    model_data = load_model()
    
    # Sidebar navigation
    st.sidebar.title("🩺 What is My Risk?")
    page = st.sidebar.radio("Navigation", ["Home", "Risk Assessment", "About Test", "Our Team"])
    
    if page == "Home":
        show_home_page(model_data)
    elif page == "Risk Assessment":
        show_risk_assessment(model_data)
    elif page == "About Test":
        show_about_test()
    elif page == "Our Team":
        show_team()

def show_home_page(model_data):
    # Header section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("What is My Risk?")
        st.markdown("---")
    with col2:
        st.markdown("### Powered by")
        st.markdown("**QCRI | QBRI**")
    
    # Project info
    st.markdown("""
    ### Project · Team
    
    ---
    
    ### 🔍 Could You Have Prediabetes?
    **Complete your risk assessment today!**
    
    ---
    """)
    
    # Age group information
    st.subheader("👥 What is your age group?")
    
    age_groups = [
        "Younger Than 25",
        "26-35 Years", 
        "36-45 Years",
        "46-55 Years",
        "56-66 Years",
        "66+ Years"
    ]
    
    for group in age_groups:
        st.markdown(f"- **{group}**")
    
    st.info("💡 **You are at a higher risk for prediabetes the older you are.**")
    
    # Call to action
    st.markdown("---")
    st.success("🚀 **Ready to check your risk? Navigate to 'Risk Assessment' in the sidebar!**")

def show_risk_assessment(model_data):
    if model_data is None:
        st.error("❌ Model not available. Please check if the model file exists.")
        return
    
    st.title("🎯 Prediabetes Risk Assessment")
    st.markdown("---")
    
    st.markdown("""
    ### Welcome to the Prediabetes Risk Test in Qatar
    
    **Early warning will save lives**
    
    Type 2 diabetes is one of the fastest growing chronic conditions in Qatar.
    
    The sooner you know you have prediabetes, the sooner you can take action to reverse it and prevent type 2 diabetes.
    """)
    
    st.markdown("---")
    
    # Feature inputs organized by category
    st.subheader("📋 Please provide your information:")
    
    # Personal Information
    st.markdown("#### 👤 Personal Information")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=30, step=1)
        sex = st.selectbox("Sex", ["Male", "Female"])
        heart_rate = st.number_input("Heart Rate", min_value=0, max_value=200, value=72, step=1)
        respiratory_rate = st.number_input("Respiratory Rate", min_value=0, max_value=100, value=16, step=1)
    
    with col2:
        rectal_temp = st.number_input("Rectal Temperature (°C)", min_value=30.0, max_value=45.0, value=37.5, step=0.1)
        gut_sounds = st.number_input("Gut Sounds", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        digital_pulses = st.number_input("Digital Pulses", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        body_weight = st.number_input("Body Weight (kg)", min_value=0.0, max_value=300.0, value=70.0, step=0.1)
    
    # Body Condition Scoring
    st.markdown("#### 📊 Body Measurements")
    col1, col2 = st.columns(2)
    
    with col1:
        body_condition = st.number_input("Body Condition Scoring (out of 9)", min_value=1.0, max_value=9.0, value=5.0, step=0.1)
        length_rf = st.number_input("Length RF", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
        length_lf = st.number_input("Length LF", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
        length_rh = st.number_input("Length RH", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    
    with col2:
        width_rf = st.number_input("Width RF", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
        width_lf = st.number_input("Width LF", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
        width_rh = st.number_input("Width RH", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
        htrf = st.number_input("HTRF", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
        htrh = st.number_input("HTRH", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
        lerf = st.number_input("LERF", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
    
    # Predict button
    if st.button("🎯 Assess My Risk", type="primary", use_container_width=True):
        try:
            # Prepare input data
            input_data = {
                'Age(years)': age,
                'Sex': 1 if sex == "Male" else 0,
                'HeartRate': heart_rate,
                'Respiratoryrate': respiratory_rate,
                'Rectaltemperature': rectal_temp,
                'Gutsounds': gut_sounds,
                'Digitalpulses': digital_pulses,
                'Bodyweight(kg)': body_weight,
                'BodyConditionScoring(outof9)': body_condition,
                'LengthRF': length_rf,
                'LengthLF': length_lf,
                'LengthRH': length_rh,
                'WidthRF': width_rf,
                'WidthLF': width_lf,
                'WidthRH': width_rh,
                'HTRF': htrf,
                'HTRH': htrh,
                'LERF': lerf
            }
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            with st.spinner('🔬 Analyzing your risk factors...'):
                prediction = model_data['pipeline'].predict(input_df)[0]
                probabilities = model_data['pipeline'].predict_proba(input_df)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Your Risk Assessment Results")
            
            # Risk level and confidence
            risk_probability = probabilities[1]  # Probability of class 1 (high risk)
            risk_level, color = calculate_risk_level(risk_probability)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk level indicator
                st.markdown(f"### Risk Level: :{color}[{risk_level}]")
                
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risk_probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risk Confidence"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 20], 'color': "#2E8B57"},
                            {'range': [20, 40], 'color': "#32CD32"},
                            {'range': [40, 60], 'color': "#FFD700"},
                            {'range': [60, 80], 'color': "#FF8C00"},
                            {'range': [80, 100], 'color': "#FF4500"}
                        ],
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Probability distribution
                prob_fig = go.Figure(data=[
                    go.Bar(
                        x=['Low Risk', 'High Risk'],
                        y=probabilities,
                        marker_color=['green', 'red'],
                        text=[f'{probabilities[0]:.1%}', f'{probabilities[1]:.1%}'],
                        textposition='auto'
                    )
                ])
                prob_fig.update_layout(
                    title="Risk Probability Distribution",
                    xaxis_title="Risk Category",
                    yaxis_title="Probability",
                    height=300
                )
                st.plotly_chart(prob_fig, use_container_width=True)
                
                # Detailed probabilities
                st.markdown("#### 📈 Detailed Analysis")
                st.write(f"- **Probability of Low Risk**: {probabilities[0]:.1%}")
                st.write(f"- **Probability of High Risk**: {probabilities[1]:.1%}")
                
                if risk_probability > 0.6:
                    st.warning("""
                    💡 **Recommendation**: Consider consulting with a healthcare professional 
                    for further evaluation and preventive measures.
                    """)
                else:
                    st.success("""
                    💡 **Recommendation**: Maintain healthy lifestyle habits and 
                    consider regular check-ups for continued monitoring.
                    """)
        
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

def show_about_test():
    st.title("📋 About the Screening Test")
    st.markdown("---")
    
    st.markdown("""
    ## How Your Screening Test Will Be Scored?
    
    We divided the risk of having prediabetes into five categories:
    
    - 🟢 **Very Low Risk** (5-6 points)
    - 🟡 **Low Risk** (7-17 points) 
    - 🟠 **Moderate Risk** (18-36 points)
    - 🟣 **High Risk** (37-46 points)
    - 🔴 **Very High Risk** (47-58 points)
    
    ---
    
    ### 🎯 What is My Risk?
    
    - **Project** · **Team**
    
    ---
    
    ### Welcome to the Prediabetes Risk Test in Qatar
    
    **Early warning will save lives**
    
    Type 2 diabetes is one of the fastest growing chronic conditions in Qatar.
    
    The sooner you know you have prediabetes, the sooner you can take action to reverse it and prevent type 2 diabetes.
    
    ---
    
    ### ❓ Why Should I Care About Prediabetes?
    
    If you have prediabetes, now is your time to take action. The sooner you know you have prediabetes, 
    the sooner you can take action to reverse it and prevent type 2 diabetes.
    """)

def show_team():
    st.title("👥 Our Team")
    st.markdown("---")
    
    st.markdown("### Prediabetes Team")
    
    # Team table
    team_data = {
        "Name": [
            "Halima Benamal",
            "Rajhwenda Mall", 
            "Mostafa Hamza",
            "Baan Ullah",
            "Abdelilah Areeksuani",
            "Abdelkader Lattab"
        ],
        "Title": [
            "Principal Scientist",
            "Senior Scientist", 
            "Postdoc",
            "Software Engineer",
            "Senior Scientist",
            "Senior Software Engineer"
        ],
        "Institute": [
            "QCRI",
            "QCRI",
            "QCRI", 
            "QCRI",
            "QBRI",
            "QCRI"
        ]
    }
    
    team_df = pd.DataFrame(team_data)
    st.table(team_df)
    
    st.markdown("---")
    st.markdown("### 📞 Contact Us")
    st.markdown("For more information, please contact our team.")

if __name__ == "__main__":
    main()
