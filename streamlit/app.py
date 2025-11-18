# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import os
import datetime
import xgboost as xgb
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import threading
import time
from PIL import Image
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from preprocessor_helper import fillna_missing


# setting the current working directory
file_path = "/Users/junan/School/DSA4263/Project part 1/DSA4263_Vehicle-Insurance-Fraud-Detection/"

feature_order_for_model = [
        "months_as_customer", "age", "policy_csl", "policy_deductable", "policy_annual_premium", 
        "umbrella_limit", "insured_zip", "insured_sex", "insured_education_level", "capital-gains", 
        "capital-loss", "incident_hour_of_the_day", "number_of_vehicles_involved", "bodily_injuries", 
        "witnesses", "total_claim_amount", "injury_claim", "property_claim", "policy_state_IL", "policy_state_IN", 
        "policy_state_OH", "insured_occupation_Manual", "insured_occupation_Professional", "insured_occupation_Sales/Service",
        "insured_occupation_Technical", "insured_hobbies_Extreme", "insured_hobbies_Games", "insured_hobbies_Leisure", 
        "insured_hobbies_Outdoor", "insured_hobbies_Sports", "insured_relationship_husband", "insured_relationship_not-in-family", 
        "insured_relationship_other-relative", "insured_relationship_own-child", "insured_relationship_unmarried", "insured_relationship_wife", 
        "incident_type_Parked Car", "incident_type_Single Vehicle Collision", "incident_type_Vehicle Theft", "collision_type_Front Collision", 
        "collision_type_Missing", "collision_type_Rear Collision", "collision_type_Side Collision", "incident_severity_Major Damage", 
        "incident_severity_Minor Damage", "incident_severity_Total Loss", "incident_severity_Trivial Damage", "authorities_contacted_Ambulance", 
        "authorities_contacted_Fire", "authorities_contacted_None", "authorities_contacted_Other", "authorities_contacted_Police", 
        "incident_state_NC", "incident_state_NY", "incident_state_OH", "incident_state_PA", "incident_state_SC", "incident_state_VA",
        "incident_state_WV", "incident_city_Arlington", "incident_city_Columbus", "incident_city_Hillsdale", "incident_city_Northbend", 
        "incident_city_Northbrook", "incident_city_Riverwood", "incident_city_Springfield", "property_damage_Missing", "property_damage_NO", 
        "property_damage_YES", "police_report_available_Missing", "police_report_available_NO", "police_report_available_YES"
    ]

def run_explainability(processed_data_array, model, preprocessor):
    """
    Calculates the contribution of each feature (weight * value) to the model's output (log-odds).
    Returns the top 5 most influential features.
    """
    
    if not hasattr(model, 'coef_') or not hasattr(model, 'intercept_'):
        st.error("Model does not appear to be a linear model (missing 'coef_'). Explainability cannot be run.")
        return None, None
        
    # model coefficients
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    
    # input feature values 
    feature_values = processed_data_array[0]


    # calculate the linear contribution for each feature: weight * feature_value
    contributions = feature_values * coefficients
    
    contribution_df = pd.DataFrame({
        'Feature_Name': feature_order_for_model,
        'Contribution_Score': contributions
    })
    
    # Sort by absolute contribution score to find the most influential features
    contribution_df['Abs_Contribution'] = contribution_df['Contribution_Score'].abs()
    top_contributions = contribution_df.sort_values(by='Abs_Contribution', ascending=False).head(5)
    
    return top_contributions, intercept

def create_contribution_plot(contributions_df):
    """
    Creates a Plotly horizontal bar chart showing the top feature contributions.
    """
    
    # Sort by the score (for visualization)
    df_plot = contributions_df.sort_values(by='Contribution_Score', ascending=True)

    # Determine color: Red for positive (more fraud), Green for negative (less fraud)
    colors = ['#E74C3C' if score > 0 else '#2ECC71' for score in df_plot['Contribution_Score']]
    
    fig = go.Figure(go.Bar(
        y=df_plot['Feature_Name'],
        x=df_plot['Contribution_Score'],
        orientation='h',
        marker=dict(color=colors),
        hovertemplate='Contribution: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Top 5 Feature Contributions to Fraud Likelihood',
        xaxis_title='Contribution Score (Positive = More Fraud Likely)',
        yaxis_title='Feature',
        height=350,
        margin=dict(l=150, r=20, t=50, b=50), 
        plot_bgcolor='black',
        paper_bgcolor='black',
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='lightgray')
    )
    return fig


# --- Brief Introduction ---
def page_introduction():
    """Sets up the introduction section of the app."""
    st.title("🛡️ Insurance Fraud Detection Simulator")
    st.subheader("Leveraging Logistic Regression for Proactive Claims Review")

    st.markdown("""
        Our goal is to **identify potential fraud cases** quickly and accurately so that our real-life agents can focus their time on cases that truly require human investigation.

        ---

        ### The System Flow (Model Working Behind the Scenes)
        In a live environment, the process is automatic:
        1.  **Customer Files Claim:** New insurance and claim data are recorded instantly.
        2.  **Model Prediction:** Our **Logistic Regression Model** runs on the backend, calculating a fraud probability score.
        3.  **Agent Review:** High-risk claims are immediately flagged for human agents to check.

        This website simulates Step 2 and 3, allowing you to **input data** and see the model's fraud prediction in real-time.
    """)

    

# --- Model Showcase & Data Input ---
def page_model_showcase():
    """Sets up the input form for model simulation."""
    st.title("🔍 Model Simulation: Enter Claim Details")
    st.subheader("Input your policy and incident information below:")

    prediction_placeholder = st.empty()

    with st.form("claim_input_form"):
        # --- Policyholder and Policy Details ---
        st.header("1. Policy & Customer Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.number_input("Months as Customer", min_value=0, max_value=600, value=100, step=1, key="months_as_customer", help="Number of months the individual has been a customer.")
            st.number_input("Age", min_value=18, max_value=100, value=40, step=1, key="age", help="Age of the insured individual.")
            st.selectbox("Policy State", options=['OH', 'IN', 'IL'], key="policy_state")
            st.selectbox("Insured Sex", options=['Male', 'Female'], key="insured_sex")
            #st.number_input("Policy Number", min_value=0, max_value=9999999, value=1234567, step=1, key="policy_number", help="Unique identifier for the insurance policy.")


        with col2:
            st.selectbox("Education Level", options=['JD', 'High School', 'Associate', 'MD', 'Masters', 'PhD', 'College'], key="insured_education_level")
            st.selectbox("Relationship", options=['Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative', 'Not-in-family'], key="insured_relationship")
            st.date_input("Policy Bind Date", value=datetime.date(2015, 1, 1), key="policy_bind_date")
            st.selectbox("Policy CSL", options=['100/300', '250/500', '500/1000'], key="policy_csl")


        with col3:
            st.number_input("Deductible", min_value=500, max_value=2000, value=1000, step=100, key="policy_deductable")
            st.number_input("Annual Premium", value=1200.00, format="%.2f", key="policy_annual_premium")
            st.number_input("Umbrella Limit", min_value=0, max_value=1000000, value=0, step=10000, key="umbrella_limit")

        st.markdown("---")

        # --- Incident and Claim Details ---
        st.header("2. Incident & Claim Information")
        col4, col5, col6 = st.columns(3)

        with col4:
            st.selectbox("Incident Type", options=['Single Vehicle Collision', 'Multi-vehicle Collision', 'Vehicle Theft', 'Parked Car'], key="incident_type")
            st.selectbox("Collision Type", options=['Rear Collision', 'Side Collision', 'Front Collision', 'Missing'], key="collision_type")
            st.selectbox("Severity", options=['Minor Damage', 'Major Damage', 'Total Loss', 'Trivial Damage'], key="incident_severity")
            st.text_input("Incident state", value="OH", key="incident_state")
            #st.text_input("Incident location", value="1234 Main St", key="incident_location")

        with col5:
            st.text_input("Incident city", value="Columbus", key="incident_city")
            st.selectbox("Authorities Contacted", options=['Police', 'Fire', 'Ambulance', 'None', 'Other'], key="authorities_contacted")
            st.date_input("Incident Date", value=datetime.date(2015, 1, 1), key="incident_date")
            st.number_input("Hour of Incident (0-23)", min_value=0, max_value=23, value=14, step=1, key="incident_hour_of_the_day")
            
            
        with col6:
            st.number_input("Vehicles Involved", min_value=0, value = 0, step=1, key="number_of_vehicles_involved")
            st.number_input("Total Claim Amount", min_value=0, value=50000, step=1000, key="total_claim_amount")
            st.number_input("Injury Claim", min_value=0, value=5000, step=500, key="injury_claim")
            st.number_input("Property Claim", min_value=0, value=5000, step=500, key="property_claim")
            #st.number_input("Vehicle Claim", min_value=0, value=40000, step=1000, key="vehicle_claim")

        # supplementary de
        st.markdown("---")
        st.subheader("3. Supplementary Details")
        col7, col8, col9 = st.columns(3)

        with col7:
            st.selectbox("Insured Occupation", options=['tech-support', 'craft-repair', 'sales', 'adm-clerical', 'protective-serv', 'transport-moving', 'machine-op-inspct','handlers-cleaners','farming-fishing', 'other-service', 'priv-house-serv', 'exec-managerial', 'prof-specialty', 'armed-forces'], key="insured_occupation")
            st.number_input("Insured Zip", min_value=0, max_value=999999, value=0, step=1, key="insured_zip")
            st.number_input("Witnesses", min_value=0, max_value=10, value=0, key="witnesses")
            st.selectbox("Property Damage?", options = ['Yes', 'No', 'Missing'], key="property_damage")

        with col8:
            st.number_input("Capital Gains", value=0, step=100, key="capital-gains")
            st.number_input("Capital Loss", value=0, step=100, key="capital-loss")
            st.number_input("Bodily Injuries", value = 0, min_value = 0, max_value = 2, step = 1, key="bodily_injuries")
            st.selectbox("Insured Hobbies", options=['sleeping', 'reading', 'movies', 'board-games', 'chess', 'video-games', 'bungie-jumping', 'base-jumping', 'skydiving', 'golf', 'basketball', 'polo', 'cross-fit', 'exercise', 'camping', 'hiking', 'yachting', 'paintball', 'kayaking', 'dancing'], key="insured_hobbies")

        with col9:
            st.text_input("Auto Make", value = "Honda", key="auto_make")
            st.text_input("Auto Model", value = "Vezel", key="auto_model")
            st.number_input("Auto Year", min_value=1990, max_value=2025, value=2015, step=1, key="auto_year")
            st.selectbox("Police Report Available?", options = ['Yes', 'No', 'Missing'],key="police_report_available")


        # submit button
        st.markdown("---")
        submitted = st.form_submit_button("Submit Claim for Fraud Prediction")

    # --- Prediction Logic ---
    if submitted:
        # check if model/preprocessor loaded successfully
        if model is not None and preprocessor is not None:
            try:
                fraud_probability, processed_data_array = run_model_prediction(st.session_state)
            except Exception as e:
                st.error(f"Prediction failed due to internal error: {e}")
                fraud_probability = 0.5 
                processed_data_array = None
            
            # --- Render Prediction Result ---
            st.subheader("⚡ Model Prediction Result")
            
            OPTIMAL_THRESHOLD = 0.5 # can be adjusted based on business needs

            
            if fraud_probability >= OPTIMAL_THRESHOLD:
                st.error(f"**FRAUD LIKELY!** 🚨 The model predicts a {fraud_probability*100:.2f}% chance of fraud.")
                st.info("Recommendation: Flag this claim for immediate human agent review.")
            else:
                st.success(f"**LOW RISK.** ✅ The model predicts a {fraud_probability*100:.2f}% chance of fraud.")
                st.info("Recommendation: Proceed with standard automated claim processing.")

            # --- Explainability Section ---
            if processed_data_array is not None:
                st.markdown("---")
                with st.expander("🔬 View Explainability: Why was this prediction made?", expanded=True):
                    
                    st.markdown("The Logistic Regression model determines risk by calculating a **Log-Odds Score**. Features that contribute positively (red bars) increase the score and push the prediction toward **Fraud**; features that contribute negatively (green bars) decrease the score and push it toward **Non-Fraud**.")

                    try:
                        top_contributions, intercept = run_explainability(processed_data_array, model, preprocessor)
                        
                        fig_explainer = create_contribution_plot(top_contributions)
                        st.plotly_chart(fig_explainer, use_container_width=True)

                        
                        # display the top features in a small table
                        st.markdown("#### Top Feature Summary")
                        st.dataframe(
                            top_contributions[['Feature_Name', 'Contribution_Score']].rename(
                                columns={'Feature_Name': 'Feature', 'Contribution_Score': 'Impact (Log-Odds)'}
                            ).style.format({"Impact (Log-Odds)": "{:.4f}"}),
                            use_container_width=True
                        )


                    except Exception as e:
                        st.warning(f"Could not generate feature contributions: {e}")

        else:
            st.error("Cannot run prediction: Model assets failed to load.")

MODEL_PATH = file_path + 'models/model.pkl'  # Or whatever you saved the model as
PREPROCESSOR_PATH = file_path + 'streamlit/preprocessor.pkl' # Or whatever you saved the transformer as


# --- 1. Load Model and Preprocessor (Paths updated for robustness) ---
@st.cache_resource
def load_assets():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', 'models', 'model.pkl')
    preprocessor_path = os.path.join(script_dir, 'preprocessor.pkl')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found. Looked at: {model_path}") 
        st.error(f"Error: Model file not found. Please verify 'model.pkl' exists in the 'models' directory.")
        return None, None
    if not os.path.exists(preprocessor_path):
        print(f"Error: Preprocessor file not found. Looked at: {preprocessor_path}")
        st.error(f"Error: Preprocessor file not found. Please verify 'preprocessor.pkl' is in the 'streamlit' directory.")
        return None, None

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model or preprocessor: {e}")
        return None, None
        
    return model, preprocessor

model, preprocessor = load_assets()

# --- 2. Model Prediction Function ---

def run_model_prediction(input_data):
    """
    Transforms the user input and runs the logistic regression model.
    """
    if model is None or preprocessor is None:
        return 0.5 # Return a neutral probability if loading failed

    # feature order must match training order for model
    feature_order = [
        'months_as_customer', 'age', 'policy_csl', 'policy_deductable', 'policy_annual_premium', 
        'umbrella_limit', 'insured_zip', 'capital-gains', 'capital-loss', 
        'incident_hour_of_the_day', 'number_of_vehicles_involved', 'witnesses',
        'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim', 
        'auto_year', 
        'policy_state',  'insured_sex',
        'insured_education_level', 'insured_occupation', 'insured_hobbies', 
        'insured_relationship', 'incident_type', 'collision_type', 
        'incident_severity', 'authorities_contacted', 'incident_state', 
        'incident_city', 'incident_location', 'property_damage', 
        'bodily_injuries', 'police_report_available'
    ]

    input_df = pd.DataFrame([input_data], columns=feature_order)


    columns_to_convert =  [
    'policy_state_IL', 'policy_state_IN', 'policy_state_OH',
    'insured_occupation_Manual', 'insured_occupation_Professional', 'insured_occupation_Sales/Service', 'insured_occupation_Technical',
    'insured_hobbies_Extreme', 'insured_hobbies_Games', 'insured_hobbies_Leisure', 'insured_hobbies_Outdoor', 'insured_hobbies_Sports',
    'insured_relationship_husband', 'insured_relationship_not-in-family', 'insured_relationship_other-relative', 
    'insured_relationship_own-child', 'insured_relationship_unmarried', 'insured_relationship_wife',
    'incident_type_Parked Car', 'incident_type_Single Vehicle Collision', 'incident_type_Vehicle Theft',
    'collision_type_Front Collision', 'collision_type_Missing', 'collision_type_Rear Collision', 'collision_type_Side Collision',
    'incident_severity_Major Damage', 'incident_severity_Minor Damage', 'incident_severity_Total Loss', 'incident_severity_Trivial Damage',
    'authorities_contacted_Ambulance', 'authorities_contacted_Fire', 'authorities_contacted_None', 
    'authorities_contacted_Other', 'authorities_contacted_Police',
    'incident_state_NC', 'incident_state_NY', 'incident_state_OH', 'incident_state_PA', 'incident_state_SC', 
    'incident_state_VA', 'incident_state_WV',
    'incident_city_Arlington', 'incident_city_Columbus', 'incident_city_Hillsdale', 'incident_city_Northbend', 
    'incident_city_Northbrook', 'incident_city_Riverwood', 'incident_city_Springfield',
    'property_damage_Missing', 'property_damage_NO', 'property_damage_YES',
    'police_report_available_Missing', 'police_report_available_NO', 'police_report_available_YES'
    ]

    # --- Preprocessing ---
    processed_data = preprocessor.transform(input_df)
    processed_data = pd.DataFrame(processed_data, columns=preprocessor.get_feature_names_out())
    processed_data = processed_data.rename(columns={'authorities_contacted_Missing':'authorities_contacted_None'})
    processed_data = processed_data.drop(columns=['incident_type_Multi-vehicle Collision'])
    
    # --- Prediction ---
    # predict_proba returns the probabilities for both classes: [P(No Fraud), P(Fraud)]
    processed_data = pd.DataFrame(processed_data, columns=feature_order_for_model)
    processed_data[columns_to_convert] = processed_data[columns_to_convert].astype(bool)
    fraud_probability = model.predict_proba(processed_data)[:, 1][0] # select probability of the 'Fraud' class
    
    return fraud_probability, processed_data.to_numpy()
# --- Main App Execution ---
if __name__ == "__main__":
    page_introduction()
    st.markdown("---")
    page_model_showcase()
