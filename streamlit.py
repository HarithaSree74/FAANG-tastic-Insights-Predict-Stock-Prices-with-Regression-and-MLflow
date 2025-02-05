import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="FAANG Stock Prediction",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
    <style>
    /* Remove top padding */
    .main > div {
        padding-top: 1rem;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #1E1E1E;
    }
    /* Make sidebar text white */
    .sidebar [data-testid="stNumberInput"] label {
        color: white !important;
    }
    .sidebar [data-testid="stSelectbox"] label {
        color: white !important;
    }
    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Model paths
model_path = r"C:\Users\Haritha Sree D\.vscode\faang stock analysis\LinearModelnew.pkl"

# Main title with reduced spacing
st.markdown("# 🚀 FAANG STOCK PRICE PREDICTION")
st.markdown("---")
# Information Box
st.markdown("""
    <div class='info-box'>
        <h4>📈 About this Stock Price Predictor</h4>
        <p>Welcome to our FAANG Stock Price Prediction Tool! This application uses machine learning to predict stock closing prices 
        for major tech companies (Facebook, Apple, Amazon, Netflix, and Google) based on key market indicators.</p>
        <p><b>How to use:</b></p>
        <ul>
            <li>Select your target company from the sidebar</li>
            <li>Enter the required market values and temporal information</li>
            <li>Click 'Predict' to get the estimated closing price ($)</li>
        </ul>
        <p><i>Note: This tool uses historical data for predictions and is intended for educational purposes only.</i></p>
    </div>
""", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.markdown("<h3 style='color: black; margin-bottom: 0;'>📊 Input Parameters</h3>", unsafe_allow_html=True)

# Company selection
company = st.sidebar.selectbox(
    "Choose a company",
    options=["Apple", "Google", "Facebook", "Amazon", "Netflix"]
)

# Stock encoding (one-hot encoding for companies)
Ticker_AMZN = 1 if company.upper() == "AMAZON" else 0
Ticker_AAPL = 1 if company.upper() == "APPLE" else 0
Ticker_META = 1 if company.upper() == "FACEBOOK" else 0
Ticker_GOOGL = 1 if company.upper() == "GOOGLE" else 0
Ticker_NFLX = 1 if company.upper() == "NETFLIX" else 0

# Load models
with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(r"C:\Users\Haritha Sree D\.vscode\faang stock analysis\standard_scaler.pkl", "rb") as file:
    scaler_model = pickle.load(file)

# Market data inputs
feature_1 = st.sidebar.number_input("Opening Price ($)", 
    min_value=0.0, 
    max_value=10000.0,
    format="%.8f",
    value=0.0)

feature_2 = st.sidebar.number_input("High Price ($)", 
    min_value=0.0, 
    max_value=10000.0, 
    format="%.8f",
    value=0.0)

feature_3 = st.sidebar.number_input("Low Price ($)", 
    min_value=0.0, 
    max_value=10000.0,
    format="%.8f", 
    value=0.0)

feature_4 = st.sidebar.number_input("Trading Volume", 
    min_value=0, 
    max_value=10000000000, 
    value=0)

# Temporal features
day_of_week = st.sidebar.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
dow_encoded = {
    "Monday": [1, 0, 0, 0, 0],
    "Tuesday": [0, 1, 0, 0, 0],
    "Wednesday": [0, 0, 1, 0, 0],
    "Thursday": [0, 0, 0, 1, 0],
    "Friday": [0, 0, 0, 0, 1]
}[day_of_week]

month = st.sidebar.selectbox("Month", ["January", "February", "March", "April", "May", "June", 
                                     "July", "August", "September", "October", "November", "December"])
month_encoded = {
    "January": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "February": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "March": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "April": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "May": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "June": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "July": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "August": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "September": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "October": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "November": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "December": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}[month]

year_encoded = st.sidebar.number_input("Year", min_value=2005, max_value=2024, value=2024)

# Input validation
if feature_3 > feature_2:
    st.error("⚠️ Low price cannot be higher than High price!")
elif feature_1 < feature_3 or feature_1 > feature_2:
    st.warning("⚠️ Opening price should be between Low and High prices!")
else:
    # Create input array with all features
    input_features = np.array([[
        feature_1, feature_2, feature_3, feature_4,
        Ticker_AAPL, Ticker_AMZN, Ticker_GOOGL, Ticker_META, Ticker_NFLX,
        year_encoded
    ] + month_encoded + dow_encoded])
    
    # Scale the numerical features (first 4 features only)
    numerical_features = input_features[:, :4]
    scaled_numerical = scaler_model.transform(numerical_features)
    
    # Combine scaled numerical features with categorical features
    input_features[:, :4] = scaled_numerical
    
    # Prediction section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 Predict Stock Price"):
            prediction = model.predict(input_features)
            st.markdown(f"""
                <div style='background-color: #4CAF50; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;'>
                    <h2 style='color: white; margin: 0;'>Predicted Close Price</h2>
                    <h1 style='color: white; margin: 10px 0;'>${prediction[0]:.2f}</h1>
                    <p style='color: white; margin: 0;'>for {company}</p>
                </div>
            """, unsafe_allow_html=True)









