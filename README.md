# FAANG-tastic-Insights-Predict-Stock-Prices-with-Regression-and-MLflow
## Project Description
This project uses Linear Regression to predict stock market closing price, with MLflow tracking experiments and a Streamlit web interface for easy access by financial analysts and investors.
## Technologies Used in this Project
-  **Python** for core development and data analysis
-  **Pandas** for data manipulation and processing
-  **Scikit-learn** for machine learning implementation
-  **MLflow** for experiment tracking and model versioning
-  **Streamlit** for web application interface
## Data Preprocessing
- **Handling Missing Values:**  Checked and handled for any missing data.
- **Remove Outliers:** Used **IQR (Interquartile Range)** to remove or clip the data points outside the acceptable range.
- **Encoding Categorical Variables:** Convert categorical variables into numerical values using
**One-Hot Encoding** for nominal values(Company, Ticker) and **Label Encoding** for ordinal values.
- **Exploratory Data Analysis (EDA):**  Analyzed distributions, correlations, and trends using Data Visualization
- **Feature Selection:** Used only numerical columns for training.
## Features
- **Open:** Opening price of the stock.
- **High:** Highest price during the trading period.
- **Low:** Lowest price during the trading period.
- **Volume:** Total number of shares traded.
## Target
- **Close:** Closing price of the stock.
## Requirements
- Install dependencies
```bash
  pip install streamlit
  pip install numpy
  pip install pandas
  pip install scikit-learn
  pip install mlflow
```
##  Model Development
- **Train-Test Split:** Split the dataset into training and testing sets (e.g., 80% training, 20% testing).
- **Normalize Numerical Features:** Scale the features to a standard range (e.g., 0 to 1) using
Standard Scaling.
- **Model Selection:** Start with regression models, such as:
Linear Regression,
Decision Trees,
Random Forest Regressors,
Gradient Boosting Models (e.g., XGBoost).
- **Model Training:** Train the selected models on the training dataset.

Python script for Model Training
```python
# MODEL BUILDING
# Test-Train Split
from sklearn.model_selection import train_test_split

X = final_df[['Open', 'High', 'Low','Volume']]
y = final_df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standard Scaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import pickle

with open("standard_scaler.pkl", "wb") as s:
    pickle.dump(scaler, s)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

models = [
("Linear Regressor", LinearRegression()),
("Decision Tree Regressor", DecisionTreeRegressor()),
("Random Forest Regressor", RandomForestRegressor()),
("XGBoost Regressor", XGBRegressor())
]

reports = []
# Model training
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    reports.append((name, model, rmse, mae, r2))


for name, model, rmse, mae, r2 in reports:
    print(f"Model: {name}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")
    print("\n")
```

## MLflow Integration
- **Set Up MLflow for Experiment Tracking:** Install MLflow and start the tracking server: 
```bash
  mlflow ui
```
- **Log experiments during model training:** Track metrics like MAE, RMSE, and R². Save trained model artifacts for later use.
- **Use MLflow for Model Comparison:** Compare different model performances in the MLflow UI.
Identify the best-performing model for deployment.
- **Deploy the Best Model with MLflow:** Register the best model in the MLflow model registry and make a pickle file of that model for later use.

Python code for MLflow Integration,
```python
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("FAANG Stock Close Price Prediction")
for name, model, rmse, mae, r2 in reports:
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        
        if name == "Linear Regressor":
            mlflow.sklearn.log_model(model, "LR_model")
        elif name == "Decision Tree Regressor":
            mlflow.sklearn.log_model(model, "DT_model")
        elif name == "Random Forest Regressor":
            mlflow.sklearn.log_model(model, "RF_model")
        elif name == "XGBoost Regressor":
            mlflow.xgboost.log_model(model, "XGB_model")

# REGISTER BEST MODEL
model_name ='Linear Regressor'
run_id = '5fdb4f39c93b4e2ea75d3bb83a0edde8'
model_uri = f'runs:/{run_id}/LR_model'

with mlflow.start_run(run_id=run_id):
    mlflow.register_model(model_uri= model_uri , name= model_name)

# PICKLE FILE 
import mlflow
import mlflow.sklearn
import pickle

# Set MLflow Tracking URI
#mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Define model name and version
model_name = "Linear Regressor"
model_version = 6# Change if you have a different version

# Load the model from MLflow Model Registry
model_uri = f"models:/{model_name}/{model_version}"
LR_model = mlflow.sklearn.load_model(model_uri)

# Save the model as a pickle file
with open("LinearModel.pkl", "wb") as f:
    pickle.dump(LR_model, f)

print("LR Model saved as 'LinearModel.pkl'")

```
- **Deployment with Streamlit:** Install Streamlit and create an app file (e.g., app.py).
- **Develop the following components:** User Input Section: Allow users to input stock details or use sliders/dropdowns. Prediction Output: Display predicted stock price.**
- **Run the app locally using:**
```bash
streamlit run app.py
```
Python code for Streamlit deployment,

```python
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
model_path = r"C:\Users\Haritha Sree D\.vscode\faang stock analysis\LinearModel.pkl"

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
            <li>Enter the required market values (Opening Price ($), High Price ($), Low Price ($), Trading Volume)</li>
            <li>Click 'Predict' to get the estimated closing price ($)</li>
        </ul>
        <p><i>Note: This tool uses historical data for predictions and is intended for educational purposes only, not as the sole basis for investment decisions.</i></p>
    </div>
""", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.markdown("<h3 style='color: black; margin-bottom: 0;'>📊 Input Parameters</h3>", unsafe_allow_html=True)

company = st.sidebar.selectbox(
    "Choose a company",
    options=["Apple", "Google", "Facebook", "Amazon", "Netflix"]
)

# Load models
with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(r"C:\Users\Haritha Sree D\.vscode\faang stock analysis\standard_scaler.pkl", "rb") as file:
    scaler_model = pickle.load(file)

# Input fields with validation
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

# Input validation
if feature_3 > feature_2:
    st.error("⚠️ Low price cannot be higher than High price!")
elif feature_1 < feature_3 or feature_1 > feature_2:
    st.warning("⚠️ Opening price should be between Low and High prices!")
else:
    # Create input array and scale values
    user_input = np.array([[feature_1, feature_2, feature_3, feature_4]])
    scaled_values = scaler_model.transform(user_input)

    # Prediction section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 Predict Stock Price"):
            prediction = model.predict(scaled_values)
            st.markdown(f"""
                <div style='background-color: #4CAF50; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;'>
                    <h2 style='color: white; margin: 0;'>Predicted Close Price</h2>
                    <h1 style='color: white; margin: 10px 0;'>${prediction[0]:.2f}</h1>
                    <p style='color: white; margin: 0;'>for {company}</p>
                </div>
            """, unsafe_allow_html=True)
```
## Model Evaluation
- **MAE (Mean Absolute Error):** Measures average error in absolute terms.
- **RMSE (Root Mean Squared Error):** Measures error with higher penalties on large deviations.
- **R² Score:** Determines how well the model fits the data.
- Use these metrics to evaluate performance of the model
  
## Streamlit Application Results
- Links for the Streamlit Application

    Local URL: http://localhost:8502

    Network URL: http://192.168.237.103:8502
- Output Visualization:
![Image](https://github.com/user-attachments/assets/6202fc3d-41bc-42ba-8c85-840f42a4ea00)


![Image](https://github.com/user-attachments/assets/71fcbe6d-07aa-4796-992b-3fb7c0b1c2ed)

  




 




  



