import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('energy_forecasting_model.pkl')

# Feature preprocessing function
def preprocess_input(year, month, sector, sector_value):
    # Calculate features
    year_trend = year - 1973  # Adjust based on dataset
    season_mapping = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3}
    season = get_season(month)
    season_code = season_mapping[season]
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Combine into a DataFrame
    feature_dict = {
        'Year_Trend': [year_trend],
        'Season_Code': [season_code],
        'Month_Sin': [month_sin],
        'Month_Cos': [month_cos],
        'Hydroelectric Power': [0.0],  # Default values
        'Solar Energy': [0.0],
    }

    # Set the user-provided value for the selected sector
    feature_dict[sector] = [sector_value]

    return pd.DataFrame(feature_dict)

# Function to determine season based on month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

# Streamlit app
st.title("🌱 Renewable Energy Forecasting App")
st.write("Predict total renewable energy based on year, month, and energy sources.")

# Input fields
year = st.number_input("Enter the year:", min_value=1973, max_value=2100, step=1)
month = st.selectbox("Select the month:", list(range(1, 13)))

# Add sector selection
sector = st.selectbox(
    "Select the energy sector for prediction:",
    ["Hydroelectric Power", "Solar Energy"]
)

# Allow user to specify sector value
sector_value = st.slider(f"Enter the {sector} production (in MW):", 0.0, 10.0, step=0.1, value=1.0)

# Prediction button
if st.button("Predict"):
    try:
        # Preprocess input
        features = preprocess_input(year, month, sector, sector_value)
        # Make prediction
        prediction = model.predict(features)[0]
        st.success(f"🌟 Predicted Total Renewable Energy for {sector}: {prediction:.2f} MW")
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
