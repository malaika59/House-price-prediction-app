import streamlit as st
import pandas as pd
import joblib

# Load saved model and pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

st.title("🏡 House Price Prediction App")

# Input fields
income = st.number_input("Median Income", 0.0, 20.0, 3.0)
age = st.number_input("House Age", 0.0, 100.0, 20.0)
rooms = st.number_input("Total Rooms", 0.0, 50.0, 5.0)
bedrooms = st.number_input("Total Bedrooms", 0.0, 20.0, 1.0)
pop = st.number_input("Population", 0.0, 10000.0, 1000.0)
occup = st.number_input("Households", 0.0, 20.0, 3.0)
lat = st.number_input("Latitude", 30.0, 50.0, 34.0)
lon = st.number_input("Longitude", -125.0, -100.0, -118.0)
ocean = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

# When button is clicked
if st.button("Predict"):
    data = pd.DataFrame([[
        income, age, rooms, bedrooms, pop, occup, lat, lon, ocean
    ]], columns=[
        "median_income", "housing_median_age", "total_rooms", "total_bedrooms",
        "population", "households", "latitude", "longitude", "ocean_proximity"
    ])

    X_prepared = pipeline.transform(data)
    prediction = model.predict(X_prepared)[0]
    st.success(f"Predicted Median House Value: ${prediction * 100000:.2f}")
