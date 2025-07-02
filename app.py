import streamlit as st
import pandas as pd
import joblib
import os

# Load saved model
model = joblib.load("my_model.pkl")

# Rebuild preprocessing pipeline (same structure used during training)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define numeric and categorical features
num_features = [
    "median_income", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households",
    "latitude", "longitude"
]
cat_features = ["ocean_proximity"]

# Numeric pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder())
])

# Full preprocessing pipeline
pipeline = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# Streamlit app layout
st.title("California House Price Prediction App")

# User inputs
income = st.number_input("Median Income", 0.0, 20.0, 3.0)
age = st.number_input("House Age", 0.0, 100.0, 20.0)
rooms = st.number_input("Total Rooms", 0.0, 100.0, 5.0)
bedrooms = st.number_input("Total Bedrooms", 0.0, 50.0, 1.0)
pop = st.number_input("Population", 0.0, 50000.0, 1000.0)
occup = st.number_input("Households", 0.0, 100.0, 3.0)
lat = st.number_input("Latitude", 30.0, 50.0, 34.0)
lon = st.number_input("Longitude", -125.0, -100.0, -118.0)
ocean = st.selectbox("Ocean Proximity", [
    "<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"
])

# Prediction
if st.button("Predict"):
    # Convert input into a DataFrame
    input_data = pd.DataFrame([[
        income, age, rooms, bedrooms, pop, occup, lat, lon, ocean
    ]], columns=num_features + cat_features)

    # Apply preprocessing
    input_prepared = pipeline.fit_transform(input_data)  # Fit because it's just one input

    # Make prediction
    prediction = model.predict(input_prepared)[0]
    st.success(f"Predicted Median House Value: ${prediction * 100000:.2f}")
