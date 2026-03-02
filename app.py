import re
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Page config
st.set_page_config(page_title="Power Consumption Prediction")

st.title("⚡ Tetuan City Power Consumption Prediction")

# Load dataset
df = pd.read_csv("Tetuan City power consumption.csv")

# Drop DateTime if exists
if "DateTime" in df.columns:
    df = df.drop("DateTime", axis=1)

# Define features and target
X = df.drop("Zone 1 Power Consumption", axis=1)
y = df["Zone 1 Power Consumption"]

# Train model directly inside app (no pickle needed)
model = DecisionTreeRegressor(max_depth=6, random_state=42)
model.fit(X, y)

st.write("Enter input values:")

# Create input fields dynamically based on dataset columns
inputs = []

for column in X.columns:
    value = st.number_input(f"{column}", value=float(X[column].mean()))
    inputs.append(value)

# Prediction button
if st.button("Predict Power Consumption"):

    input_data = np.array([inputs])
    prediction = model.predict(input_data)

    st.success(f"Predicted Zone 1 Power Consumption: {prediction[0]:.2f}")