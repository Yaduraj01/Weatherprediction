import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import os

# Load the trained model (ensure the model file is in the same directory or provide the correct path)
model_path = 'my_model.keras'  # Replace with your actual model path
model = load_model(model_path)

# Function to make predictions based on user input
def predict_weather(input_data):
    # Reshape input data if necessary (based on the model's requirements)
    input_data = np.array(input_data).reshape(1, -1)  # Example for 1D data input
    prediction = model.predict(input_data)
    return prediction

# Streamlit UI
st.title('Weather Condition Prediction')
st.write("This app predicts the weather condition based on input features.")

# Input fields (example inputs based on your model)
temp = st.number_input('Temperature (°C)', min_value=-50, max_value=50, value=25)
humidity = st.number_input('Humidity (%)', min_value=0, max_value=100, value=60)
wind_speed = st.number_input('Wind Speed (km/h)', min_value=0, max_value=150, value=20)

# Example: input data for prediction
input_data = [temp, humidity, wind_speed]

# When the user clicks the button, make prediction
if st.button('Predict'):
    prediction = predict_weather(input_data)
    st.write(f"Predicted Weather Condition: {prediction}")
