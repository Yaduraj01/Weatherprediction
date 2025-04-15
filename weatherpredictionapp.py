import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the model
model = load_model('weathermodel.keras')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input size (adjust if needed)
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit app title and description
st.title("Weather Condition Detection")
st.write("Upload an image to predict the weather condition.")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Open image using PIL
    image = Image.open(uploaded_image)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Predict using the model
    prediction = model.predict(processed_image)
    
    # Mapping prediction to weather conditions
    class_labels = ["Cloudy", "Rainy", "Shiny", "Sunrise"]  # Modify based on your model's classes
    predicted_label = class_labels[np.argmax(prediction)]
    
    # Display the image and prediction
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write(f"Prediction: {predicted_label}")
