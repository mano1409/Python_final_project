
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
MODEL_FILENAME = "silent-pug-207_final_model.joblib"  # Replace with your model file
model = joblib.load(MODEL_FILENAME)

# Define a function for preprocessing user input
def preprocess_input(data, label_encoders):
    for col, encoder in label_encoders.items():
        if col in data:
            data[col] = encoder.transform([data[col]])[0]
    return data

# Load label encoders if applicable
label_encoders = {
    # Replace with the label encoders you used for categorical features
    # Example:
    # "Transmission": joblib.load("transmission_encoder.joblib"),
    # "Body_Style": joblib.load("body_style_encoder.joblib"),
}

# Streamlit app UI
st.title("Car Sales Classifier")
st.write("Interact with the model to classify car attributes in real-time.")

# User inputs
st.sidebar.header("Car Information")
company = st.sidebar.text_input("Company", "Toyota")
model_name = st.sidebar.text_input("Model", "Camry")
engine = st.sidebar.text_input("Engine", "V6")
transmission = st.sidebar.selectbox("Transmission", ["Automatic", "Manual"])
color = st.sidebar.text_input("Color", "Red")
annual_income = st.sidebar.number_input("Annual Income", min_value=0, value=50000)
price = st.sidebar.number_input("Price", min_value=0, value=25000)

# Create input DataFrame
input_data = {
    "Company": company,
    "Model": model_name,
    "Engine": engine,
    "Transmission": transmission,
    "Color": color,
    "Annual_Income": annual_income,
    "Price": price,
}

# Preprocess the data
processed_data = preprocess_input(input_data, label_encoders)

# Prediction
if st.button("Classify"):
    # Convert to DataFrame for prediction
    input_df = pd.DataFrame([processed_data])
    prediction = model.predict(input_df)
    st.write(f"Predicted Body Style: {prediction[0]}")
