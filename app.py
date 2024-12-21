from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# Load the model
model_path = os.path.join(os.getcwd(), "silent-pug-207_final_model.joblib")
model = joblib.load(model_path)

# Create a FastAPI app
app = FastAPI()

# Define the input data schema
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Car Sales Model API!"}

# Prediction endpoint
@app.post("/predict/")
def predict(data: InputData):
    # Convert input data to a list of features for the model
    input_features = [[data.feature1, data.feature2, data.feature3, data.feature4]]
    
    # Make a prediction
    prediction = model.predict(input_features)
    
    return {"prediction": prediction[0]}

