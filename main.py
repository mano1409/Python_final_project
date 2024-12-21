from fastapi import FastAPI
import joblib

app = FastAPI()

# Load your model
model = joblib.load("silent-pug-207_final_model.joblib")

@app.get("/")
def home():
    return {"message": "FastAPI Model Serving"}

@app.post("/predict")
def predict(data: list):
    prediction = model.predict(data)
    return {"prediction": prediction}

