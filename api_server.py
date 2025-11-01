import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

# --- 1. Define Application and Load Model ---

# Define the model file path
# Assumes the model file is in the same directory as this script.
# Make sure your saved model is named 'best_car_price_model.pkl'
MODEL_FILE = 'best_car_price_model.pkl'

# Initialize the FastAPI app
app = FastAPI(
    title="Used Car Price Predictor API",
    description="API for predicting the price of a used car based on its features.",
    version="1.0.0"
)

# Load the model from the file
model = None

@app.on_event("startup")
def load_model():
    """
    Load the model from disk when the application starts.
    This ensures the model is in memory and ready for predictions.
    """
    global model
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found.")

    try:
        model = joblib.load(MODEL_FILE)
        print(f"Model '{MODEL_FILE}' loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # In a real app, you might want to stop the server from starting
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


# --- 2. Define the Input Data Model (Pydantic) ---

class CarFeatures(BaseModel):
    """
    Defines the input features for a single car price prediction.
    These must match the features your CatBoost model was trained on.
    """
    # Order matches your CatBoost training:
    vehicle_age: int = Field(..., example=5, description="Age of the vehicle in years")
    km_driven: int = Field(..., example=70000, description="Total kilometers driven")
    seller_type: Literal['Individual', 'Dealer', 'Trustmark Dealer'] = Field(..., example="Individual")
    fuel_type: Literal['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'] = Field(..., example="Petrol")
    transmission_type: Literal['Manual', 'Automatic'] = Field(..., example="Manual")
    mileage_cleaned: float = Field(..., example=18.5, description="Mileage (kmpl or km/kg)")
    engine_cleaned: float = Field(..., example=1197.0, description="Engine displacement in CC")
    max_power_cleaned: float = Field(..., example=82.0, description="Max power in bhp")
    seats: int = Field(..., example=5, description="Number of seats")
    brand: str = Field(..., example="Maruti", description="Brand of the car")
    model: str = Field(..., example="Swift", description="Model of the car")

    class Config:
        schema_extra = {
            "example": {
                "vehicle_age": 8,
                "km_driven": 120000,
                "seller_type": "Individual",
                "fuel_type": "Diesel",
                "transmission_type": "Manual",
                "mileage_cleaned": 22.3,
                "engine_cleaned": 1248.0,
                "max_power_cleaned": 88.7,
                "seats": 5,
                "brand": "Maruti",
                "model": "Swift Dzire"
            }
        }

# --- 3. Define API Endpoints ---

@app.get("/", tags=["Health Check"])
async def root():
    """
    Root endpoint for health checking.
    Returns a welcome message if the API is running.
    """
    return {"message": "Welcome to the Used Car Price Predictor API!"}


@app.post("/predict/", tags=["Prediction"])
async def predict_price(features: CarFeatures):
    """
    Predict the selling price of a used car.

    Takes a JSON object with car features and returns the
    predicted price in Rupees.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please try again later.")

    try:
        # --- 4. Prepare Data for Model ---
        # Convert the Pydantic model to a dictionary
        input_data = features.dict()

        # Create a pandas DataFrame from the dictionary
        # The model expects a DataFrame with the exact column names
        # it was trained on, in the correct order.
        input_df = pd.DataFrame([input_data])

        # Ensure column order matches training
        # (Based on your CatBoost features_to_use_catboost list)
        ordered_cols = [
            'vehicle_age', 'km_driven', 'seller_type', 'fuel_type',
            'transmission_type', 'mileage_cleaned', 'engine_cleaned',
            'max_power_cleaned', 'seats', 'brand', 'model'
        ]
        input_df = input_df[ordered_cols]

        # --- 5. Make Prediction ---
        # Predict the log_price
        prediction_log = model.predict(input_df)

        # The prediction is an array, get the first element
        predicted_log_price = prediction_log[0]

        # Convert the log price back to the original scale
        predicted_price = np.expm1(predicted_log_price)

        # Ensure the price is not negative
        if predicted_price < 0:
            predicted_price = 0

        # --- 6. Return Response ---
        return {
            "predicted_price_inr": round(predicted_price, 2),
            "model_input": input_data
        }

    except Exception as e:
        # Catch any errors during the prediction process
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# --- 7. Run the App (for local testing) ---
if __name__ == "__main__":
    import uvicorn
    # This allows running the app locally with `python api_server.py`
    uvicorn.run(app, host="127.0.0.1", port=8000)