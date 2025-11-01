import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware

# --- 1. Define Application and Load Model ---

# Define the model file path (Ensure your partner's file is named this)
MODEL_FILE = 'best_car_price_model.pkl'

# Initialize the FastAPI app
app = FastAPI(
    title="Used Car Price Predictor API",
    description="API for predicting the price of a used car based on its features.",
    version="1.0.0"
)

# --- ADD CORS MIDDLEWARE TO ALLOW FRONTEND CONNECTION ---
origins = [
    "*",  # Allows requests from *any* domain (necessary for the Render/GitHub Pages environment)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],    # Allows POST requests
    allow_headers=["*"],    # Allows all headers
)
# --------------------------------------------------------

# Load the model from the file
model = None

@app.on_event("startup")
def load_model():
    """
    Load the model from disk when the application starts.
    """
    global model
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found.")

    try:
        model = joblib.load(MODEL_FILE)
        print(f"Model '{MODEL_FILE}' loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


# --- 2. Define the Input Data Model (Pydantic) ---

class CarFeatures(BaseModel):
    """
    Defines the input features. This matches the JSON schema being sent by the frontend.
    """
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
    """
    return {"message": "Welcome to the Used Car Price Predictor API!"}


@app.post("/selling_price", tags=["Prediction"])
async def predict_price(features: CarFeatures):
    """
    Predict the selling price of a used car.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please try again later.")

    try:
        # --- 4. Prepare Data for Model ---
        # Convert the Pydantic model to a dictionary
        input_data = features.dict()
        input_df = pd.DataFrame([input_data])

        # Ensure column order matches training
        ordered_cols = [
            'vehicle_age', 'km_driven', 'seller_type', 'fuel_type',
            'transmission_type', 'mileage_cleaned', 'engine_cleaned',
            'max_power_cleaned', 'seats', 'brand', 'model'
        ]
        input_df = input_df[ordered_cols]

        # --- 5. Make Prediction ---
        prediction_log = model.predict(input_df)
        predicted_log_price = prediction_log[0]
        
        # Convert the log price back to the original scale
        predicted_price = np.expm1(predicted_log_price)

        if predicted_price < 0:
            predicted_price = 0

        # --- 6. Return Response (Matching FE Expectation: predicted_price_inr) ---
        return {
            "predicted_price_inr": round(predicted_price, 0),
            "model_input": input_data # For debugging
        }

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# --- 7. Run the App (for local testing) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
