from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()


# Define the request schema using Pydantic
class PredictRequest(BaseModel):
    Warehouse_block: Literal["A", "B", "C", "D", "F"]
    Mode_of_Shipment: Literal["Flight", "Ship", "Road"]
    Customer_care_calls: int
    Customer_rating: int
    Cost_of_the_Product: int
    Prior_purchases: int
    Product_importance: Literal["low", "medium", "high"]
    Gender: Literal["M", "F"]
    Discount_offered: int
    Weight_in_gms: int


# Define the /predict route
@app.post("/predict")
async def predict(data: PredictRequest):
    # Load the pre-trained Random Forest model
    model_path = "./rf_model_pipeline_final.pkl"
    model = joblib.load(model_path)
    print(model)

    # Create a DataFrame from the input data
    input_data = pd.DataFrame(
        [
            {
                "Warehouse_block": data.Warehouse_block,
                "Mode_of_Shipment": data.Mode_of_Shipment,
                "Customer_care_calls": data.Customer_care_calls,
                "Customer_rating": data.Customer_rating,
                "Cost_of_the_Product": data.Cost_of_the_Product,
                "Prior_purchases": data.Prior_purchases,
                "Product_importance": data.Product_importance,
                "Gender": data.Gender,
                "Discount_offered": data.Discount_offered,
                "Weight_in_gms": data.Weight_in_gms,
            }
        ],
        columns=[
            "ID",
            "Warehouse_block",
            "Mode_of_Shipment",
            "Customer_care_calls",
            "Customer_rating",
            "Cost_of_the_Product",
            "Prior_purchases",
            "Product_importance",
            "Gender",
            "Discount_offered",
            "Weight_in_gms",
        ],
    )

    # Make the prediction
    output = model.predict(input_data)

    # Make the prediction
    return {"message": "Prediction successful", "prediction": output[0].item()}


# Define the /health route
@app.get("/health")
async def health():
    return {"status": "API is running"}
