from api.api import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running"}


def test_predict_endpoint_invalid_input():
    # Define invalid input data
    invalid_input = {
        "Warehouse_block": "Z",  # Invalid value
        "Mode_of_Shipment": "Flight",
        "Customer_care_calls": 3,
        "Customer_rating": 4,
        "Cost_of_the_Product": 200,
        "Prior_purchases": 2,
        "Product_importance": "medium",
        "Gender": "M",
        "Discount_offered": 10,
        "Weight_in_gms": 500,
    }

    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_endpoint_missing_field():
    # Define input data with a missing field
    missing_field_input = {
        "Warehouse_block": "A",
        "Mode_of_Shipment": "Flight",
        "Customer_care_calls": 3,
        "Customer_rating": 4,
        "Cost_of_the_Product": 200,
        "Prior_purchases": 2,
        "Product_importance": "medium",
        "Gender": "M",
        "Discount_offered": 10,
        # Missing Weight_in_gms
    }

    response = client.post("/predict", json=missing_field_input)
    assert response.status_code == 422  # Unprocessable Entity
