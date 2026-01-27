from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# Load the saved pipeline (includes preprocessing + model)
pipeline = joblib.load("model/house_price_gb_model.pkl")

app = FastAPI(title="House Price Prediction API")


class HouseFeatures(BaseModel):
    area: int
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: int
    guestroom: int
    basement: int
    hotwaterheating: int
    airconditioning: int
    parking: int
    prefarea: int
    furnishingstatus: int



@app.post("/predict")
def predict_price(features: HouseFeatures):
    # Convert to dictionary
    user_input = features.dict()

    # # Derive bathrooms from bedrooms
    # bedrooms = user_input['bedrooms']
    # if bedrooms == 1:
    #     bathrooms = 1
    # elif bedrooms == 2:
    #     bathrooms = 2
    # else:
    #     bathrooms = 2 + (bedrooms - 2)//2

    # # Hardcode hot water heating
    # hotwaterheating = 1  # or '0' based on your dataset

    # Prepare final input for model
    model_input = {
    "area": user_input['area'],
    "bedrooms": user_input['bedrooms'],
    "bathrooms": user_input['bathrooms'],
    "stories" : user_input['stories'],
    "mainroad" : user_input['mainroad'],
    "guestroom" : user_input['guestroom'],
    "basement" : user_input['basement'],
    "hotwaterheating" : user_input['hotwaterheating'],
    "airconditioning" : user_input['airconditioning'],
    "parking" : user_input['parking'],
    "prefarea" : user_input['prefarea'],
    "furnishingstatus" : user_input['furnishingstatus']
    }

    # Convert to array for model (respect feature order in pipeline)
    X = np.array([list(model_input.values())])

    # Predict in log space
    log_price = pipeline.predict(X)[0]

    # Convert back to real price
    real_price = np.exp(log_price)

    return {"predicted_price": round(real_price, 2)}
