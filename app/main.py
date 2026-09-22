from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])

    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4)
    }

app.mount("/static", StaticFiles(directory="static", html=True), name="static")