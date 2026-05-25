from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import pandas as pd
import joblib

# Load trained models
model = joblib.load("models/outage_model.pkl")
anomaly_model = joblib.load("models/anomaly_model.pkl")

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": None,
            "probability": None,
            "status": None,
            "anomaly": None
        }
    )


@app.post("/analyze", response_class=HTMLResponse)
def analyze(
    request: Request,
    cpu_usage: float = Form(...),
    memory_usage: float = Form(...),
    latency: float = Form(...),
    error_rate: float = Form(...)
):

    input_data = pd.DataFrame([{
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "latency": latency,
        "error_rate": error_rate
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    anomaly_prediction = anomaly_model.predict(input_data)[0]

    if probability > 0.8:
        risk = "Critical"
    elif probability > 0.4:
        risk = "Warning"
    else:
        risk = "Healthy"

    if anomaly_prediction == -1:
        anomaly_status = "Anomaly Detected"
    else:
        anomaly_status = "Normal"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": int(prediction),
            "probability": round(probability * 100, 2),
            "status": risk,
            "anomaly": anomaly_status
        }
    )