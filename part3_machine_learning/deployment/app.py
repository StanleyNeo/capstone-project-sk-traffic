"""
app.py - Part 3, Task 5: mock deployment of the traffic models with FastAPI.

Serves the two Day 6 models as a REST API:
    GET  /health   -> service + model status
    POST /predict  -> traffic volume forecast + congestion probability
                      for one hour described by raw inputs
                      (the API does the feature engineering internally)

Run from anywhere:  python part3_machine_learning/deployment/app.py
Then open the interactive docs at http://127.0.0.1:8000/docs
"""

import logging
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "part3_machine_learning" / "models"

WEATHER_CONDITIONS = ["Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist",
                      "Rain", "Smoke", "Snow", "Squall", "Thunderstorm"]
RUSH_HOURS = [6, 7, 8, 9, 15, 16, 17, 18]
FEATURES = ["temp_c", "rain_1h", "snow_1h", "clouds_all", "hour", "day_of_week", "month",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
            "is_weekend", "is_rush_hour", "is_holiday",
            "weather_Clear", "weather_Clouds", "weather_Drizzle", "weather_Fog",
            "weather_Haze", "weather_Mist", "weather_Rain", "weather_Smoke",
            "weather_Snow", "weather_Squall", "weather_Thunderstorm"]


class TrafficInput(BaseModel):
    """Raw inputs for one hour - the API derives all 27 engineered features."""
    hour: int = Field(..., ge=0, le=23, description="Hour of day, 0-23")
    day_of_week: int = Field(..., ge=0, le=6, description="0 = Monday ... 6 = Sunday")
    month: int = Field(..., ge=1, le=12)
    temp_c: float = Field(..., description="Air temperature in Celsius")
    rain_1h: float = Field(0.0, ge=0.0, description="Rain in the last hour (mm)")
    snow_1h: float = Field(0.0, ge=0.0, description="Snow in the last hour (mm)")
    clouds_all: float = Field(50.0, ge=0.0, le=100.0, description="Cloud cover %")
    weather_main: str = Field("Clear", description="One of: " + ", ".join(WEATHER_CONDITIONS))
    is_holiday: bool = Field(False, description="True if the date is a public holiday")


class Prediction(BaseModel):
    predicted_traffic_volume: float
    congestion_probability: float
    congestion_predicted: bool
    risk_level: str


def build_feature_vector(inp: TrafficInput) -> pd.DataFrame:
    """Mirror of Day 4 feature engineering for a single hour."""
    row = {
        "temp_c": inp.temp_c, "rain_1h": inp.rain_1h, "snow_1h": inp.snow_1h,
        "clouds_all": inp.clouds_all, "hour": inp.hour,
        "day_of_week": inp.day_of_week, "month": inp.month,
        "hour_sin": math.sin(2 * math.pi * inp.hour / 24),
        "hour_cos": math.cos(2 * math.pi * inp.hour / 24),
        "dow_sin": math.sin(2 * math.pi * inp.day_of_week / 7),
        "dow_cos": math.cos(2 * math.pi * inp.day_of_week / 7),
        "month_sin": math.sin(2 * math.pi * inp.month / 12),
        "month_cos": math.cos(2 * math.pi * inp.month / 12),
        "is_weekend": int(inp.day_of_week >= 5),
        "is_rush_hour": int(inp.hour in RUSH_HOURS),
        "is_holiday": int(inp.is_holiday),
    }
    for w in WEATHER_CONDITIONS:
        row[f"weather_{w}"] = int(inp.weather_main == w)
    return pd.DataFrame([row])[FEATURES]


app = FastAPI(title="Smart City Traffic API",
              description="Mock deployment of the Metro Interstate traffic models",
              version="1.0.0")
regressor = None
classifier = None


@app.on_event("startup")
def load_models():
    global regressor, classifier
    reg_path = MODEL_DIR / "traffic_regressor.joblib"
    cls_path = MODEL_DIR / "congestion_classifier.joblib"
    if not reg_path.exists() or not cls_path.exists():
        logger.error("Model files missing in %s - run train_supervised.py first", MODEL_DIR)
        raise RuntimeError("Model files not found")
    regressor = joblib.load(reg_path)
    classifier = joblib.load(cls_path)
    logger.info("Models loaded from %s", MODEL_DIR)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": regressor is not None and classifier is not None}


@app.post("/predict", response_model=Prediction)
def predict(inp: TrafficInput):
    if inp.weather_main not in WEATHER_CONDITIONS:
        raise HTTPException(
            status_code=422,
            detail=f"weather_main must be one of {WEATHER_CONDITIONS}",
        )
    X = build_feature_vector(inp)
    X = X[list(regressor.feature_names_in_)]  # match the model's training column order
    volume = float(regressor.predict(X)[0])
    proba = float(classifier.predict_proba(X)[0, 1])
    congested = proba >= 0.5
    risk = "HIGH" if proba >= 0.75 else "MODERATE" if proba >= 0.4 else "LOW"
    logger.info("predict hour=%d dow=%d wx=%s -> volume=%.0f p_cong=%.3f (%s)",
                inp.hour, inp.day_of_week, inp.weather_main, volume, proba, risk)
    return Prediction(predicted_traffic_volume=round(volume, 1),
                      congestion_probability=round(proba, 4),
                      congestion_predicted=congested,
                      risk_level=risk)


if __name__ == "__main__":
    handler_file = logging.FileHandler(BASE_DIR / "pipeline.log", mode="a", encoding="utf-8")
    handler_console = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler_file.setFormatter(fmt)
    handler_console.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=[handler_file, handler_console])
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")