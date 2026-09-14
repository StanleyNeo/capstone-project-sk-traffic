"""
track_experiments.py - Part 3, Task 4: experiment tracking with MLflow.

Re-trains the three Day 6 model configurations and logs each one as an
MLflow run (parameters, metrics, model artifact) so experiments can be
compared in the MLflow UI instead of scrolling terminal output.

Runs tracked:
    1. linear-regression      (baseline)
    2. random-forest-regressor
    3. random-forest-classifier

After running, launch the UI from THIS folder:
    cd part3_machine_learning/mlflow
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
then open http://127.0.0.1:5000

Run from anywhere:  python part3_machine_learning/mlflow/track_experiments.py
"""

import logging
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_CSV = BASE_DIR / "data" / "processed" / "traffic_features.csv"
DB_PATH = Path(__file__).resolve().parent / "mlflow.db"
SPLIT_DATE = pd.Timestamp("2018-01-01")

FEATURES = ["temp_c", "rain_1h", "snow_1h", "clouds_all", "hour", "day_of_week", "month",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
            "is_weekend", "is_rush_hour", "is_holiday",
            "weather_Clear", "weather_Clouds", "weather_Drizzle", "weather_Fog",
            "weather_Haze", "weather_Mist", "weather_Rain", "weather_Smoke",
            "weather_Snow", "weather_Squall", "weather_Thunderstorm"]


def load_and_split():
    """Same leakage-free time split as Day 6."""
    df = pd.read_csv(FEATURES_CSV, parse_dates=["date_time"])
    train = df[df["date_time"] < SPLIT_DATE]
    test = df[df["date_time"] >= SPLIT_DATE]
    logger.info("Train %s rows | Test %s rows", f"{len(train):,}", f"{len(test):,}")
    return train, test


def common_params(extra):
    params = {"split_date": str(SPLIT_DATE.date()), "n_features": len(FEATURES),
              "train_rows": 34042, "test_rows": 6533}
    params.update(extra)
    return params


def track_linear_regression(Xtr, ytr, Xte, yte):
    with mlflow.start_run(run_name="linear-regression"):
        model = LinearRegression().fit(Xtr, ytr)
        pred = model.predict(Xte)
        mlflow.log_params(common_params({"model_type": "LinearRegression"}))
        mlflow.log_metrics({"mae": mean_absolute_error(yte, pred),
                            "rmse": mean_squared_error(yte, pred) ** 0.5,
                            "r2": r2_score(yte, pred)})
        mlflow.sklearn.log_model(model, name="model")
        logger.info("Logged run: linear-regression")


def track_rf_regressor(Xtr, ytr, Xte, yte):
    with mlflow.start_run(run_name="random-forest-regressor"):
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(Xtr, ytr)
        pred = model.predict(Xte)
        mlflow.log_params(common_params({"model_type": "RandomForestRegressor",
                                         "n_estimators": 100, "random_state": 42}))
        mlflow.log_metrics({"mae": mean_absolute_error(yte, pred),
                            "rmse": mean_squared_error(yte, pred) ** 0.5,
                            "r2": r2_score(yte, pred)})
        mlflow.sklearn.log_model(model, name="model")
        logger.info("Logged run: random-forest-regressor")


def track_rf_classifier(Xtr, ytr, Xte, yte):
    with mlflow.start_run(run_name="random-forest-classifier"):
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(Xtr, ytr)
        pred = model.predict(Xte)
        mlflow.log_params(common_params({"model_type": "RandomForestClassifier",
                                         "n_estimators": 100, "random_state": 42,
                                         "target": "congestion (traffic_volume >= Q3)"}))
        mlflow.log_metrics({"accuracy": accuracy_score(yte, pred),
                            "precision": precision_score(yte, pred),
                            "recall": recall_score(yte, pred),
                            "f1": f1_score(yte, pred)})
        mlflow.sklearn.log_model(model, name="model")
        logger.info("Logged run: random-forest-classifier")


def main():
    train, test = load_and_split()
    Xtr, Xte = train[FEATURES], test[FEATURES]

    tracking_uri = "sqlite:///" + DB_PATH.as_posix()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("traffic-congestion")
    logger.info("MLflow tracking URI: %s", tracking_uri)

    track_linear_regression(Xtr, train["traffic_volume"], Xte, test["traffic_volume"])
    track_rf_regressor(Xtr, train["traffic_volume"], Xte, test["traffic_volume"])
    track_rf_classifier(Xtr, train["congestion"], Xte, test["congestion"])

    logger.info("All 3 runs logged. View them with:")
    logger.info("    cd part3_machine_learning/mlflow")
    logger.info("    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")


if __name__ == "__main__":
    handler_file = logging.FileHandler(BASE_DIR / "pipeline.log", mode="a", encoding="utf-8")
    handler_console = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler_file.setFormatter(fmt)
    handler_console.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=[handler_file, handler_console])
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    main()