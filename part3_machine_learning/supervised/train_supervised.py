"""
train_supervised.py — Part 3, Task 1: Supervised learning.

Three experiments on data/processed/traffic_features.csv, all using a
TIME-BASED split (train = Oct 2012 – Dec 2017, test = Jan – Sep 2018) —
never a random shuffle on time-series data:

  1. Regression      predict traffic_volume   LinearRegression vs RandomForest
  2. Classification  predict congestion        LogisticRegression vs RandomForest
  3. High-risk proxy predict high_risk         RandomForest (class_weight=balanced)
                     high_risk = congestion AND (severe weather or heavy rain)

Metrics are printed, logged, and written to
part3_machine_learning/results/supervised_results.txt.
The two RandomForest models are saved to part3_machine_learning/models/
for the Day 7 FastAPI service.

Run from anywhere:  python part3_machine_learning/supervised/train_supervised.py
"""

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
IN_CSV = BASE_DIR / "data" / "processed" / "traffic_features.csv"
RESULTS_DIR = BASE_DIR / "part3_machine_learning" / "results"
MODELS_DIR = BASE_DIR / "part3_machine_learning" / "models"
RESULTS_FILE = RESULTS_DIR / "supervised_results.txt"
LOG_FILE = BASE_DIR / "pipeline.log"

SPLIT_DATE = "2018-01-01"
SEVERE_WEATHER = ["weather_Snow", "weather_Thunderstorm", "weather_Squall", "weather_Fog"]
HEAVY_RAIN_MM = 5.0

lines: list[str] = []  # collected for the results file


def report(text: str = "") -> None:
    """Log one line of output (console + file) and collect it for results."""
    logger.info(text)
    lines.append(text)


def load_and_split():
    df = pd.read_csv(IN_CSV, parse_dates=["date_time"])
    features = [c for c in df.columns
                if c not in ("date_time", "traffic_volume", "congestion")]

    severe = (df[SEVERE_WEATHER].sum(axis=1) > 0) | (df["rain_1h"] > HEAVY_RAIN_MM)
    df["high_risk"] = ((df["congestion"] == 1) & severe).astype(int)

    train = df[df["date_time"] < SPLIT_DATE]
    test = df[df["date_time"] >= SPLIT_DATE]
    report(f"Features: {len(features)} | train {len(train):,} rows "
           f"(< {SPLIT_DATE}) | test {len(test):,} rows (2018)")
    report(f"high_risk label: {df['high_risk'].sum():,} positives "
           f"({df['high_risk'].mean() * 100:.2f}% of all hours)")
    return train, test, features


def regression(train, test, features) -> None:
    report("\n=== 1. REGRESSION — predict traffic_volume ===")
    report(f"{'Model':<26} {'MAE':>8} {'RMSE':>8} {'R2':>7}")
    y_train, y_test = train["traffic_volume"], test["traffic_volume"]
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1),
    }
    for name, model in models.items():
        model.fit(train[features], y_train)
        pred = model.predict(test[features])
        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        r2 = r2_score(y_test, pred)
        report(f"{name:<26} {mae:8.1f} {rmse:8.1f} {r2:7.4f}")
        if name == "RandomForestRegressor":
            joblib.dump(model, MODELS_DIR / "traffic_regressor.joblib")
    report("Baseline reference: predicting the train mean gives MAE "
           f"{mean_absolute_error(y_test, [y_train.mean()] * len(y_test)):.1f}")


def classification(train, test, features) -> None:
    report("\n=== 2. CLASSIFICATION — predict congestion (top quartile) ===")
    report(f"{'Model':<26} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    y_train, y_test = train["congestion"], test["congestion"]
    models = {
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1),
    }
    for name, model in models.items():
        model.fit(train[features], y_train)
        pred = model.predict(test[features])
        proba = model.predict_proba(test[features])[:, 1]
        report(f"{name:<26} {accuracy_score(y_test, pred):7.4f} "
               f"{precision_score(y_test, pred):7.4f} {recall_score(y_test, pred):7.4f} "
               f"{f1_score(y_test, pred):7.4f} {roc_auc_score(y_test, proba):7.4f}")
        report(f"  confusion matrix [[TN,FP],[FN,TP]]: "
               f"{confusion_matrix(y_test, pred).tolist()}")
        if name == "RandomForestClassifier":
            joblib.dump(model, MODELS_DIR / "congestion_classifier.joblib")
            importance = pd.Series(model.feature_importances_, index=features)
            report("  top-5 features: " + ", ".join(
                f"{k} ({v:.3f})" for k, v in importance.nlargest(5).items()))


def high_risk(train, test, features) -> None:
    report("\n=== 3. HIGH-RISK PROXY — congestion AND severe weather ===")
    y_train, y_test = train["high_risk"], test["high_risk"]
    report(f"positives: train {y_train.sum():,} | test {y_test.sum():,} "
           "(heavily imbalanced — class_weight='balanced')")
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(train[features], y_train)
    pred = model.predict(test[features])
    proba = model.predict_proba(test[features])[:, 1]
    report(f"{'RandomForest(balanced)':<26} {accuracy_score(y_test, pred):7.4f} "
           f"{precision_score(y_test, pred):7.4f} {recall_score(y_test, pred):7.4f} "
           f"{f1_score(y_test, pred):7.4f} {roc_auc_score(y_test, proba):7.4f}")
    report(f"  confusion matrix [[TN,FP],[FN,TP]]: "
           f"{confusion_matrix(y_test, pred).tolist()}")
    report("  note: with ~1.5% positives, recall matters more than accuracy —")
    report("  a model that says 'never high-risk' scores 98.5% accuracy but is useless.")


def run() -> None:
    logger.info("Loading features from %s", IN_CSV)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train, test, features = load_and_split()
    regression(train, test, features)
    classification(train, test, features)
    high_risk(train, test, features)
    RESULTS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report(f"\nResults written to {RESULTS_FILE}")
    report(f"Models saved to {MODELS_DIR}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, mode="a"),
                  logging.StreamHandler(sys.stdout)],
    )
    try:
        run()
        logger.info("Supervised training finished successfully")
    except Exception:
        logger.error("Supervised training failed", exc_info=True)
        sys.exit(1)