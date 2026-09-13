"""
feature_engineering.py — Part 2, Task 2: Feature engineering.

Reads data/processed/traffic_clean.csv (output of pipeline.py) and builds the
model-ready feature table data/processed/traffic_features.csv.

Features created:
  - temp_c                     temperature in Celsius
  - hour/day_of_week/month     calendar components
  - hour_sin/cos, dow_sin/cos, month_sin/cos   cyclical encodings
  - is_weekend, is_rush_hour   calendar flags (rush = 06-09 & 15-18)
  - is_holiday                 date-based fix: the raw flag marks only the
                               00:00 hour of a holiday; we flag all 24 hours
  - weather_*                  one-hot encoding of weather_main (11 columns)
  - congestion                 TARGET: 1 if traffic_volume >= Q3 (top quartile)

Run from anywhere:  python part2_python/feature_engineering.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
IN_CSV = BASE_DIR / "data" / "processed" / "traffic_clean.csv"
OUT_CSV = BASE_DIR / "data" / "processed" / "traffic_features.csv"
LOG_FILE = BASE_DIR / "pipeline.log"

RUSH_HOURS = [6, 7, 8, 9, 15, 16, 17, 18]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar components plus cyclical (sin/cos) encodings."""
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek  # 0 = Monday
    df["month"] = df["date_time"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    logger.debug("Added calendar + cyclical features (hour, dow, month)")
    return df


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flags: weekend, rush hour, and date-based holiday."""
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin(RUSH_HOURS).astype(int)

    # Note: reading the clean CSV turns the "None" strings back into NaN
    # (pandas default), so test with notna() as well as != "None".
    has_holiday = df["holiday"].notna() & (df["holiday"] != "None")
    raw_flag = int(has_holiday.sum())
    holiday_dates = set(df.loc[has_holiday, "date_time"].dt.date)
    df["is_holiday"] = df["date_time"].dt.date.isin(holiday_dates).astype(int)
    logger.info(
        "Holiday fix: raw flag marked %d rows (00:00 only); "
        "date-based is_holiday flags %d hours across %d holiday dates",
        raw_flag, int(df["is_holiday"].sum()), len(holiday_dates),
    )
    return df


def add_weather_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode weather_main (drop_first=False keeps all 11)."""
    dummies = pd.get_dummies(df["weather_main"], prefix="weather").astype(int)
    df = pd.concat([df, dummies], axis=1)
    logger.debug("One-hot encoded weather_main into %d columns", dummies.shape[1])
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Congestion target: top quartile of traffic_volume (>= Q3)."""
    q3 = df["traffic_volume"].quantile(0.75)
    df["congestion"] = (df["traffic_volume"] >= q3).astype(int)
    rate = df["congestion"].mean() * 100
    logger.info(
        "Target built: congestion = traffic_volume >= Q3 (%.0f) -> "
        "%d positives (%.2f%% of %d rows)",
        q3, int(df["congestion"].sum()), rate, len(df),
    )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the final model-ready table."""
    df["temp_c"] = df["temp"] - 273.15
    df = add_time_features(df)
    df = add_flags(df)
    df = add_weather_dummies(df)
    df = add_target(df)

    drop_cols = ["holiday", "temp", "weather_main", "weather_description"]
    df = df.drop(columns=drop_cols)

    assert int(df.isna().sum().sum()) == 0, "no NaNs allowed in feature table"
    logger.info(
        "Feature table: %d rows x %d columns (%d features + date_time, "
        "traffic_volume, congestion)",
        df.shape[0], df.shape[1], df.shape[1] - 3,
    )
    return df


def run() -> None:
    logger.info("Loading cleaned data from %s", IN_CSV)
    df = pd.read_csv(IN_CSV, parse_dates=["date_time"])
    df = build_features(df)
    df.to_csv(OUT_CSV, index=False)
    logger.info("Wrote features -> %s", OUT_CSV)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, mode="a"),
                  logging.StreamHandler(sys.stdout)],
    )
    try:
        run()
        logger.info("Feature engineering finished successfully")
    except Exception:
        logger.error("Feature engineering failed", exc_info=True)
        sys.exit(1)