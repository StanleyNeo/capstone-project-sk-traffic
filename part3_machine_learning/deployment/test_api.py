"""
test_api.py - Part 3, Task 5: smoke tests for the FastAPI deployment.

Uses FastAPI's TestClient (no live server needed) to verify:
    1. /health reports models loaded
    2. weekday evening rush hour -> high volume, congestion likely
    3. Sunday 03:00 -> low volume, congestion unlikely
    4. invalid input (hour = 25) -> HTTP 422 validation error

Run from anywhere:  python part3_machine_learning/deployment/test_api.py
"""

import logging
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import app  # noqa: E402

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[2]


def main():
    with TestClient(app) as client:
        # 1. health
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["models_loaded"], r.text
        logger.info("PASS  /health -> %s", r.json())

        # 2. weekday 17:00 rush hour, clear weather
        r = client.post("/predict", json={
            "hour": 17, "day_of_week": 2, "month": 6, "temp_c": 20.0,
            "weather_main": "Clear"})
        assert r.status_code == 200, r.text
        body = r.json()
        logger.info("PASS  weekday 17:00 Clear -> %s", body)
        assert body["predicted_traffic_volume"] > 4000, "rush hour should predict high volume"
        assert body["congestion_probability"] > 0.5, "rush hour should be congestion-prone"

        # 3. Sunday 03:00
        r = client.post("/predict", json={
            "hour": 3, "day_of_week": 6, "month": 6, "temp_c": 15.0,
            "weather_main": "Clear"})
        assert r.status_code == 200, r.text
        body = r.json()
        logger.info("PASS  Sunday 03:00 Clear  -> %s", body)
        assert body["predicted_traffic_volume"] < 1500, "3am Sunday should be quiet"
        assert body["congestion_probability"] < 0.5, "3am Sunday should not be congested"

        # 4. invalid hour -> 422
        r = client.post("/predict", json={
            "hour": 25, "day_of_week": 2, "month": 6, "temp_c": 20.0})
        assert r.status_code == 422, r.text
        logger.info("PASS  hour=25 rejected with HTTP 422 (input validation works)")

        # 5. invalid weather -> 422 with helpful message
        r = client.post("/predict", json={
            "hour": 8, "day_of_week": 2, "month": 6, "temp_c": 20.0,
            "weather_main": "banana"})
        assert r.status_code == 422, r.text
        logger.info("PASS  weather_main='banana' rejected with HTTP 422")

    logger.info("All 5 API tests passed.")


if __name__ == "__main__":
    handler_file = logging.FileHandler(BASE_DIR / "pipeline.log", mode="a", encoding="utf-8")
    handler_console = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler_file.setFormatter(fmt)
    handler_console.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=[handler_file, handler_console])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()