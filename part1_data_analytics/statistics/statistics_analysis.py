"""Part 1 - Tasks 2 & 3: Descriptive statistics, correlation and probability analysis.

Task 2: traffic volume statistics + temperature/traffic correlation.
Task 3: probability, conditional probability, independence test and odds ratio
        for congestion (traffic_volume > 5,500) vs weather.

Reads the raw CSV (Part 1 analyses raw data; cleaning is Part 2) and writes
formatted results WITH interpretation to statistics_results.txt.
"""
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]  # repo root
CSV_PATH = BASE_DIR / "data" / "Metro_Interstate_Traffic_Volume.csv"
RESULTS_PATH = Path(__file__).resolve().parent / "statistics_results.txt"

CONGESTION_THRESHOLD = 5500   # vehicles per hour (per capstone instructions)
HIGH_TEMP_K = 292             # Kelvin (per capstone instructions)

sections = []


def section(title: str, body: str) -> None:
    sections.append(f"### {title}\n{body}\n")
    logger.info("Completed analysis section: %s", title)


def task2_descriptive_stats(df: pd.DataFrame) -> None:
    tv = df["traffic_volume"]
    body = f"""Count                : {tv.count():,}
Mean                 : {tv.mean():,.2f} vehicles/hour
Median               : {tv.median():,.0f} vehicles/hour
Std deviation (sample): {tv.std():,.2f}
Variance (sample)    : {tv.var():,.2f}
Minimum              : {tv.min():,}
Maximum              : {tv.max():,}
Range                : {tv.max() - tv.min():,}
Q1 / Q3 / IQR        : {tv.quantile(0.25):,.0f} / {tv.quantile(0.75):,.0f} / {tv.quantile(0.75) - tv.quantile(0.25):,.0f}
Skewness             : {tv.skew():.2f}

INTERPRETATION:
- A typical hour carries ~3,260 vehicles (mean) and the median (3,380) is close
  to the mean, so the distribution is roughly symmetric (skewness ~ -0.1).
- The standard deviation (~1,987) is LARGE relative to the mean (~61% of it):
  traffic volume is highly variable across hours - quiet overnight hours and
  busy rush hours differ by thousands of vehicles.
- The range spans 0 to 7,280. The 0 values are suspicious (a real freeway
  sensor rarely reads exactly zero) and are flagged for Part 2 data cleaning.
- Half of all hours fall between 1,193 and 4,933 vehicles (IQR = 3,740)."""
    section("Task 2.1 - Traffic volume descriptive statistics", body)


def task2_correlation(df: pd.DataFrame) -> None:
    r = df["temp"].corr(df["traffic_volume"])
    body = f"""Pearson correlation coefficient r (temp vs traffic_volume) = {r:.4f}

INTERPRETATION:
- DIRECTION: positive - warmer hours are very slightly associated with higher
  traffic volume.
- STRENGTH: r ~ 0.13 is a WEAK correlation (|r| < 0.3). Temperature explains
  only r^2 = {r**2:.1%} of the variance in traffic volume - practically nothing.
- CORRELATION IS NOT CAUSATION: this weak link does not mean warm weather
  causes traffic. Both variables are driven by confounders - time of day and
  day of week (rush hours vs nights, weekdays vs weekends/holidays) and
  seasonality. A proper causal claim would need to control for those."""
    section("Task 2.2 - Correlation: temperature vs traffic volume", body)


def task3_probability(df: pd.DataFrame) -> None:
    n = len(df)
    cong = df["traffic_volume"] > CONGESTION_THRESHOLD
    clear = df["weather_main"] == "Clear"
    cloudy = df["weather_main"] == "Clouds"
    hot = df["temp"] > HIGH_TEMP_K

    p_cong, p_clear = cong.mean(), clear.mean()
    p_both = (cong & clear).mean()
    p_clear_given_cong = (cong & clear).sum() / cong.sum()
    p_hot_given_cong = (cong & hot).sum() / cong.sum()
    p_product = p_cong * p_clear

    p_cong_clear = (cong & clear).sum() / clear.sum()
    p_cong_cloudy = (cong & cloudy).sum() / cloudy.sum()
    odds_clear = p_cong_clear / (1 - p_cong_clear)
    odds_cloudy = p_cong_cloudy / (1 - p_cong_cloudy)
    odds_ratio = odds_clear / odds_cloudy

    body = f"""Definition: CONGESTION = traffic_volume > {CONGESTION_THRESHOLD:,} vehicles/hour
Dataset size N = {n:,} hourly records

3.1 BASIC PROBABILITY
P(Congestion)            = {cong.sum():,} / {n:,} = {p_cong:.4f}  (~14.7% of hours)
P(Clear weather)         = {clear.sum():,} / {n:,} = {p_clear:.4f}
P(Congestion AND Clear)  = {(cong & clear).sum():,} / {n:,} = {p_both:.4f}

3.2 CONDITIONAL PROBABILITY
P(Clear | Congestion)      = {(cong & clear).sum():,} / {cong.sum():,} = {p_clear_given_cong:.4f}
P(Temp>292K | Congestion)  = {(cong & hot).sum():,} / {cong.sum():,} = {p_hot_given_cong:.4f}

INDEPENDENCE TEST  P(A n B) =? P(A) x P(B)
P(Congestion) x P(Clear) = {p_product:.4f}
P(Congestion n Clear)    = {p_both:.4f}
-> {p_both:.4f} != {p_product:.4f}, so weather and congestion are NOT independent.
   The joint probability is LOWER than the independence baseline: congestion
   happens slightly LESS often under clear skies than independence predicts.

ODDS RATIO of congestion: Clear vs Cloudy weather
P(Congestion | Clear)  = {(cong & clear).sum():,} / {clear.sum():,} = {p_cong_clear:.4f}  -> odds = {odds_clear:.4f}
P(Congestion | Cloudy) = {(cong & cloudy).sum():,} / {cloudy.sum():,} = {p_cong_cloudy:.4f}  -> odds = {odds_cloudy:.4f}
Odds ratio (Clear / Cloudy) = {odds_ratio:.4f}

CONCLUSION:
- Congestion occurs in ~14.7% of all hours - it is a rush-hour phenomenon.
- The odds of congestion in clear weather are only ~0.74x the odds in cloudy
  weather (about 26% lower). Clear weather does NOT increase congestion risk.
- This is consistent with the weak correlation in Task 2.2: congestion is
  driven mainly by time-of-day and day-of-week demand, not by weather.
  Weather conditions show only mild secondary effects (cloudy 17.1% vs
  clear 13.2% congestion rates; snow lowest at 9.5%)."""
    section("Task 3 - Probability and congestion analysis", body)


def main() -> None:
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        logger.error("CSV not found at %s", CSV_PATH, exc_info=True)
        sys.exit(1)
    logger.info("Loaded %d rows x %d columns from %s", df.shape[0], df.shape[1], CSV_PATH.name)

    task2_descriptive_stats(df)
    task2_correlation(df)
    task3_probability(df)

    RESULTS_PATH.write_text("\n".join(sections), encoding="utf-8")
    logger.info("Results written to %s", RESULTS_PATH)
    # print is used only to display results to the user running the script
    print("\n".join(sections))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )
    main()