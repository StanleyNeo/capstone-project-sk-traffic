"""
recommender.py - Part 3, Task 7: traffic recommendation system.

Turns the project's analytical outputs into actionable advice for two
audiences - commuters and traffic operations staff - for a requested
(hour, day type, weather) combination.

Evidence used (all computed from data/processed/traffic_features.csv):
    1. Conditional congestion rate: how often the requested conditions were
       congested historically, versus the 25.02% base rate.
    2. Association rules mined in Day 6 (top-5 by lift, embedded below from
       part3_machine_learning/results/unsupervised_results.txt).
    3. Quietest alternative hours for the same day type.

Examples:
    python part3_machine_learning/recommendation_system/recommender.py --hour 17 --day weekday --weather Clear
    python part3_machine_learning/recommendation_system/recommender.py --hour 3 --day weekend --weather Clear
    python part3_machine_learning/recommendation_system/recommender.py --hour 8 --day weekday --weather Snow
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_CSV = BASE_DIR / "data" / "processed" / "traffic_features.csv"

WEATHER_CONDITIONS = ["Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist",
                      "Rain", "Smoke", "Snow", "Squall", "Thunderstorm"]

# Top-5 association rules for "High traffic" by lift - verified output of
# part3_machine_learning/unsupervised/train_unsupervised.py (Day 6).
TOP_RULES = [
    ("Afternoon (12-16) + Clouds + Weekday", 0.741, 2.96),
    ("Afternoon (12-16) + Weekday",          0.716, 2.86),
    ("Morning (06-11) + Weekday",            0.593, 2.37),
    ("Afternoon (12-16) + Clouds",           0.564, 2.25),
    ("Afternoon (12-16)",                    0.545, 2.18),
]

HIGH_RATIO, MODERATE_RATIO = 1.5, 0.8   # vs the 25.02% base congestion rate


def time_of_day(hour: int) -> str:
    if 6 <= hour <= 11:
        return "Morning (06-11)"
    if 12 <= hour <= 16:
        return "Afternoon (12-16)"
    if 17 <= hour <= 21:
        return "Evening (17-21)"
    return "Night (22-05)"


def risk_level(ratio: float) -> str:
    if ratio >= HIGH_RATIO:
        return "HIGH"
    if ratio >= MODERATE_RATIO:
        return "MODERATE"
    return "LOW"


def matching_rules(hour: int, day: str, weather: str):
    tod = time_of_day(hour)
    matched = []
    for antecedent, confidence, lift in TOP_RULES:
        parts = [p.strip() for p in antecedent.split("+")]
        ok = True
        for p in parts:
            if p in ("Weekday",) and day != "weekday":
                ok = False
            elif p == "Clouds" and weather != "Clouds":
                ok = False
            elif p.startswith(("Morning", "Afternoon", "Evening", "Night")) and p != tod:
                ok = False
        if ok:
            matched.append((antecedent, confidence, lift))
    return matched


def main():
    parser = argparse.ArgumentParser(
        description="Traffic recommendation for a given hour / day type / weather.")
    parser.add_argument("--hour", type=int, required=True, choices=range(24),
                        metavar="0-23", help="Hour of day")
    parser.add_argument("--day", required=True, choices=["weekday", "weekend"],
                        help="Day type")
    parser.add_argument("--weather", required=True, choices=WEATHER_CONDITIONS,
                        help="Weather condition")
    args = parser.parse_args()

    df = pd.read_csv(FEATURES_CSV, parse_dates=["date_time"])
    base_rate = df["congestion"].mean()

    mask = (df["hour"] == args.hour) & (df[f"weather_{args.weather}"] == 1)
    mask &= (df["day_of_week"] < 5) if args.day == "weekday" else (df["day_of_week"] >= 5)
    subset = df[mask]
    if len(subset) < 30:
        print(f"Warning: only {len(subset)} matching hours in history - estimate is rough.")
    cond_rate = subset["congestion"].mean()
    ratio = cond_rate / base_rate
    level = risk_level(ratio)
    avg_volume = subset["traffic_volume"].mean()

    # quietest alternative hours for the same day type
    day_df = df[(df["day_of_week"] < 5) if args.day == "weekday" else (df["day_of_week"] >= 5)]
    hourly = day_df.groupby("hour")["traffic_volume"].mean()
    quiet = hourly.nsmallest(3)

    print("=" * 64)
    print(f"TRAFFIC RECOMMENDATION - {args.day} {args.hour:02d}:00, {args.weather}")
    print("=" * 64)
    print(f"Historical matches:        {len(subset):,} hours")
    print(f"Congestion rate:           {cond_rate:.1%}  (base rate: {base_rate:.1%}, ratio {ratio:.2f}x)")
    print(f"Average volume:            {avg_volume:,.0f} vehicles/hour")
    print(f"RISK LEVEL:                {level}")

    matched = matching_rules(args.hour, args.day, args.weather)
    if matched:
        print("\nMatching association rules (Day 6):")
        for antecedent, confidence, lift in matched:
            print(f"  {{{antecedent}}} -> High traffic   conf {confidence:.3f}, lift {lift:.2f}")
    else:
        print("\nNo top-5 association rule matches this combination directly")
        print("(the mined rules cover Morning / Afternoon weekday patterns).")

    print("\nQuietest alternative hours (same day type):")
    for h, v in quiet.items():
        print(f"  {h:02d}:00  avg {v:,.0f} vehicles/hour")

    print("\nFor commuters:")
    if level == "HIGH":
        print(f"  Avoid {args.hour:02d}:00 if possible - shift to a quiet hour above,")
        print("  or switch to public transport for this trip.")
    elif level == "MODERATE":
        print("  Expect some delays; allow a small time buffer.")
    else:
        print("  Good time to travel - historically free-flowing conditions.")

    print("\nFor traffic operations:")
    if level == "HIGH":
        print("  Pre-position incident response, activate ramp metering /")
        print("  dynamic message signs, and consider transit signal priority.")
    elif level == "MODERATE":
        print("  Standard monitoring; be ready to escalate if incidents occur.")
    else:
        print("  Routine operations; suitable window for road maintenance.")
    print("=" * 64)
    logger.info("recommendation served: %s %02d:00 %s -> %s (ratio %.2f)",
                args.day, args.hour, args.weather, level, ratio)


if __name__ == "__main__":
    handler_file = logging.FileHandler(BASE_DIR / "pipeline.log", mode="a", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler_file.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=[handler_file])
    main()