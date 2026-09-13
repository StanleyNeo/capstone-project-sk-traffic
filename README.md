# Smart City Traffic Intelligence — Capstone Project

**NUS SOC AI/ML/DS Capstone** · Stanley Neo · [github.com/StanleyNeo/capstone-project-sk-traffic](https://github.com/StanleyNeo/capstone-project-sk-traffic)

An end-to-end data science project on westbound I-94 (Minneapolis–St Paul) hourly
traffic: SQL analytics, statistics, a Power BI dashboard, a logged Python data
pipeline with feature engineering and a CLI app, and machine learning models
for congestion prediction.

## Dataset

`data/Metro_Interstate_Traffic_Volume.csv` — 48,204 hourly records, Oct 2012 – Sep 2018
(UCI ML Repository: Metro Interstate Traffic Volume). Known quality issues handled
by the pipeline: 7,629 duplicate timestamps, 10 zero-Kelvin temperatures, a
9,831 mm rainfall spike, and a holiday flag that marks only the 00:00 hour.

## Repository structure

```
├── data/
│   ├── Metro_Interstate_Traffic_Volume.csv   # raw dataset
│   └── processed/                            # pipeline outputs (clean + features)
├── part1_data_analytics/
│   ├── sql/          # SQLite loader + 10 analysis queries + results
│   ├── statistics/   # descriptive stats, correlation, probability analysis
│   ├── powerbi/      # traffic_intelligence_dashboard.pbix + PDF export
│   └── reports/      # insights_report.pdf
├── part2_python/
│   ├── pipeline.py             # Task 1: cleaning pipeline (logged)
│   ├── feature_engineering.py  # Task 2: 27 features + congestion target
│   ├── visualizations.py       # Task 3: 4 figures -> figures/
│   ├── cli_app/main.py         # Task 4: CLI (summary | hourly | weather | top)
│   ├── figures/                # generated charts
│   └── reports/                # part2_report.pdf
├── part3_machine_learning/     # (Day 6–7)
├── pipeline.log                # full audit trail of every pipeline run
└── requirements.txt
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## How to run

```powershell
python part2_python/pipeline.py                    # raw CSV -> data/processed/traffic_clean.csv
python part2_python/feature_engineering.py         # -> data/processed/traffic_features.csv
python part2_python/visualizations.py              # -> part2_python/figures/*.png
python part2_python/cli_app/main.py summary        # CLI: summary | hourly | weather <x> | top --n N
```

Every run appends to `pipeline.log` (INFO/WARNING/DEBUG with timestamps).

## Part 1 — Data analytics (complete)

- SQL: yearly trends (coverage-corrected), holiday vs non-holiday (−20%),
  holiday temperature comparison — see `part1_data_analytics/sql/`
- Statistics: mean 3,260 · median 3,380 · std 1,987 · r(temp, traffic) = 0.13
- Probability: P(congestion) = 14.7% · weather↔congestion not independent (OR 0.74)
- Power BI: 4 visuals + 3 KPI cards + 3 slicers → `part1_data_analytics/powerbi/`
- Findings: `part1_data_analytics/reports/insights_report.pdf`

## Part 2 — Python pipeline (complete)

- Cleaning: 48,204 → 40,575 rows (dedup, 0 K imputation, rain-spike fix) — all logged
- Features: 27 (cyclical time encodings, holiday fix 53 → 1,203 h, weather one-hot)
- Target: congestion = top quartile (≥ 4,952 veh/h) → 25.02% positive
- 4 figures + 4-command CLI — see `part2_python/reports/part2_report.pdf`

## Part 3 — Machine learning (in progress)

Supervised + unsupervised models, LSTM with SHAP, MLflow tracking, FastAPI mock
service, monitoring, and a recommendation component.

## License

MIT — see [LICENSE](LICENSE).