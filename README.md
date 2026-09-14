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
├── part3_machine_learning/
│   ├── supervised/             # train_supervised.py: regression + classification + high-risk
│   ├── unsupervised/           # train_unsupervised.py: K-means + association rules
│   ├── notebooks/              # lstm_shap_analysis.ipynb + SHAP figures
│   ├── mlflow/                 # track_experiments.py (3 runs, SQLite backend)
│   ├── deployment/             # FastAPI app.py + test_api.py (5 smoke tests)
│   ├── monitoring/             # drift_check.py + drift_report.txt (PASS/ALERT)
│   ├── recommendation_system/  # recommender.py CLI advisor
│   ├── models/                 # *.joblib (gitignored — regenerate via train_supervised.py)
│   ├── results/                # supervised_results.txt + unsupervised_results.txt
│   └── reports/                # part3_ml_report.pdf + responsible_ai_report.pdf + evidence
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

## Part 3 — Machine Learning

Supervised + unsupervised models, LSTM with SHAP, MLflow tracking, FastAPI mock
service, monitoring, and a recommendation component.

**Supervised learning** (`part3_machine_learning/supervised/`) — leakage-free time-based
split (train < 2018-01-01: 34,042 h; test = 2018: 6,533 h):

| Task | Best model | Key metrics |
|------|-----------|-------------|
| Volume regression | RandomForest (100 trees) | MAE 241.2, RMSE 406.8, R² 0.9575 (naive baseline MAE 1,728.3) |
| Congestion classification | RandomForest | Accuracy 0.941, F1 0.885, ROC-AUC 0.983 |
| High-risk hours (1.46% positive) | RandomForest, balanced weights | Recall 0.949, precision 0.689 |

**Unsupervised learning** (`part3_machine_learning/unsupervised/`) — K-means (k = 3 by
silhouette, 0.163) rediscovered the calendar without labels: weekend, summer-weekday and
winter-weekday clusters. Association rules (Apriori): {Afternoon, Weekday} → High traffic
at confidence 0.716, lift 2.86.

**Advanced analysis** (`part3_machine_learning/notebooks/lstm_shap_analysis.ipynb`) —
LSTM next-hour forecaster (MAE 389.4, beats persistence 588.9, loses to RF 241.2 — feature
engineering encodes domain knowledge that a small LSTM must relearn) and SHAP
explainability confirming calendar features dominate congestion prediction.

**MLOps & deployment**
- `mlflow/track_experiments.py` — MLflow tracking: 3 runs with params, metrics, artifacts (SQLite backend)
- `deployment/app.py` + `test_api.py` — FastAPI mock service, 5 passing smoke tests; interactive docs at `/docs`
- `monitoring/drift_check.py` — PSI data-drift + performance monitoring with PASS/WARN/ALERT verdicts and a documented retraining policy
- `recommendation_system/recommender.py` — CLI advisor for commuters and traffic operations

**Reports** — see `part3_machine_learning/reports/part3_ml_report.pdf` (full Part 3 results)
and `part3_machine_learning/reports/responsible_ai_report.pdf`
(proxy-label limitations, fairness, transparency via SHAP, monitoring-based accountability).

> Trained model binaries (`*.joblib`) are gitignored — regenerate them with
> `python part3_machine_learning/supervised/train_supervised.py`.

## License

MIT — see [LICENSE](LICENSE).