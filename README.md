# Donor Segmentation & Churn Risk Analysis

A Flask web application that performs **RFM segmentation**, **churn risk prediction**, and **PDF report generation** on donation data. A simplified, publicly shareable version of a donor analytics project originally built for an organization — the institution-specific strategy deliverable is not included; only the analysis and reporting pipeline.

> 🔁 Multilingual UI (Turkish / English) · 📊 Auto-generated PDF report · 🤖 Pre-trained ensemble model · 🎲 No upload required — synthetic data is generated on the fly

🔗 **Live demo:** [https://donor-analytics.emrebayir.com](https://donor-analytics.emrebayir.com)

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [Usage](#usage)
- [Churn Prediction Model](#churn-prediction-model)
- [RFM Segmentation](#rfm-segmentation)
- [Reports](#reports)
- [Notes & Disclaimers](#notes--disclaimers)
- [License](#license)

---

## Features

- **One-click analysis pipeline** — generates synthetic donor data, runs consolidation, churn prediction, and RFM analysis, then builds a downloadable PDF + CSV.
- **Synthetic data generator** — vectorized (NumPy/Pandas) generator producing realistic Turkish donation records (names, cities, campaigns, channels, currencies, log-normal amounts) with reproducible seeds.
- **Pre-trained churn ensemble** — LightGBM + XGBoost + CatBoost with Optuna-optimized ensemble weights, loaded from `utils/models/`.
- **RFM segmentation** — Recency / Frequency / Monetary scoring with two-period (previous vs. current quarter) mobility analysis and value/behavior/final segment classification.
- **Auto-generated bilingual PDF report** (TR / EN) — cover page, general stats, distribution charts, churn analysis, RFM segments, Pareto revenue, segment mobility & transition charts, second-donation performance, and violin time-to-second-donation.
- **Live progress tracking** — long-running analysis runs in a background thread; the UI polls a `status.json` file and renders a step-by-step progress log.
- **Configurable** — donor count, donation count, random seed (for reproducibility), cutoff date, output filename, and institution name.

---

## How It Works

When you click **Start Analysis**, the app runs a 6-step pipeline in a background subprocess:

1. **Generate demo data** — synthetic `donations` + `donors` tables (`utils/demo_data_generator.py`).
2. **Consolidate** — per-donor aggregation of numeric, categorical, and date features (`utils/consolidator.py`).
3. **Churn risk prediction** — eligible donors (≥2 donations in the lookback window) are scored by the ensemble model (`utils/churn_risk_predictor.py`).
4. **RFM analysis** — two sliding windows (prev/next) for 24 months with a 3-month shift (`utils/rfm_analyzer.py`).
5. **Report generation** — PDF with embedded charts via `fpdf2` + `matplotlib`/`seaborn`/`geopandas` (`utils/report_generator.py`).
6. **Final dataset export** — combined CSV (consolidated + churn + RFM + donor names) with `;`-separator and `,`-decimal for Excel-TR compatibility.

The merged `final_data` is the single source for both the PDF report and the CSV download.

---

## Tech Stack

| Area                  | Libraries                                                 |
| --------------------- | --------------------------------------------------------- |
| Web framework         | Flask                                                     |
| Data                  | pandas, numpy, pyarrow                                    |
| ML models             | LightGBM, XGBoost, CatBoost, scikit-learn                 |
| Hyperparameter tuning | Optuna                                                    |
| Reporting             | fpdf2, matplotlib, seaborn, geopandas                     |
| Fonts                 | DejaVu (bundled in `utils/fonts/` for Turkish characters) |

Full versioned list in `requirements.txt`.

---

## Project Structure

```
analiz_portfolio/
├── app.py                          # Flask app + UI + pipeline orchestration
├── requirements.txt
├── Dockerfile                      # Container build definition
├── notebooks/
│   └── train_model.ipynb           # Notebook used to train the ensemble model
└── utils/
    ├── __init__.py
    ├── demo_data_generator.py      # Synthetic donations/donors generator
    ├── consolidator.py             # Per-donor aggregation
    ├── training_data_generator.py  # Time-sliced training set + churn labels
    ├── churn_risk_predictor.py     # Feature engineering + ensemble inference
    ├── rfm_analyzer.py             # RFM scoring + segment rules
    ├── report_generator.py         # PDF builder + chart functions
    ├── assets/
    │   ├── logo.png
    │   └── tr_cities.json          # Türkiye GeoJSON for province maps
    ├── fonts/                      # DejaVu TTF (for Turkish characters in PDF)
    └── models/
        ├── preprocessor.joblib
        ├── lightgbm.joblib
        ├── xgboost.joblib
        ├── catboost.joblib
        ├── weights.json            # Ensemble weights (Optuna-optimized)
        ├── feature_names.json      # Ordered feature list for inference
        ├── churn_prediction_model.joblib
        └── train_model.py          # Full training pipeline (Optuna + 5-fold CV)
```

---

## Getting Started

### Prerequisites

- Python 3.10+ (Docker image uses 3.12-slim)
- ~2 GB free disk for `pip install` (XGBoost, CatBoost, GeoPandas are heavy)

There are two ways to run the app: directly on your machine (**Local Setup**), or inside a container (**Docker Setup**). Pick one.

### Local Setup

```bash
# 1. Clone
git clone https://github.com/emrebayir1/donor-churn-rfm.git
cd donor-churn-rfm

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python app.py
```

The app starts on **[http://0.0.0.0:8080](http://0.0.0.0:8080)** (Flask dev server, bound to all interfaces) and your browser opens automatically after ~1.5 seconds. Open it locally at `http://127.0.0.1:8080` or `http://localhost:8080`.

### Docker Setup

The container runs the app with **Gunicorn**, also bound to `0.0.0.0:8080` so it's reachable from outside the container. It's designed to sit behind a reverse proxy (e.g. Nginx) rather than being browsed to directly on port 8080.

**1. `Dockerfile`** (included in the repo root):

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn==23.0.0
COPY . .
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "app:app"]
```

**2. `docker-compose.yml`** (add as a service, e.g. alongside other apps on a shared network):

```yaml
  # DONOR ANALYTICS
  donor-analytics:
    build: ./donor-churn-rfm
    container_name: donor-analytics
    expose:
      - "8080"
    restart: unless-stopped
    networks:
      - <yournetwork>
```

> Note: `expose` only publishes the port to other containers on the same Docker network — not to the host. Use `ports: ["8080:8080"]` instead if you want to reach the container directly from your host without a reverse proxy.

**3. Reverse proxy (Nginx)** — example config used to serve the app over HTTPS at the live domain:

```nginx
# ─── donor-analytics.<yourdomain> ───
server {
    listen 443 ssl;
    http2 on;
    server_name donor-analytics.<yourdomain>;
    ssl_certificate /etc/nginx/certs/archive/<yourdomain>/fullchain1.pem;
    ssl_certificate_key /etc/nginx/certs/archive/<yourdomain>/privkey1.pem;
    location / {
        proxy_pass http://donor-analytics:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**4. Build and run:**

```bash
docker compose up -d --build donor-analytics
```

---

## Usage

1. Open the app in your browser.
2. (Optional) Toggle **Demo Data** parameters:- **Donor count** (10–10,000)- **Donation count** (1,000–100,000)- **Reuse the same data** checkbox + **seed** number for reproducibility
3. (Optional) Set **Parameters**:- **Cutoff date** — reference end date for churn analysis (empty = last day of the month preceding the latest donation)- **PDF output filename**
4. Click **Start Analysis**.
5. Watch the live step-by-step progress log.
6. When finished, download the **PDF report** and/or **CSV dataset**.

Switch the UI language with the **TR / EN** links in the top-right corner. The selected language also drives the language of the generated PDF.

---

## Churn Prediction Model

The churn model is a **weighted ensemble** of three gradient-boosted classifiers.

### Methodology

- **Cutoff logic (time-rewinding)** — the model selects a past month-end as a "decision moment" and only looks at donor history up to that point.
- **Eligibility** — donors with at least **2 donations in the 365 days** before the cutoff are considered eligible. The focus is protecting the active donor base.
- **Labeling** — a forward window (default 180–365 days) is observed: if the donor gives again, they are "retained"; otherwise "churned".
- **Monthly checkpoints** — each month-end is a separate cutoff, so the same donor may appear multiple times in the training set across different months, giving the model more learning examples.

### Training pipeline (`utils/models/train_model.py`)

1. Generate training data via `training_data_generator.py`.
2. Feature engineering — diversity ratios (`Tür_orani`, `Kanal_orani`, `Kampanya_orani`), spread metrics, frequency + one-hot encoding.
3. **Time-based train/test split** (last 6 months as test, with optional embargo).
4. `ColumnTransformer` with `OneHotEncoder(handle_unknown="ignore")` for robustness.
5. **5-fold StratifiedKFold** + Optuna (50 trials per model, optimized on Average Precision).
6. **Optuna ensemble weight optimization** (1,000 trials on OOF predictions, AP metric).
7. Export preprocessor, individual models, weights, and feature names as `joblib`/`json`.

At inference, `churn_risk_predictor.py` automatically aligns features to `feature_names.json` (missing columns → 0), transforms via the saved preprocessor, and produces a weighted sum of base model probabilities. Output is formatted as `"%XX.XX"`.

---

## RFM Segmentation

Scoring follows the classic RFM approach:

- **R — Recency**: days since last donation (5 bins, reverse — recent = high score).
- **F — Frequency**: binary (1 = single donation, 2 = repeat donor).
- **M — Monetary**: 5 bins by total donation amount.

### Segments

**Value segments** (from M score): `VIP`, `High`, `Medium`, `Standard`.

**Behavioral segments** (from R + F):

| Segment (TR)        | Segment (EN)        | R   | F   |
| ------------------- | ------------------- | --- | --- |
| Şampiyonlar         | Champions           | 5   | 2   |
| Sadık Bağışçılar    | Loyal Donors        | 4   | 2   |
| Potansiyel Sadıklar | Potential Loyalists | 3   | 2   |
| Yeni Bağışçılar     | New Donors          | 4–5 | 1   |
| İlgi Gerektiren     | Need Attention      | 3   | 1   |
| Uykuda              | Dormant             | 2   | 1–2 |
| Kayıp Riskli        | At Risk             | 1   | 1–2 |

**Final segment** — combines value + behavior (e.g., `VIP Champions`, `Standard At Risk`).

Two sliding windows are computed by default:

- `prev`: `[ref-27m, ref-3m]`
- `next`: `[ref-24m, ref]`

This enables **segment mobility** (improving / stable / declining) and **per-segment transition** analyses shown in the report.

> ⚠️ RFM is sensitive to the selected time window. The same donor can land in different segments under a different date range. The original project included a separate strategy for handling segment transitions; this public version exposes only the analysis.

---

## Reports

The generated PDF (A4, portrait) contains:

1. **Cover page** — logo + report title.
2. **General Information** — date range, donor/donation counts, totals, averages, medians, max, repeat-donor rate, one-time donors.
3. **General Statistics** — donut charts for donation type, channel, payment method, currency; bar chart of top campaigns; Türkiye choropleth maps for total and average donations by province.
4. **Churn Analysis** — eligible vs. not-eligible bar, top donors with the highest churn risk.
5. **RFM Segment Analysis** — behavioral & value segment distributions, average donation per segment, Pareto revenue per segment, segment mobility bar, and transition charts for `Champions`, `Loyal Donors`, `Potential Loyalists`, `Dormant`, `At Risk`.
6. **Second-donation performance** — total first donors, repeat donors, repeat-within-60-days; violin plot of days from 1st to 2nd donation.
7. **Information & Disclaimer**.

All charts are generated via `matplotlib`/`seaborn`/`geopandas`, written to temporary PNGs, embedded into the PDF, and removed afterward.

---

## Notes & Disclaimers

- Demo data is **fully synthetic** and contains no real personal data.

- The bundled churn models were trained on synthetic data — they are intended for demonstration of the pipeline, not for production decisions on real donors.

- Turbine outputs (PDFs and CSVs) are excluded from git via `.gitignore`.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this project in your work, attribution is appreciated.
Created by **Emre Bayır**.