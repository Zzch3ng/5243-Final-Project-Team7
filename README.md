# STAT 5243 Final Project Team 7

## Project Title

Multi-Dimensional Effects of NYC Congestion Pricing: Taxi Demand, Transportation Patterns, and Urban Mobility

## Overview

This project studies NYC Yellow Taxi demand before and after the Manhattan CBD congestion pricing policy started on January 5, 2025. The final workflow uses taxi trip records from November 2024 to February 2025, taxi zone data, hourly weather data, holiday indicators, policy variables, unsupervised zone features, supervised demand models, stacking, DID analysis, and a Streamlit web app.

The final reproducible notebook is:

```text
5243_final_project.ipynb
```

Run this notebook from top to bottom to reproduce the full project.

## Repository Structure

```text
.
├── 5243_final_project.ipynb
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── outputs/
│   ├── eda/
│   ├── fe/
│   ├── modeling/
│   ├── stacking/
│   └── did/
├── src/
│   └── 01_data_pipeline.py
└── web/
    ├── app.py
    ├── README.md
    └── requirements.txt
```

## Setup

Run these commands from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then launch Jupyter from the project root:

```bash
jupyter notebook
```

Open and run:

```text
5243_final_project.ipynb
```

Important: start Jupyter from the project root folder. If Jupyter is launched from another directory, the notebook may not find `data/raw` or `data/processed`.

## Required Raw Data

Place the raw input files in:

```text
data/raw/
```

Required files:

```text
yellow_tripdata_2024-11.parquet
yellow_tripdata_2024-12.parquet
yellow_tripdata_2025-01.parquet
yellow_tripdata_2025-02.parquet
taxi_zone_lookup.csv
weather_hourly.csv
us_holidays.csv
```

## Main Outputs

The final notebook regenerates these outputs:

```text
data/processed/
  clean_trips.parquet
  cleaning_log.csv
  data_quality_report.md
  feature_matrix_demand.parquet
  feature_matrix_tip.parquet
  zone_clusters.csv

models/
  poisson_model.pkl
  ridge_model.pkl
  random_forest_model.pkl
  boosting_model.pkl
  final_model.pkl
  stacking_model.pkl
  encoders.pkl

outputs/eda/
outputs/fe/
outputs/modeling/
outputs/stacking/
outputs/did/
```

Generated processed data, models, and output plots are ignored by git because some files are too large for GitHub. Recreate them by running `5243_final_project.ipynb`.

## Direct Results Download

If you want to directly view the generated results without rerunning the full pipeline, download the processed outputs and model artifacts from Google Drive:

https://drive.google.com/drive/folders/1gx2zPE-AV8BgAMtIa6xlnyTtvpk4T6eU?usp=drive_link

After downloading, place the files back into the matching folders, such as `data/processed/`, `models/`, and `outputs/`.

## Run Order

Recommended full reproduction:

1. Install dependencies using `requirements.txt`.
2. Make sure all raw files are in `data/raw/`.
3. Launch Jupyter from the project root.
4. Run `5243_final_project.ipynb` from top to bottom.
5. Check generated outputs in `data/processed/`, `models/`, and `outputs/`.

## Web App

After the notebook has generated the required model and feature files, run the Streamlit app from the project root:

```bash
python -m streamlit run "web/app.py"
```

The app uses:

```text
models/stacking_model.pkl
models/boosting_model.pkl
data/processed/feature_matrix_demand.parquet
data/processed/zone_clusters.csv
```

If `stacking_model.pkl` exists, the app uses the stacking model. Otherwise, it falls back to the LightGBM boosting model.

## Notes

The main modeling task predicts zone-hour taxi demand. The supervised models include Poisson Regression, Ridge Regression, Random Forest, LightGBM, and a stacking ensemble. Evaluation outputs are saved under `outputs/modeling/` and `outputs/stacking/`.

The DID analysis outputs are saved under `outputs/did/`. EDA figures are saved under `outputs/eda/`, and feature engineering / unsupervised learning figures are saved under `outputs/fe/`.
