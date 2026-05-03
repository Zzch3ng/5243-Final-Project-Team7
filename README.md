# STAT 5243 NYC Taxi Congestion Pricing Project

This repository is configured so it can be cloned or downloaded from GitHub and rerun with relative paths. The notebooks automatically find the project root by walking up from the current working directory until they find the project `data/` or `src/` folder.

## Setup

Run these commands from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter notebook
```

## Required Data Layout

Place the raw input files here:

```text
data/raw/
  yellow_tripdata_2024-11.parquet
  yellow_tripdata_2024-12.parquet
  yellow_tripdata_2025-01.parquet
  yellow_tripdata_2025-02.parquet
  taxi_zone_lookup.csv
  weather_hourly.csv
  us_holidays.csv
```

The main pipeline writes regenerated outputs to:

```text
data/processed/
models/
outputs/modeling/
output/figures/
```

Processed data and model outputs are intentionally ignored by git because some files are too large for GitHub. Recreate them by running the notebooks.

## Recommended Run Order

For the full project, run:

```text
5243 final project1.ipynb
```

For modular execution, run:

```text
output/jupyter-notebook/nyc-taxi-data-cleaning-pipeline.ipynb
output/jupyter-notebook/02_eda.ipynb
output/jupyter-notebook/03_fe_unsupervised_chloe.ipynb
output/jupyter-notebook/05_modeling.ipynb
```

The modeling notebook expects:

```text
data/processed/feature_matrix_demand.parquet
data/processed/zone_clusters.csv
```

These are produced by the feature engineering notebook.
