# Deploy to shinyapps.io

This folder is a Shiny for Python version of the original Streamlit app.

## Folder structure

Keep this structure before deploying:

```text
project/
  app.py
  requirements.txt
  models/
    stacking_model.pkl      # preferred
    boosting_model.pkl      # fallback
  data/
    processed/
      zone_clusters.csv                  # optional
      feature_matrix_demand.parquet      # optional, for lag features
```

## Local test

```bash
pip install -r requirements.txt
shiny run --reload app.py
```

## Deploy

```bash
rsconnect add --account YOUR_ACCOUNT --name YOUR_NAME --token YOUR_TOKEN --secret YOUR_SECRET
rsconnect deploy shiny . --name YOUR_ACCOUNT --title nyc-taxi-demand-predictor
```

If the deploy command complains about a Python version, set Python to 3.11 or 3.12 locally, reinstall `requirements.txt`, then deploy again.


## Dark theme

This version forces a black/dark UI through custom CSS and uses dark Plotly backgrounds for the bar chart and map.
