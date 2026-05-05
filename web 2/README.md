

# NYC Taxi Demand Predictor Web App

This folder contains the Streamlit web app for the STAT 5243 NYC Taxi congestion pricing project. The app predicts hourly yellow taxi pickup demand for selected NYC taxi zones using the trained machine learning models from the project.

## What The App Uses

The app automatically searches upward from this folder to find the project root. The project root must contain:

```text
models/
  stacking_model.pkl      # preferred model, if available
  boosting_model.pkl      # LightGBM fallback model

data/processed/
  feature_matrix_demand.parquet
  zone_clusters.csv
```

If `stacking_model.pkl` exists, the app uses the stacking ensemble. If not, it falls back to `boosting_model.pkl`.

## Install Dependencies

From the project root, run:

```bash
python -m pip install -r requirements.txt
```

If you only want to install the web app dependencies from inside this folder, run:

```bash
python -m pip install -r requirements.txt
```

## Run The App

From the project root:

```bash
python -m streamlit run "web 2/app.py"
```

Or from inside the `web 2/` folder:

```bash
python -m streamlit run app.py
```

After running the command, Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Open that URL in a browser to use the app.

## How To Use

1. Choose a pickup zone from the sidebar.
2. Select a date and hour.
3. Set weather inputs such as temperature and precipitation.
4. Click **Predict demand**.
5. Review the predicted hourly trips, interval estimate, model breakdown, and map view.

## Reproducing From GitHub

A full reproducible workflow is:

```bash
git clone <repo-url>
cd <repo-folder>
python -m pip install -r requirements.txt
jupyter notebook "5243_final_project.ipynb"
python -m streamlit run "web 2/app.py"
```

Run the notebook first if the `models/` or `data/processed/` files are missing. The notebook generates the cleaned data, feature matrix, trained models, and output files.

If the repository already includes trained models and processed data, you can run the web app directly after installing dependencies.

## Common Issues

### `ModuleNotFoundError: No module named 'streamlit'`

Install dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

### `No model file found`

The app could not find:

```text
models/stacking_model.pkl
models/boosting_model.pkl
```

Run the modeling section of the notebook first, or place the trained model files in the project root `models/` folder.

### `Recent panel data was not found`

The app could not find:

```text
data/processed/feature_matrix_demand.parquet
```

Run the preprocessing and feature engineering sections of the notebook first.

## Notes

This app is for project demonstration, not production deployment. Predictions depend heavily on lag features from the most recent available zone-hour panel, so results are most meaningful for dates close to the project time window.
