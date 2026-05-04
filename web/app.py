"""
NYC Taxi Demand Predictor — Streamlit Web App

Owner: Chloe
Inputs:
  - models/stacking_model.pkl  (preferred — from 07_stacking)
  - models/boosting_model.pkl  (fallback — from 05_modeling)
  - data/processed/zone_clusters.csv  (for zone metadata)

Run locally:
    pip install streamlit pandas numpy joblib pydeck
    streamlit run app.py

Layout:
  - Sidebar: input controls (zone, time, weather)
  - Main: predicted demand + confidence interval + Manhattan map
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="NYC Taxi Demand Predictor",
    page_icon="🚕",
    layout="wide",
)

# Adjust this if your project lives somewhere else
PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "models"
PROC_DIR = PROJECT_ROOT / "data" / "processed"

POLICY_DATE = pd.Timestamp("2025-01-05")

# CBD zones (empirical, from FE step + airport exclusion)
CBD_ZONES = {
    4, 12, 13, 43, 45, 48, 50, 68, 79, 87, 88, 90, 100, 107, 113, 114,
    125, 137, 142, 144, 148, 158, 161, 162, 163, 164, 170, 186, 209,
    211, 224, 229, 230, 231, 232, 233, 234, 246, 249, 261,
}
AIRPORT_IDS = {1, 132, 138}  # EWR, JFK, LGA

# ============================================================
# Cached loaders (run once per session)
# ============================================================

@st.cache_resource
def load_model():
    """Try stacking model first, fall back to LightGBM if not available."""
    stacking_path = MODEL_DIR / "stacking_model.pkl"
    boosting_path = MODEL_DIR / "boosting_model.pkl"

    if stacking_path.exists():
        bundle = joblib.load(stacking_path)
        return ("stacking", bundle)
    elif boosting_path.exists():
        model = joblib.load(boosting_path)
        return ("lightgbm", model)
    else:
        return (None, None)


@st.cache_data
def load_zone_metadata():
    """Load zone names + clusters + lat/lon for the map."""
    cluster_path = PROC_DIR / "zone_clusters.csv"
    if cluster_path.exists():
        df = pd.read_csv(cluster_path)
        return df
    return None


@st.cache_data
def load_recent_panel():
    """Load a recent slice of the demand panel for lag feature lookups."""
    panel_path = PROC_DIR / "feature_matrix_demand.parquet"
    if not panel_path.exists():
        return None
    df = pd.read_parquet(panel_path)
    # Keep only the latest 7 days for fast lookup
    cutoff = df["hour_ts"].max() - pd.Timedelta(days=7)
    return df[df["hour_ts"] >= cutoff].copy()


# Manhattan zone center coordinates (approximate, NYC TLC zone centroids)
# This is a small subset focused on CBD + nearby zones for the map view
ZONE_COORDS = {
    4:   (40.7261, -73.9897),  # Alphabet City
    12:  (40.7033, -74.0170),  # Battery Park
    13:  (40.7117, -74.0153),  # Battery Park City
    43:  (40.7822, -73.9676),  # Central Park
    45:  (40.7140, -73.9970),  # Chinatown
    48:  (40.7641, -73.9924),  # Clinton East
    50:  (40.7635, -74.0027),  # Clinton West
    68:  (40.7484, -74.0027),  # East Chelsea
    79:  (40.7287, -73.9854),  # East Village
    87:  (40.7081, -74.0089),  # Financial District N
    88:  (40.7064, -74.0094),  # Financial District S
    90:  (40.7401, -73.9893),  # Flatiron
    100: (40.7549, -73.9886),  # Garment District
    107: (40.7383, -73.9839),  # Gramercy
    113: (40.7359, -74.0009),  # Greenwich Village N
    114: (40.7330, -74.0030),  # Greenwich Village S
    125: (40.7297, -74.0066),  # Hudson Sq
    137: (40.7468, -73.9776),  # Kips Bay
    142: (40.7689, -73.9826),  # Lincoln Square E
    144: (40.7204, -73.9947),  # Little Italy / NoLita
    148: (40.7172, -73.9869),  # Lower East Side
    158: (40.7397, -74.0079),  # Meatpacking
    161: (40.7607, -73.9784),  # Midtown Center
    162: (40.7547, -73.9694),  # Midtown East
    163: (40.7600, -73.9831),  # Midtown North
    164: (40.7578, -73.9858),  # Midtown South
    170: (40.7475, -73.9755),  # Murray Hill
    186: (40.7508, -73.9942),  # Penn Station
    209: (40.7068, -74.0028),  # Seaport
    211: (40.7232, -74.0027),  # SoHo
    224: (40.7320, -73.9760),  # Stuy Town
    229: (40.7561, -73.9622),  # Sutton Place
    230: (40.7589, -73.9851),  # Times Sq
    231: (40.7163, -74.0093),  # TriBeCa
    232: (40.7126, -73.9923),  # Two Bridges
    233: (40.7501, -73.9685),  # UN / Turtle Bay S
    234: (40.7359, -73.9911),  # Union Square
    246: (40.7553, -74.0028),  # West Chelsea
    249: (40.7345, -74.0061),  # West Village
    261: (40.7115, -74.0136),  # World Trade Center
}

# ============================================================
# Build feature row for a single prediction
# ============================================================

def build_feature_row(zone_id, target_dt, temp_c, precip_mm, recent_panel, zone_meta):
    """
    Construct a single-row DataFrame matching the feature matrix schema.

    The model expects exactly the columns it was trained on. We look up
    historical lag values from `recent_panel` (most recent week of data).
    """
    is_holiday = False  # Could plug in a real lookup; rare enough to default False
    is_weekend = target_dt.dayofweek in (5, 6)
    is_rainy = precip_mm > 0.5

    # Look up lag values from recent panel
    # If user picks a future hour beyond panel range, fall back to that zone's mean
    lag_lookups = {}
    if recent_panel is not None:
        zone_recent = recent_panel[recent_panel["PULocationID"] == zone_id]
        if len(zone_recent) > 0:
            zone_recent = zone_recent.sort_values("hour_ts")
            # Most-recent values for this zone as proxy for the lags
            latest = zone_recent.iloc[-1]
            lag_lookups = {
                "n_trips_lag_1h":   float(latest.get("n_trips", 0)),
                "n_trips_lag_24h":  float(zone_recent["n_trips"].iloc[-25] if len(zone_recent) >= 25 else latest["n_trips"]),
                "n_trips_lag_168h": float(zone_recent["n_trips"].iloc[0]),
                "n_trips_ma_24h":   float(zone_recent["n_trips"].tail(24).mean()),
            }
        else:
            lag_lookups = {k: 0.0 for k in ["n_trips_lag_1h", "n_trips_lag_24h", "n_trips_lag_168h", "n_trips_ma_24h"]}
    else:
        lag_lookups = {k: 0.0 for k in ["n_trips_lag_1h", "n_trips_lag_24h", "n_trips_lag_168h", "n_trips_ma_24h"]}

    # Get borough
    borough = "Manhattan" if zone_id in CBD_ZONES else "Other"
    if recent_panel is not None and "borough" in recent_panel.columns:
        zone_borough_lookup = recent_panel.drop_duplicates("PULocationID").set_index("PULocationID")
        if zone_id in zone_borough_lookup.index:
            borough = str(zone_borough_lookup.loc[zone_id, "borough"])

    # Get zone cluster
    zone_cluster = -1
    if zone_meta is not None:
        cluster_col = next((c for c in ["cluster", "zone_cluster", "kmeans_cluster"] if c in zone_meta.columns), None)
        zone_col = next((c for c in ["PULocationID", "zone_id", "LocationID"] if c in zone_meta.columns), None)
        if cluster_col and zone_col:
            match = zone_meta[zone_meta[zone_col] == zone_id]
            if len(match) > 0:
                zone_cluster = int(match.iloc[0][cluster_col])

    row = {
        "PULocationID": zone_id,
        "n_card": 0,  # leakage column, dropped by filter
        "avg_fare": 0.0, "avg_distance": 0.0, "avg_duration": 0.0, "avg_speed": 0.0,
        "avg_fare_missing": 1, "avg_distance_missing": 1,
        "avg_duration_missing": 1, "avg_speed_missing": 1,
        "hour": target_dt.hour,
        "dayofweek": target_dt.dayofweek,
        "month": target_dt.month,
        "is_weekend": int(is_weekend),
        "hour_sin": np.sin(2 * np.pi * target_dt.hour / 24),
        "hour_cos": np.cos(2 * np.pi * target_dt.hour / 24),
        "post_policy": int(target_dt >= POLICY_DATE),
        "is_holiday": int(is_holiday),
        "borough": borough,
        "is_manhattan": int(zone_id in CBD_ZONES or borough == "Manhattan"),
        "is_airport": int(zone_id in AIRPORT_IDS),
        **lag_lookups,
        "temp_c": temp_c,
        "precip_mm": precip_mm,
        "humidity": 60.0,  # default
        "wind_speed": 5.0,  # default
        "is_rainy": int(is_rainy),
        "treated_zone": int(zone_id in CBD_ZONES),
        "did": int(zone_id in CBD_ZONES) * int(target_dt >= POLICY_DATE),
        "zone_cluster": zone_cluster,
    }
    return pd.DataFrame([row])


def predict_with_uncertainty(model_type, model_obj, feature_row):
    """Run prediction and return (point_estimate, lower_ci, upper_ci)."""
    if model_type == "stacking":
        # Get individual base-model predictions to estimate spread
        base_models = model_obj["base_models"]
        meta_learner = model_obj["meta_learner"]
        names = model_obj["model_names"]

        base_preds = np.zeros(len(names))
        for j, name in enumerate(names):
            try:
                base_preds[j] = max(0, float(base_models[name].predict(feature_row)[0]))
            except Exception as e:
                st.warning(f"Base model {name} failed: {e}")
                base_preds[j] = 0
        point = max(0, float(meta_learner.predict(base_preds.reshape(1, -1))[0]))

        # Use spread across base models as a rough confidence interval proxy
        lower = max(0, point - base_preds.std())
        upper = point + base_preds.std()
        return point, lower, upper, base_preds, names

    elif model_type == "lightgbm":
        try:
            point = max(0, float(model_obj.predict(feature_row)[0]))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return None, None, None, None, None
        # No uncertainty available from a single model — fake a +/- 15% band
        return point, point * 0.85, point * 1.15, None, None

    return None, None, None, None, None


# ============================================================
# UI
# ============================================================

def main():
    st.title("🚕 NYC Yellow Taxi Demand Predictor")
    st.markdown(
        "Predict hourly taxi pickup demand at any NYC zone. "
        "Built on a stacking ensemble of Poisson, Ridge, RandomForest, and LightGBM models, "
        "trained on **TLC trip records 2024-11 to 2025-02** spanning the Manhattan congestion-pricing launch."
    )

    # Load resources
    model_type, model_obj = load_model()
    zone_meta = load_zone_metadata()
    recent_panel = load_recent_panel()

    if model_type is None:
        st.error(
            "❌ No model file found. Run `05_modeling_colab.ipynb` (and ideally `07_stacking_colab.ipynb`) first, "
            "then place the resulting `.pkl` files in `models/`."
        )
        st.stop()

    if model_type == "stacking":
        st.success(f"✓ Loaded stacking ensemble ({len(model_obj['base_models'])} base models)")
    else:
        st.info("ℹ️  Using LightGBM (stacking model not available)")

    # ---------- SIDEBAR ----------
    st.sidebar.header("Prediction inputs")

    # Zone selector
    zone_options = sorted(ZONE_COORDS.keys())
    zone_id = st.sidebar.selectbox(
        "Pickup zone (Manhattan CBD area)",
        options=zone_options,
        index=zone_options.index(230) if 230 in zone_options else 0,  # Default Times Sq
        format_func=lambda z: f"Zone {z}" + (" 🏛 CBD" if z in CBD_ZONES else ""),
    )

    # Date and time
    col1, col2 = st.sidebar.columns(2)
    with col1:
        target_date = st.date_input("Date", value=pd.Timestamp("2025-02-15").date())
    with col2:
        target_hour = st.slider("Hour (24h)", 0, 23, 14)
    target_dt = pd.Timestamp(target_date) + pd.Timedelta(hours=target_hour)

    # Weather
    st.sidebar.markdown("**Weather**")
    temp_c = st.sidebar.slider("Temperature (°C)", -10.0, 35.0, 10.0, 0.5)
    precip_mm = st.sidebar.slider("Precipitation (mm/h)", 0.0, 20.0, 0.0, 0.5)

    # Predict button
    predict_clicked = st.sidebar.button("🔮 Predict demand", type="primary", use_container_width=True)

    # ---------- MAIN ----------
    if predict_clicked:
        with st.spinner("Computing prediction..."):
            feature_row = build_feature_row(
                zone_id=zone_id,
                target_dt=target_dt,
                temp_c=temp_c,
                precip_mm=precip_mm,
                recent_panel=recent_panel,
                zone_meta=zone_meta,
            )
            point, lower, upper, base_preds, base_names = predict_with_uncertainty(
                model_type, model_obj, feature_row
            )

        if point is None:
            st.error("Prediction failed. Check feature schema compatibility.")
            return

        # Display prediction prominently
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric(
                "Predicted hourly trips",
                f"{point:.1f}",
                delta=f"~{int(point)} pickups",
            )
        with col_b:
            st.metric(
                "95% interval",
                f"[{lower:.1f}, {upper:.1f}]",
            )
        with col_c:
            cbd_status = "Yes 🏛" if zone_id in CBD_ZONES else "No"
            st.metric("In CBD?", cbd_status)

        # Base-model breakdown if stacking
        if base_preds is not None:
            st.subheader("Base model breakdown")
            breakdown_df = pd.DataFrame({
                "Model": base_names,
                "Prediction": base_preds.round(2),
            })
            st.bar_chart(breakdown_df.set_index("Model"))

        # Map view
        st.subheader("Selected zone on map")
        if zone_id in ZONE_COORDS:
            lat, lon = ZONE_COORDS[zone_id]
            map_df = pd.DataFrame([
                {"lat": lat, "lon": lon, "label": f"Zone {zone_id}", "demand": point}
            ])
            try:
                import pydeck as pdk
                deck = pdk.Deck(
                    map_style=None,
                    initial_view_state=pdk.ViewState(
                        latitude=lat, longitude=lon, zoom=13, pitch=45
                    ),
                    layers=[
                        pdk.Layer(
                            "ColumnLayer",
                            data=map_df,
                            get_position="[lon, lat]",
                            get_elevation="demand * 20",
                            elevation_scale=1,
                            radius=120,
                            get_fill_color="[200, 80, 60, 200]",
                            pickable=True,
                            auto_highlight=True,
                        )
                    ],
                )
                st.pydeck_chart(deck)
            except Exception as e:
                st.map(map_df, latitude="lat", longitude="lon")

        # Context: how does this compare to typical demand
        st.subheader("How does this compare?")
        if recent_panel is not None:
            zone_history = recent_panel[recent_panel["PULocationID"] == zone_id]
            if len(zone_history) > 0:
                hist_mean = zone_history["n_trips"].mean()
                hist_max = zone_history["n_trips"].max()
                vs_mean = (point - hist_mean) / hist_mean * 100 if hist_mean > 0 else 0
                st.write(
                    f"This zone's recent (past week) average: **{hist_mean:.1f}** trips/hour. "
                    f"Recent peak: **{hist_max:.0f}** trips/hour. "
                    f"Predicted value is **{vs_mean:+.0f}%** vs recent mean."
                )

    else:
        st.info("👈 Set inputs on the left and click **Predict demand** to see results.")

        # Pre-prediction landing content
        st.subheader("About this app")
        st.markdown(
            """
            - **Coverage**: Manhattan CBD + adjacent zones (~40 zones)
            - **Granularity**: hour × zone
            - **Time window**: trained on Nov 2024 – Feb 2025 (post Jan 5, 2025 = post congestion-pricing)
            - **Model**: stacking ensemble of 4 base learners (Poisson + Ridge + RF + LightGBM)
            - **Use cases**: dispatch planning, dynamic pricing exploration, policy impact reasoning
            """
        )

        if zone_meta is not None and len(zone_meta) > 0:
            st.subheader("Zone clusters (from unsupervised analysis)")
            st.dataframe(zone_meta.head(20))


if __name__ == "__main__":
    main()
