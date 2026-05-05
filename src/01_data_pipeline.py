from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MONTH_FILES = {
    "2024-11": "yellow_tripdata_2024-11.parquet",
    "2024-12": "yellow_tripdata_2024-12.parquet",
    "2025-01": "yellow_tripdata_2025-01.parquet",
    "2025-02": "yellow_tripdata_2025-02.parquet",
}

RAW_FILES = {
    "zones": "taxi_zone_lookup.csv",
    "weather": "weather_hourly.csv",
    "holidays": "us_holidays.csv",
}

ALLOWED_PAYMENT_TYPES = {1, 2, 3, 4}
PROJECT_START = pd.Timestamp("2024-11-01")
PROJECT_END = pd.Timestamp("2025-03-01")
POLICY_START = pd.Timestamp("2025-01-05").date()


def print_progress(message: str) -> None:
    print(f"[data-pipeline] {message}")


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"Missing required input files:\n{missing_text}")


def append_log(log_rows: list[dict[str, float]], step: str, before: int, after: int) -> None:
    dropped = before - after
    percent_dropped = (dropped / before * 100.0) if before else 0.0
    log_rows.append(
        {
            "step_name": step,
            "rows_before": before,
            "rows_after": after,
            "rows_dropped": dropped,
            "percent_dropped": round(percent_dropped, 4),
        }
    )


def filter_rows(
    df: pd.DataFrame,
    drop_mask: pd.Series,
    step: str,
    log_rows: list[dict[str, float]],
) -> pd.DataFrame:
    before = len(df)
    cleaned = df.loc[~drop_mask].copy()
    append_log(log_rows, step, before, len(cleaned))
    print_progress(f"{step}: dropped {before - len(cleaned):,} rows; remaining {len(cleaned):,}")
    return cleaned


def detect_first_matching_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"Could not find a {label} column. Available columns: {list(df.columns)}")


def rename_first_available(
    df: pd.DataFrame,
    candidate_map: dict[str, list[str]],
) -> tuple[pd.DataFrame, list[str]]:
    selected = []
    rename_map: dict[str, str] = {}
    for target_name, candidates in candidate_map.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename_map[candidate] = target_name
                selected.append(target_name)
                break
    renamed = df.rename(columns=rename_map)
    return renamed, selected


def downcast_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    integer_columns = df.select_dtypes(include=["int64", "Int64"]).columns
    float_columns = df.select_dtypes(include=["float64"]).columns

    for column in integer_columns:
        df[column] = pd.to_numeric(df[column], downcast="integer")

    for column in float_columns:
        df[column] = pd.to_numeric(df[column], downcast="float")

    for column in [
        "source_month",
        "store_and_fwd_flag",
        "PUBorough",
        "DOBorough",
        "PUZone",
        "DOZone",
    ]:
        if column in df.columns:
            df[column] = df[column].astype("category")

    return df


def markdown_table_from_dataframe(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_\n"

    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in df.iterrows():
        values = [str(row[column]) for column in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines) + "\n"


def load_monthly_taxi_data(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    frames: list[pd.DataFrame] = []
    source_counts: dict[str, int] = {}

    for source_month, filename in MONTH_FILES.items():
        path = raw_dir / filename
        print_progress(f"Loading {path.relative_to(raw_dir.parent)}")
        monthly = pd.read_parquet(path)
        monthly["source_month"] = source_month

        if "cbd_congestion_fee" not in monthly.columns:
            monthly["cbd_congestion_fee"] = 0.0

        if source_month in {"2024-11", "2024-12"}:
            monthly["cbd_congestion_fee"] = 0.0

        source_counts[source_month] = len(monthly)
        frames.append(monthly)

    combined = pd.concat(frames, ignore_index=True)
    source_counts["combined"] = len(combined)
    print_progress(f"Combined monthly files into {len(combined):,} rows")
    return combined, source_counts


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "tpep_pickup_datetime": "pickup_datetime",
        "tpep_dropoff_datetime": "dropoff_datetime",
        "VendorID": "vendor_id",
        "RatecodeID": "ratecode_id",
        "Airport_fee": "airport_fee",
    }
    standardized = df.rename(columns=rename_map)
    print_progress("Standardized required column names")
    return standardized


def clean_trip_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, float]], dict[str, int]]:
    log_rows: list[dict[str, float]] = []
    imputation_summary = {
        "filled_passenger_count": 0,
        "cbd_fee_negatives_reset": 0,
        "dropped_missing_core_fields": 0,
    }

    print_progress("Parsing datetime columns")
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")

    required_core = [
        "pickup_datetime",
        "dropoff_datetime",
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "payment_type",
        "PULocationID",
        "DOLocationID",
    ]
    missing_core_mask = df[required_core].isna().any(axis=1)
    imputation_summary["dropped_missing_core_fields"] = int(missing_core_mask.sum())
    df = filter_rows(df, missing_core_mask, "drop_missing_core_fields", log_rows)

    df = filter_rows(
        df,
        (df["pickup_datetime"] < PROJECT_START) | (df["pickup_datetime"] >= PROJECT_END),
        "drop_pickup_outside_project_window",
        log_rows,
    )

    df = filter_rows(
        df,
        df["dropoff_datetime"] <= df["pickup_datetime"],
        "drop_non_positive_trip_time",
        log_rows,
    )

    df["duration_minutes"] = (
        (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds() / 60.0
    )

    df = filter_rows(
        df,
        df["duration_minutes"] < 1,
        "drop_duration_lt_1_minute",
        log_rows,
    )
    df = filter_rows(
        df,
        df["duration_minutes"] > 360,
        "drop_duration_gt_360_minutes",
        log_rows,
    )

    df = filter_rows(
        df,
        (df["trip_distance"] <= 0) & (df["fare_amount"] > 0),
        "drop_non_positive_distance_positive_fare",
        log_rows,
    )
    df = filter_rows(
        df,
        df["trip_distance"] > 100,
        "drop_distance_gt_100_miles",
        log_rows,
    )

    df["avg_speed_mph"] = df["trip_distance"] / (df["duration_minutes"] / 60.0)
    df = filter_rows(
        df,
        df["avg_speed_mph"] > 80,
        "drop_avg_speed_gt_80_mph",
        log_rows,
    )

    df = filter_rows(df, df["fare_amount"] < 0, "drop_negative_fare", log_rows)
    df = filter_rows(df, df["fare_amount"] == 0, "drop_zero_fare", log_rows)
    df = filter_rows(df, df["fare_amount"] > 500, "drop_fare_gt_500", log_rows)
    df = filter_rows(df, df["tip_amount"] < 0, "drop_negative_tip", log_rows)

    negative_cbd_mask = df["cbd_congestion_fee"] < 0
    imputation_summary["cbd_fee_negatives_reset"] = int(negative_cbd_mask.sum())
    if imputation_summary["cbd_fee_negatives_reset"] > 0:
        df.loc[negative_cbd_mask, "cbd_congestion_fee"] = 0.0
        print_progress(
            f"Reset {imputation_summary['cbd_fee_negatives_reset']:,} negative cbd_congestion_fee values to 0"
        )

    passenger_fill_mask = df["passenger_count"].isna() | (df["passenger_count"] == 0)
    imputation_summary["filled_passenger_count"] = int(passenger_fill_mask.sum())
    if imputation_summary["filled_passenger_count"] > 0:
        df.loc[passenger_fill_mask, "passenger_count"] = 1
        print_progress(
            f"Filled {imputation_summary['filled_passenger_count']:,} passenger_count values with 1"
        )

    df = filter_rows(
        df,
        df["passenger_count"] > 6,
        "drop_passenger_count_gt_6",
        log_rows,
    )

    df = filter_rows(
        df,
        df["PULocationID"].isin([264, 265]) | df["DOLocationID"].isin([264, 265]),
        "drop_unknown_location_ids_264_265",
        log_rows,
    )

    df = filter_rows(
        df,
        ~df["payment_type"].isin(ALLOWED_PAYMENT_TYPES),
        "drop_invalid_payment_type",
        log_rows,
    )

    print_progress("Creating derived columns")
    df["pickup_hour"] = df["pickup_datetime"].dt.hour.astype("Int16")
    df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek.astype("Int16")
    df["pickup_date"] = df["pickup_datetime"].dt.date
    df["tip_pct"] = np.where(df["fare_amount"] > 0, df["tip_amount"] / df["fare_amount"], np.nan)
    df["post_congestion_fee"] = df["pickup_date"] >= POLICY_START

    return df, log_rows, imputation_summary


def join_taxi_zones(df: pd.DataFrame, zone_path: Path) -> pd.DataFrame:
    print_progress("Joining taxi zone lookup for pickup and dropoff")
    zones = pd.read_csv(zone_path)
    required_columns = {"LocationID", "Borough", "Zone"}
    if not required_columns.issubset(zones.columns):
        raise ValueError(
            f"taxi_zone_lookup.csv must contain {required_columns}; got {set(zones.columns)}"
        )

    pickup_zones = zones[["LocationID", "Borough", "Zone"]].rename(
        columns={
            "LocationID": "PULocationID",
            "Borough": "PUBorough",
            "Zone": "PUZone",
        }
    )
    dropoff_zones = zones[["LocationID", "Borough", "Zone"]].rename(
        columns={
            "LocationID": "DOLocationID",
            "Borough": "DOBorough",
            "Zone": "DOZone",
        }
    )

    df = df.merge(pickup_zones, on="PULocationID", how="left")
    df = df.merge(dropoff_zones, on="DOLocationID", how="left")
    return df


def join_weather(df: pd.DataFrame, weather_path: Path) -> tuple[pd.DataFrame, list[str]]:
    print_progress("Joining hourly weather data")
    weather = pd.read_csv(weather_path)
    weather_time_column = detect_first_matching_column(
        weather,
        ["weather_datetime", "datetime", "time", "timestamp", "date_hour", "hour"],
        "weather datetime",
    )

    weather[weather_time_column] = pd.to_datetime(weather[weather_time_column], errors="coerce")
    weather = weather.dropna(subset=[weather_time_column]).copy()
    weather[weather_time_column] = weather[weather_time_column].dt.floor("h")

    weather_column_aliases = {
        "temperature": ["temperature", "temp", "temperature_2m"],
        "precipitation": ["precipitation", "precip", "prcp"],
        "snow": ["snow", "snowfall"],
        "humidity": ["humidity", "relative_humidity", "rhum"],
        "wind_speed": ["wind_speed", "windspeed", "wind_speed_10m", "wspd"],
    }

    weather, selected_columns = rename_first_available(weather, weather_column_aliases)
    weather_features = ["temperature", "precipitation", "snow", "humidity", "wind_speed"]
    available_features = [column for column in weather_features if column in weather.columns]
    weather = weather[[weather_time_column, *available_features]].drop_duplicates(
        subset=[weather_time_column]
    )
    weather = weather.rename(columns={weather_time_column: "pickup_hour_ts"})

    df["pickup_hour_ts"] = df["pickup_datetime"].dt.floor("h")
    df = df.merge(weather, on="pickup_hour_ts", how="left")
    return df, available_features


def join_holidays(df: pd.DataFrame, holiday_path: Path) -> pd.DataFrame:
    print_progress("Joining holiday indicators")
    holidays = pd.read_csv(holiday_path)
    holiday_date_column = detect_first_matching_column(
        holidays,
        ["date", "holiday_date", "ds", "observed_date"],
        "holiday date",
    )

    holidays[holiday_date_column] = pd.to_datetime(holidays[holiday_date_column], errors="coerce")
    holiday_dates = set(holidays[holiday_date_column].dropna().dt.date.unique())
    df["is_holiday"] = df["pickup_date"].isin(holiday_dates)
    return df


def validate_dataset(df: pd.DataFrame, weather_columns: list[str]) -> dict[str, float]:
    print_progress("Running validation checks")
    checks: dict[str, float] = {}

    checks["fare_amount_le_0"] = int((df["fare_amount"] <= 0).sum())
    checks["duration_out_of_range"] = int(
        ((df["duration_minutes"] < 1) | (df["duration_minutes"] > 360)).sum()
    )
    checks["trip_distance_out_of_range"] = int(
        ((df["trip_distance"] < 0.01) | (df["trip_distance"] > 100)).sum()
    )
    checks["pu_location_out_of_range"] = int(
        ((df["PULocationID"] < 1) | (df["PULocationID"] > 263)).sum()
    )
    checks["do_location_out_of_range"] = int(
        ((df["DOLocationID"] < 1) | (df["DOLocationID"] > 263)).sum()
    )
    checks["missing_pu_borough"] = int(df["PUBorough"].isna().sum())
    checks["missing_do_borough"] = int(df["DOBorough"].isna().sum())

    if weather_columns:
        weather_missing_rate = df[weather_columns].isna().all(axis=1).mean()
    else:
        weather_missing_rate = 1.0
    checks["weather_missing_rate"] = float(weather_missing_rate)

    checks["nonzero_2024_cbd_fee"] = int(
        (
            df["source_month"].isin(["2024-11", "2024-12"])
            & (df["cbd_congestion_fee"].fillna(0) != 0)
        ).sum()
    )
    checks["missing_post_policy_cbd_fee"] = int(
        (
            df["source_month"].isin(["2025-01", "2025-02"])
            & df["post_congestion_fee"]
            & df["cbd_congestion_fee"].isna()
        ).sum()
    )

    failing_checks = {
        name: value
        for name, value in checks.items()
        if name != "weather_missing_rate" and value != 0
    }
    if failing_checks:
        details = ", ".join(f"{name}={value}" for name, value in failing_checks.items())
        raise ValueError(f"Validation failed: {details}")

    if weather_missing_rate > 0.01:
        print_progress(
            f"WARNING: weather missing rate is {weather_missing_rate:.2%}, above the 1% threshold"
        )
    else:
        print_progress(f"Weather missing rate: {weather_missing_rate:.2%}")

    return checks


def write_data_quality_report(
    output_path: Path,
    source_counts: dict[str, int],
    cleaning_log: pd.DataFrame,
    imputation_summary: dict[str, int],
    weather_columns: list[str],
    validation_checks: dict[str, float],
    final_df: pd.DataFrame,
) -> None:
    cleaning_rules = [
        "Dropped rows outside pickup window [2024-11-01, 2025-03-01).",
        "Dropped trips with dropoff_datetime <= pickup_datetime.",
        "Dropped trips with duration_minutes < 1 or > 360.",
        "Dropped trips with trip_distance <= 0 and fare_amount > 0.",
        "Dropped trips with trip_distance > 100 or avg_speed_mph > 80.",
        "Dropped trips with fare_amount <= 0 or fare_amount > 500.",
        "Dropped trips with tip_amount < 0.",
        "Reset negative cbd_congestion_fee values to 0.",
        "Filled missing or zero passenger_count with 1, then dropped passenger_count > 6.",
        "Dropped rows with PULocationID or DOLocationID equal to 264 or 265.",
        "Dropped rows with payment_type outside {1, 2, 3, 4}.",
    ]

    imputation_lines = [
        f"- Filled missing/zero `passenger_count` with 1: {imputation_summary['filled_passenger_count']:,} rows",
        f"- Reset negative `cbd_congestion_fee` values to 0: {imputation_summary['cbd_fee_negatives_reset']:,} rows",
        f"- Dropped rows with missing core fields: {imputation_summary['dropped_missing_core_fields']:,} rows",
    ]

    join_lines = [
        "- Joined taxi zone lookup twice: pickup (`PUBorough`, `PUZone`) and dropoff (`DOBorough`, `DOZone`).",
        "- Joined hourly weather on pickup timestamp floored to the hour.",
        f"- Weather columns included: {', '.join(weather_columns) if weather_columns else 'none detected in weather_hourly.csv'}.",
        "- Added `is_holiday` by matching `pickup_date` against `us_holidays.csv`.",
    ]

    limitations = [
        "- `weather_missing_rate` above 1% should be investigated before downstream modeling.",
        "- `cbd_congestion_fee` post-policy validation only checks for non-missing values, not policy applicability by route.",
        "- The script assumes `weather_hourly.csv` and `us_holidays.csv` contain at least one parseable datetime/date column.",
    ]

    validation_df = pd.DataFrame(
        [
            {"check": key, "value": value}
            for key, value in validation_checks.items()
        ]
    )

    report = f"""# Data Quality Report

## Source Files And Row Counts
{markdown_table_from_dataframe(pd.DataFrame([source_counts]))}

## Cleaning Rules
{chr(10).join(f"- {rule}" for rule in cleaning_rules)}

## Rows Removed At Each Step
{markdown_table_from_dataframe(cleaning_log)}

## Imputation Decisions
{chr(10).join(imputation_lines)}

## Joins Performed
{chr(10).join(join_lines)}

## Validation Summary
{markdown_table_from_dataframe(validation_df)}

## Final Dataset
- Rows: {len(final_df):,}
- Columns: {len(final_df.columns):,}
- Primary output file: `data/processed/clean_trips.parquet`
- CSV export: `data/processed/clean_trips.csv`

## Known Limitations
{chr(10).join(limitations)}
"""

    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    required_paths = [raw_dir / filename for filename in MONTH_FILES.values()]
    required_paths.extend(raw_dir / filename for filename in RAW_FILES.values())
    require_files(required_paths)

    taxi_df, source_counts = load_monthly_taxi_data(raw_dir)
    taxi_df = standardize_columns(taxi_df)
    taxi_df, log_rows, imputation_summary = clean_trip_data(taxi_df)
    taxi_df = join_taxi_zones(taxi_df, raw_dir / RAW_FILES["zones"])
    taxi_df, weather_columns = join_weather(taxi_df, raw_dir / RAW_FILES["weather"])
    taxi_df = join_holidays(taxi_df, raw_dir / RAW_FILES["holidays"])
    validation_checks = validate_dataset(taxi_df, weather_columns)
    taxi_df = downcast_numeric_columns(taxi_df)

    cleaning_log = pd.DataFrame(log_rows)
    cleaning_log_path = processed_dir / "cleaning_log.csv"
    parquet_path = processed_dir / "clean_trips.parquet"
    csv_path = processed_dir / "clean_trips.csv"
    report_path = processed_dir / "data_quality_report.md"

    print_progress(f"Saving cleaned dataset to {parquet_path.relative_to(project_root)}")
    taxi_df.to_parquet(parquet_path, index=False, compression="snappy")

    print_progress(f"Saving CSV export to {csv_path.relative_to(project_root)}")
    taxi_df.to_csv(csv_path, index=False)

    print_progress(f"Saving cleaning log to {cleaning_log_path.relative_to(project_root)}")
    cleaning_log.to_csv(cleaning_log_path, index=False)

    print_progress(f"Saving data quality report to {report_path.relative_to(project_root)}")
    write_data_quality_report(
        output_path=report_path,
        source_counts=source_counts,
        cleaning_log=cleaning_log,
        imputation_summary=imputation_summary,
        weather_columns=weather_columns,
        validation_checks=validation_checks,
        final_df=taxi_df,
    )

    print_progress(
        f"Pipeline completed successfully with {len(taxi_df):,} rows and {len(taxi_df.columns):,} columns"
    )


if __name__ == "__main__":
    main()
