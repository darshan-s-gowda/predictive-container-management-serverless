
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Paths and configuration
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "azure_serverless_workload.csv"
PREDICTIONS_PATH = ROOT / "outputs" / "tft_azure_holdout_predictions.csv"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUCKETS_PER_DAY = 288  # 5-minute buckets
TRAIN_FRACTION = 0.80


def find_column(df, candidates, description):
    """Find a column using case-insensitive candidate names."""
    lookup = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Could not identify {description} column.\n"
        f"Available columns: {list(df.columns)}\n"
        f"Expected one of: {candidates}"
    )


def calculate_metrics(actual, predicted):
    """Calculate MAE and RMSE after removing invalid pairs."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "Count": 0}

    errors = actual - predicted

    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "Count": int(len(errors)),
    }


def main():
    print("Loading Azure workload data...")
    workload = pd.read_csv(DATA_PATH)

    time_col = find_column(
        workload,
        ["time_idx", "time_index", "timestamp_idx"],
        "workload time index",
    )
    load_col = find_column(
        workload,
        ["invocation_rate", "invocations", "invocation_count",
         "request_count", "load"],
        "workload",
    )

    workload = workload[[time_col, load_col]].copy()
    workload.columns = ["time_idx", "actual_load"]
    workload["time_idx"] = pd.to_numeric(
        workload["time_idx"], errors="coerce"
    )
    workload["actual_load"] = pd.to_numeric(
        workload["actual_load"], errors="coerce"
    )
    workload = (
        workload.dropna()
        .sort_values("time_idx")
        .drop_duplicates("time_idx", keep="last")
        .reset_index(drop=True)
    )

    if workload.empty:
        raise ValueError("The workload CSV contains no usable rows.")

    split_idx = int(len(workload) * TRAIN_FRACTION)
    training = workload.iloc[:split_idx].copy()
    holdout = workload.iloc[split_idx:].copy()

    # Use a threshold based only on training data.
    high_load_threshold = float(training["actual_load"].quantile(0.80))

    print("Loading existing TFT predictions...")
    predictions = pd.read_csv(PREDICTIONS_PATH)

    pred_time_col = find_column(
        predictions,
        ["time_idx", "decoder_time_idx", "timestamp_idx",
         "time_index"],
        "prediction time index",
    )
    actual_col = find_column(
        predictions,
        ["actual", "actual_load", "actual_invocation_rate",
         "actual_invocations", "y_true", "target"],
        "actual workload in prediction file",
    )
    tft_col = find_column(
        predictions,
        ["prediction", "predicted", "tft_prediction",
         "tft_predicted", "prediction_median", "predicted_load",
         "tft"],
        "TFT prediction",
    )

    predictions = predictions[
        [pred_time_col, actual_col, tft_col]
    ].copy()
    predictions.columns = ["time_idx", "actual_from_file", "tft"]
    predictions["time_idx"] = pd.to_numeric(
        predictions["time_idx"], errors="coerce"
    )
    predictions["actual_from_file"] = pd.to_numeric(
        predictions["actual_from_file"], errors="coerce"
    )
    predictions["tft"] = pd.to_numeric(
        predictions["tft"], errors="coerce"
    )

    predictions = (
        predictions.dropna(subset=["time_idx"])
        .sort_values("time_idx")
        .drop_duplicates("time_idx", keep="first")
    )

    # Join predictions to the original workload by timestamp.
    # The original dataset is authoritative for actual workload.
    results = predictions[["time_idx", "tft"]].merge(
        workload, on="time_idx", how="inner"
    )

    # Keep only chronological holdout timestamps.
    holdout_times = set(holdout["time_idx"].tolist())
    results = results[
        results["time_idx"].isin(holdout_times)
    ].copy()

    # Seasonal-naive forecast: workload at the same time yesterday.
    previous_day = workload[["time_idx", "actual_load"]].copy()
    previous_day["time_idx"] += BUCKETS_PER_DAY
    previous_day = previous_day.rename(
        columns={"actual_load": "seasonal_naive"}
    )

    results = results.merge(
        previous_day, on="time_idx", how="left"
    )

    # Persistence forecast: latest actual workload from the
    # preceding bucket. This is a fair one-step baseline.
    previous_bucket = workload[["time_idx", "actual_load"]].copy()
    previous_bucket["time_idx"] += 1
    previous_bucket = previous_bucket.rename(
        columns={"actual_load": "persistence"}
    )

    results = results.merge(
        previous_bucket, on="time_idx", how="left"
    )

    results = results.sort_values("time_idx").reset_index(drop=True)

    if results.empty:
        raise ValueError(
            "No matching holdout timestamps found. Check the time_idx "
            "column in the workload and TFT prediction CSV files."
        )

    # High-load status is based on the actual observed value.
    results["high_load"] = (
        results["actual_load"] >= high_load_threshold
    )

    forecast_columns = {
        "TFT": "tft",
        "Persistence": "persistence",
        "Seasonal naive (previous day)": "seasonal_naive",
    }

    metric_rows = []

    for model_name, forecast_col in forecast_columns.items():
        for segment_name, subset in [
            ("Overall", results),
            ("High load", results[results["high_load"]]),
            ("Other load", results[~results["high_load"]]),
        ]:
            metrics = calculate_metrics(
                subset["actual_load"], subset[forecast_col]
            )

            metric_rows.append({
                "model": model_name,
                "segment": segment_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "count": metrics["Count"],
            })

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = OUTPUT_DIR / "azure_forecast_comparison_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Underprediction on high-load periods.
    high = results[results["high_load"]]
    safety_rows = []

    for model_name, forecast_col in forecast_columns.items():
        valid = high[
            np.isfinite(high["actual_load"])
            & np.isfinite(high[forecast_col])
        ]

        if len(valid):
            underprediction_pct = (
                valid["actual_load"] > valid[forecast_col]
            ).mean() * 100
        else:
            underprediction_pct = np.nan

        safety_rows.append({
            "model": model_name,
            "high_load_count": len(valid),
            "high_load_underprediction_pct": underprediction_pct,
            "high_load_MAE": calculate_metrics(
                valid["actual_load"], valid[forecast_col]
            )["MAE"],
        })

    safety_df = pd.DataFrame(safety_rows)
    safety_path = OUTPUT_DIR / "azure_forecast_comparison_safety.csv"
    safety_df.to_csv(safety_path, index=False)

    # --------------------------------------------------------
    # Plot 1: representative section of the holdout workload
    # --------------------------------------------------------
    plot_data = results.copy()
    plot_data["absolute_error_tft"] = (
        plot_data["actual_load"] - plot_data["tft"]
    ).abs()

    # Show a representative 3-day section with the largest
    # observed workload spike in the holdout.
    spike_idx = plot_data["actual_load"].idxmax()
    start = max(0, spike_idx - BUCKETS_PER_DAY)
    end = min(
        len(plot_data),
        spike_idx + 2 * BUCKETS_PER_DAY
    )
    window = plot_data.iloc[start:end]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(
        window["time_idx"], window["actual_load"],
        label="Actual workload", linewidth=1.5
    )
    ax.plot(
        window["time_idx"], window["tft"],
        label="TFT forecast", linewidth=1.2
    )
    ax.plot(
        window["time_idx"], window["persistence"],
        label="Persistence forecast", alpha=0.8
    )
    ax.plot(
        window["time_idx"], window["seasonal_naive"],
        label="Previous-day forecast", alpha=0.8
    )
    ax.axhline(
        high_load_threshold,
        linestyle="--",
        label=f"High-load threshold ({high_load_threshold:.0f})"
    )
    ax.set_title("Azure workload: forecasts around the largest holdout spike")
    ax.set_xlabel("Elapsed 5-minute bucket index")
    ax.set_ylabel("Invocations per bucket")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plot_path = OUTPUT_DIR / "azure_forecast_comparison.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    # --------------------------------------------------------
    # Plot 2: actual vs TFT predictions
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        results["actual_load"], results["tft"],
        s=12, alpha=0.45
    )

    max_value = float(
        max(results["actual_load"].max(), results["tft"].max())
    )
    ax.plot([0, max_value], [0, max_value], linestyle="--")
    ax.set_title("Azure TFT: predicted versus actual workload")
    ax.set_xlabel("Actual invocations per bucket")
    ax.set_ylabel("TFT-predicted invocations per bucket")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    scatter_path = OUTPUT_DIR / "azure_tft_actual_vs_predicted.png"
    fig.savefig(scatter_path, dpi=160)
    plt.close(fig)

    print("\n=== Azure forecast comparison ===")
    print(f"Training buckets: {len(training)}")
    print(f"Expected holdout buckets: {len(holdout)}")
    print(f"Matched prediction timestamps: {len(results)}")
    print(f"High-load threshold: {high_load_threshold:.2f}")
    print("\nMetrics:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nHigh-load underprediction:")
    print(safety_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nSaved files:")
    print(metrics_path)
    print(safety_path)
    print(plot_path)
    print(scatter_path)

    if len(results) < len(holdout):
        print(
            "\nWARNING: Some holdout timestamps have no matching TFT "
            "prediction. Check the evaluation CSV before drawing conclusions."
        )


if __name__ == "__main__":
    main()
