
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

WORKLOAD_FILE = ROOT / "data" / "azure_serverless_workload.csv"
ORIGINAL_FILE = OUTPUT_DIR / "tft_azure_holdout_predictions.csv"
LOG_FILE = OUTPUT_DIR / "tft_azure_log_predictions.csv"

METRICS_FILE = OUTPUT_DIR / "azure_tft_fair_comparison_metrics.csv"
SAFETY_FILE = OUTPUT_DIR / "azure_tft_fair_comparison_safety.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "azure_tft_fair_comparison_predictions.csv"
PLOT_FILE = OUTPUT_DIR / "azure_tft_fair_comparison.png"


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]

    if len(actual) == 0:
        return np.nan, np.nan, 0

    error = actual - predicted
    return (
        float(np.mean(np.abs(error))),
        float(np.sqrt(np.mean(error ** 2))),
        len(actual),
    )


def main():
    print("Loading existing Azure forecast files...")

    workload = pd.read_csv(WORKLOAD_FILE)
    original = pd.read_csv(ORIGINAL_FILE)
    log = pd.read_csv(LOG_FILE)

    required_original = {
        "time_idx", "actual", "tft_prediction", "baseline_prediction"
    }
    required_log = {
        "time_idx", "actual_invocations",
        "log_tft_predicted_invocations", "persistence_prediction"
    }

    if not required_original.issubset(original.columns):
        raise ValueError(
            f"Original TFT file is missing columns: "
            f"{required_original - set(original.columns)}"
        )

    if not required_log.issubset(log.columns):
        raise ValueError(
            f"Log TFT file is missing columns: "
            f"{required_log - set(log.columns)}"
        )

    # Find the workload column for a training-only high-load threshold.
    columns = {str(c).strip().lower(): c for c in workload.columns}
    load_candidates = [
        "invocation_rate", "invocations", "invocation_count",
        "request_count", "load"
    ]
    time_candidates = ["time_idx", "time_index", "timestamp_idx"]

    load_col = next(
        (columns[c] for c in load_candidates if c in columns), None
    )
    time_col = next(
        (columns[c] for c in time_candidates if c in columns), None
    )

    if load_col is None or time_col is None:
        raise ValueError(
            f"Could not identify workload columns: {list(workload.columns)}"
        )

    workload = workload[[time_col, load_col]].copy()
    workload.columns = ["time_idx", "actual_workload"]
    workload["time_idx"] = pd.to_numeric(
        workload["time_idx"], errors="coerce"
    )
    workload["actual_workload"] = pd.to_numeric(
        workload["actual_workload"], errors="coerce"
    )
    workload = (
        workload.dropna()
        .sort_values("time_idx")
        .drop_duplicates("time_idx", keep="last")
    )

    # Preserve the original chronological 80/20 split.
    split = int(len(workload) * 0.80)
    training_workload = workload.iloc[:split]
    threshold = float(training_workload["actual_workload"].quantile(0.80))

    # Rename prediction columns to a common schema.
    original = original.rename(columns={
        "actual": "original_actual",
        "tft_prediction": "original_tft",
        "baseline_prediction": "original_persistence",
    })

    log = log.rename(columns={
        "actual_invocations": "log_actual",
        "log_tft_predicted_invocations": "log_tft",
        "persistence_prediction": "log_persistence",
    })

    for df in (original, log):
        df["time_idx"] = pd.to_numeric(df["time_idx"], errors="coerce")

    # One row per timestamp in each file.
    original = (
        original.dropna(subset=["time_idx"])
        .drop_duplicates("time_idx", keep="first")
    )
    log = (
        log.dropna(subset=["time_idx"])
        .drop_duplicates("time_idx", keep="first")
    )

    # Only compare timestamps present in both files and the original data.
    common = original[
        ["time_idx", "original_actual", "original_tft",
         "original_persistence"]
    ].merge(
        log[["time_idx", "log_actual", "log_tft", "log_persistence"]],
        on="time_idx",
        how="inner",
        validate="one_to_one",
    )

    common = common.merge(
        workload[["time_idx", "actual_workload"]],
        on="time_idx",
        how="inner",
        validate="one_to_one",
    )

    # Keep holdout timestamps only.
    holdout_times = set(workload.iloc[split:]["time_idx"])
    common = common[common["time_idx"].isin(holdout_times)].copy()
    common = common.sort_values("time_idx").reset_index(drop=True)

    if common.empty:
        raise ValueError("No common holdout timestamps were found.")

    # Actuals should agree between each file and the source dataset.
    common["actual_difference_original"] = (
        common["original_actual"] - common["actual_workload"]
    ).abs()
    common["actual_difference_log"] = (
        common["log_actual"] - common["actual_workload"]
    ).abs()

    # Compare the persistence columns across both saved evaluations.
    common["persistence_difference"] = (
        common["original_persistence"] - common["log_persistence"]
    ).abs()

    print(f"Total workload buckets: {len(workload)}")
    print(f"Expected holdout buckets: {len(workload) - split}")
    print(f"Original TFT prediction rows: {len(original)}")
    print(f"Log-TFT prediction rows: {len(log)}")
    print(f"Common holdout timestamps: {len(common)}")
    print(f"Training-derived high-load threshold: {threshold:.2f}")

    print("\n=== Alignment checks ===")
    print(
        "Max original-actual difference vs workload:",
        f"{common['actual_difference_original'].max():.6f}"
    )
    print(
        "Max log-actual difference vs workload:",
        f"{common['actual_difference_log'].max():.6f}"
    )
    print(
        "Max difference between persistence columns:",
        f"{common['persistence_difference'].max():.6f}"
    )

    common["high_load"] = common["actual_workload"] >= threshold

    prediction_columns = {
        "Original TFT": "original_tft",
        "Log-transformed TFT": "log_tft",
        "Persistence": "original_persistence",
    }

    metric_rows = []
    safety_rows = []

    for model_name, pred_col in prediction_columns.items():
        for segment, subset in [
            ("Overall", common),
            ("High load", common[common["high_load"]]),
            ("Other load", common[~common["high_load"]]),
        ]:
            mae, rmse, count = metrics(
                subset["actual_workload"], subset[pred_col]
            )
            metric_rows.append({
                "model": model_name,
                "segment": segment,
                "MAE": mae,
                "RMSE": rmse,
                "count": count,
            })

        high = common[common["high_load"]].dropna(
            subset=["actual_workload", pred_col]
        )

        if len(high):
            under_pct = (
                high["actual_workload"] > high[pred_col]
            ).mean() * 100
            high_mae, high_rmse, high_count = metrics(
                high["actual_workload"], high[pred_col]
            )
        else:
            under_pct = np.nan
            high_mae = np.nan
            high_count = 0

        safety_rows.append({
            "model": model_name,
            "high_load_count": high_count,
            "high_load_underprediction_pct": under_pct,
            "high_load_MAE": high_mae,
        })

    metrics_df = pd.DataFrame(metric_rows)
    safety_df = pd.DataFrame(safety_rows)

    metrics_df.to_csv(METRICS_FILE, index=False)
    safety_df.to_csv(SAFETY_FILE, index=False)
    common.to_csv(PREDICTIONS_FILE, index=False)

    # Plot a representative window around the largest actual spike.
    spike_position = common["actual_workload"].idxmax()
    start = max(0, spike_position - 288)
    end = min(len(common), spike_position + 2 * 288)
    window = common.iloc[start:end]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(
        window["time_idx"], window["actual_workload"],
        label="Actual workload", linewidth=1.5
    )
    ax.plot(
        window["time_idx"], window["original_tft"],
        label="Original TFT", linewidth=1.1
    )
    ax.plot(
        window["time_idx"], window["log_tft"],
        label="Log-transformed TFT", linewidth=1.1
    )
    ax.plot(
        window["time_idx"], window["original_persistence"],
        label="Persistence", linewidth=1.0, alpha=0.8
    )
    ax.axhline(
        threshold, linestyle="--",
        label=f"High-load threshold ({threshold:.0f})"
    )
    ax.set_title("Azure forecast comparison on common holdout timestamps")
    ax.set_xlabel("Elapsed 5-minute bucket index")
    ax.set_ylabel("Invocations per bucket")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=160)
    plt.close(fig)

    print("\n=== Fair forecast comparison ===")
    print(metrics_df.to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))

    print("\n=== High-load safety comparison ===")
    print(safety_df.to_string(
        index=False, float_format=lambda x: f"{x:.2f}"
    ))

    # Explicitly flag a baseline mismatch.
    if common["persistence_difference"].max() > 1e-6:
        print(
            "\nWARNING: persistence columns differ. The metric comparison "
            "uses original_persistence as the shared baseline. Investigate "
            "the timestamp or baseline definitions before final reporting."
        )
    else:
        print("\nPersistence columns agree on common timestamps.")

    print("\nSaved:")
    print(METRICS_FILE)
    print(SAFETY_FILE)
    print(PREDICTIONS_FILE)
    print(PLOT_FILE)


if __name__ == "__main__":
    main()
