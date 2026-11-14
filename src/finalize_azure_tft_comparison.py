
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

DATA = ROOT / "data" / "azure_serverless_workload.csv"
ORIGINAL = OUT / "tft_azure_holdout_predictions.csv"
LOG = OUT / "tft_azure_log_predictions.csv"

METRICS = OUT / "azure_tft_final_metrics.csv"
SAFETY = OUT / "azure_tft_final_safety.csv"
PREDICTIONS = OUT / "azure_tft_final_predictions.csv"
PLOT = OUT / "azure_tft_final_comparison.png"


def score(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]

    if not len(actual):
        return np.nan, np.nan, 0

    error = actual - predicted
    return (
        float(np.mean(np.abs(error))),
        float(np.sqrt(np.mean(error ** 2))),
        int(len(actual)),
    )


def main():
    data = pd.read_csv(DATA)
    original = pd.read_csv(ORIGINAL)
    log = pd.read_csv(LOG)

    time_col = next(
        c for c in data.columns
        if c.lower() in ("time_idx", "time_index", "timestamp_idx")
    )
    load_col = next(
        c for c in data.columns
        if c.lower() in (
            "invocation_rate", "invocations", "invocation_count",
            "request_count", "load"
        )
    )

    data = data[[time_col, load_col]].copy()
    data.columns = ["time_idx", "actual"]
    data["time_idx"] = pd.to_numeric(data["time_idx"], errors="coerce")
    data["actual"] = pd.to_numeric(data["actual"], errors="coerce")
    data = (
        data.dropna()
        .sort_values("time_idx")
        .drop_duplicates("time_idx", keep="last")
        .reset_index(drop=True)
    )

    split = int(len(data) * 0.80)
    training = data.iloc[:split]
    holdout = data.iloc[split:]

    threshold = float(training["actual"].quantile(0.80))

    # Build one canonical persistence forecast from the source workload.
    # Forecast at t uses the observed workload at t-1.
    persistence = data[["time_idx", "actual"]].copy()
    persistence["time_idx"] += 1
    persistence = persistence.rename(
        columns={"actual": "persistence"}
    )

    original = original[[
        "time_idx", "tft_prediction"
    ]].rename(columns={"tft_prediction": "original_tft"})

    log = log[[
        "time_idx", "log_tft_predicted_invocations"
    ]].rename(columns={
        "log_tft_predicted_invocations": "log_tft"
    })

    result = (
        holdout
        .merge(original, on="time_idx", how="inner", validate="one_to_one")
        .merge(log, on="time_idx", how="inner", validate="one_to_one")
        .merge(persistence, on="time_idx", how="left", validate="one_to_one")
        .sort_values("time_idx")
        .reset_index(drop=True)
    )

    if len(result) != len(holdout):
        raise ValueError(
            f"Only {len(result)} of {len(holdout)} holdout timestamps "
            "matched. Check prediction files before proceeding."
        )

    result["high_load"] = result["actual"] >= threshold
    result.to_csv(PREDICTIONS, index=False)

    forecasts = {
        "Original TFT": "original_tft",
        "Log-transformed TFT": "log_tft",
        "Persistence (previous bucket)": "persistence",
    }

    metric_rows = []
    safety_rows = []

    for name, col in forecasts.items():
        for segment, subset in [
            ("Overall", result),
            ("High load", result[result["high_load"]]),
            ("Other load", result[~result["high_load"]]),
        ]:
            mae, rmse, count = score(subset["actual"], subset[col])
            metric_rows.append({
                "model": name,
                "segment": segment,
                "MAE": mae,
                "RMSE": rmse,
                "count": count,
            })

        high = result[result["high_load"]].dropna(
            subset=["actual", col]
        )
        under_pct = (
            (high["actual"] > high[col]).mean() * 100
            if len(high) else np.nan
        )
        high_mae, _, high_count = score(high["actual"], high[col])

        safety_rows.append({
            "model": name,
            "high_load_count": high_count,
            "high_load_underprediction_pct": under_pct,
            "high_load_MAE": high_mae,
        })

    metrics = pd.DataFrame(metric_rows)
    safety = pd.DataFrame(safety_rows)
    metrics.to_csv(METRICS, index=False)
    safety.to_csv(SAFETY, index=False)

    spike = int(result["actual"].idxmax())
    window = result.iloc[
        max(0, spike - 288):min(len(result), spike + 576)
    ]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(window["time_idx"], window["actual"], label="Actual")
    ax.plot(window["time_idx"], window["original_tft"], label="Original TFT")
    ax.plot(window["time_idx"], window["log_tft"], label="Log-TFT")
    ax.plot(
        window["time_idx"], window["persistence"],
        label="Persistence", alpha=0.85
    )
    ax.axhline(
        threshold, linestyle="--",
        label=f"High-load threshold ({threshold:.0f})"
    )
    ax.set_title("Azure forecast comparison — common holdout")
    ax.set_xlabel("Elapsed 5-minute bucket index")
    ax.set_ylabel("Invocations per bucket")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT, dpi=160)
    plt.close(fig)

    print(f"Training buckets: {len(training)}")
    print(f"Holdout buckets: {len(holdout)}")
    print(f"Matched timestamps: {len(result)}")
    print(f"Training-only high-load threshold: {threshold:.2f}")

    print("\n=== Final comparison metrics ===")
    print(metrics.to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))

    print("\n=== High-load safety metrics ===")
    print(safety.to_string(
        index=False, float_format=lambda x: f"{x:.2f}"
    ))

    print("\nSaved:")
    print(METRICS)
    print(SAFETY)
    print(PREDICTIONS)
    print(PLOT)


if __name__ == "__main__":
    main()
