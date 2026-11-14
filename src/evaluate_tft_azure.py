
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "azure_serverless_workload.csv"
OUT_DIR = ROOT / "outputs"
MODEL_PATH = OUT_DIR / "tft_azure_state_dict.pt"

ENCODER_LENGTH = 36
PREDICTION_LENGTH = 12
BATCH_SIZE = 64
SAFETY_MARGINS = [0.0, 100.0, 250.0, 500.0]


def prepare_data():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("time_idx").reset_index(drop=True)

    t = df["time_idx"].to_numpy()
    df["time_of_day_sin"] = np.sin(2 * np.pi * (t % 288) / 288)
    df["time_of_day_cos"] = np.cos(2 * np.pi * (t % 288) / 288)
    df["weekly_phase_sin"] = np.sin(2 * np.pi * (t % 2016) / 2016)
    df["weekly_phase_cos"] = np.cos(2 * np.pi * (t % 2016) / 2016)
    df["series"] = "azure_functions"

    return df


def calculate_metrics(actual, predicted):
    error = actual - predicted
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run python src/train_tft_azure.py first."
        )

    df = prepare_data()
    split_idx = int(len(df) * 0.8)
    train_df = df[df["time_idx"] < split_idx].copy()

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="invocation_rate",
        group_ids=["series"],
        min_encoder_length=ENCODER_LENGTH,
        max_encoder_length=ENCODER_LENGTH,
        min_prediction_length=PREDICTION_LENGTH,
        max_prediction_length=PREDICTION_LENGTH,
        time_varying_known_reals=[
            "time_idx",
            "time_of_day_sin",
            "time_of_day_cos",
            "weekly_phase_sin",
            "weekly_phase_cos",
        ],
        time_varying_unknown_reals=[
            "invocation_rate",
            "mean_duration_seconds",
        ],
        target_normalizer="auto",
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    testing = TimeSeriesDataSet.from_dataset(
        training,
        df,
        min_prediction_idx=split_idx,
        predict=False,
        stop_randomization=True,
    )

    test_loader = testing.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,
        loss=QuantileLoss(),
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()

    print("Generating Azure holdout forecasts...")
    result = model.predict(
        test_loader,
        mode="prediction",
        return_x=True,
        return_y=True,
    )

    predictions = result.output.detach().cpu().numpy()
    targets = result.y[0] if isinstance(result.y, tuple) else result.y
    actual = targets.detach().cpu().numpy()
    decoder_idx = result.x["decoder_time_idx"].detach().cpu().numpy()
    encoder_target = result.x["encoder_target"].detach().cpu().numpy()

    predictions = np.asarray(predictions).reshape(-1, PREDICTION_LENGTH)
    actual = np.asarray(actual).reshape(-1, PREDICTION_LENGTH)
    decoder_idx = np.asarray(decoder_idx).reshape(-1, PREDICTION_LENGTH)

    if not (
        predictions.shape == actual.shape == decoder_idx.shape
    ):
        raise RuntimeError(
            f"Shape mismatch: {predictions.shape}, "
            f"{actual.shape}, {decoder_idx.shape}"
        )

    last_observed = encoder_target[:, -1]
    baseline = np.repeat(
        last_observed[:, None],
        PREDICTION_LENGTH,
        axis=1,
    )

    # Collect all forecast horizons, then keep the prediction with
    # the shortest lead time for each holdout timestamp.
    records = []

    for i in range(len(predictions)):
        for j in range(PREDICTION_LENGTH):
            idx = int(decoder_idx[i, j])

            if split_idx <= idx < len(df):
                records.append({
                    "time_idx": idx,
                    "lead_steps": j + 1,
                    "actual": float(actual[i, j]),
                    "tft_prediction": float(predictions[i, j]),
                    "baseline_prediction": float(baseline[i, j]),
                })

    if not records:
        raise RuntimeError("No Azure holdout forecasts were generated.")

    forecasts = pd.DataFrame(records)
    forecasts = (
        forecasts.sort_values(["time_idx", "lead_steps"])
        .drop_duplicates("time_idx", keep="first")
        .sort_values("time_idx")
        .reset_index(drop=True)
    )

    expected = set(range(split_idx, len(df)))
    covered = set(forecasts["time_idx"].astype(int))
    missing = sorted(expected - covered)

    forecasts["timestamp_seconds"] = (
        df.set_index("time_idx")
        .loc[forecasts["time_idx"].to_numpy(), "timestamp_seconds"]
        .to_numpy()
    )

    forecasts.to_csv(
        OUT_DIR / "tft_azure_holdout_predictions.csv",
        index=False,
    )

    y = forecasts["actual"].to_numpy()
    p = forecasts["tft_prediction"].to_numpy()
    b = forecasts["baseline_prediction"].to_numpy()

    metrics = pd.DataFrame([
        {"model": "TFT", **calculate_metrics(y, p)},
        {
            "model": "Persistence baseline",
            **calculate_metrics(y, b),
        },
    ])
    metrics.to_csv(
        OUT_DIR / "tft_azure_accuracy_metrics.csv",
        index=False,
    )

    # Define high workload using the training set only.
    threshold = float(train_df["invocation_rate"].quantile(0.75))
    high = y >= threshold

    margin_rows = []
    for margin in SAFETY_MARGINS:
        adjusted = np.maximum(0.0, p + margin)

        margin_rows.append({
            "safety_margin_invocations_per_bucket": margin,
            "overall_MAE": float(np.mean(np.abs(y - adjusted))),
            "overall_RMSE": float(np.sqrt(np.mean((y - adjusted) ** 2))),
            "high_load_MAE": (
                float(np.mean(np.abs(y[high] - adjusted[high])))
                if high.any() else float("nan")
            ),
            "high_load_underprediction_pct": (
                float(np.mean(adjusted[high] < y[high]) * 100)
                if high.any() else float("nan")
            ),
        })

    margin_df = pd.DataFrame(margin_rows)
    margin_df.to_csv(
        OUT_DIR / "tft_azure_safety_margin_results.csv",
        index=False,
    )

    # Plot an example 12-step forecast.
    first_idx = int(forecasts.iloc[0]["time_idx"])
    window = forecasts[
        (forecasts["time_idx"] >= first_idx)
        & (forecasts["time_idx"] < first_idx + PREDICTION_LENGTH)
    ]

    plt.figure(figsize=(11, 5))
    plt.plot(window["time_idx"], window["actual"], marker="o", label="Actual")
    plt.plot(
        window["time_idx"],
        window["tft_prediction"],
        marker="o",
        label="TFT",
    )
    plt.plot(
        window["time_idx"],
        window["baseline_prediction"],
        linestyle="--",
        label="Persistence",
    )
    plt.xlabel("Time index (five-minute intervals)")
    plt.ylabel("Invocation count per bucket")
    plt.title("Azure TFT Holdout Forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "tft_azure_forecast.png",
        dpi=150,
    )
    plt.close()

    print("\n=== Azure holdout accuracy ===")
    print(metrics.round(3).to_string(index=False))

    print(f"\nTraining-derived high-load threshold: {threshold:.2f}")
    print(f"High-load holdout timestamps: {int(high.sum())}")
    print(f"Unique timestamps evaluated: {len(forecasts)}")
    print(f"Expected holdout timestamps: {len(expected)}")
    print(f"Missing timestamps: {len(missing)}")

    print("\n=== Safety-margin experiment ===")
    print(margin_df.round(3).to_string(index=False))

    if missing:
        print("WARNING: Some holdout timestamps have no predictions.")
        print("First missing indices:", missing[:20])
    else:
        print("All holdout timestamps are covered exactly once.")

    print("\nSaved Azure evaluation outputs in outputs/.")


if __name__ == "__main__":
    main()
