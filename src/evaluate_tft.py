
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from lightning.pytorch import Trainer
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic_serverless_workload.csv"
OUT_DIR = ROOT / "outputs"

ENCODER_LENGTH = 36
PREDICTION_LENGTH = 12
EPOCHS = 5
BATCH_SIZE = 64
SAFETY_MARGINS = [0.0, 1.0, 2.0, 3.0]


def calculate_metrics(actual, predicted):
    errors = actual - predicted
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df["time_idx"] = np.arange(len(df))
    df["series"] = 0

    split_idx = int(len(df) * 0.8)
    train_df = df[df["time_idx"] < split_idx].copy()

    print(f"Total rows: {len(df)}")
    print(f"Training rows: {len(train_df)}")
    print(f"Expected holdout timestamps: {len(df) - split_idx}")

    # 2. Training dataset
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
            "hour",
            "day_of_week",
        ],
        time_varying_unknown_reals=[
            "invocation_rate",
            "cpu_utilization",
            "memory_utilization",
            "latency_ms",
        ],
        target_normalizer="auto",
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Keep the training encoders and scalers for holdout data.
    testing = TimeSeriesDataSet.from_dataset(
        training,
        df,
        min_prediction_idx=split_idx,
        predict=False,
        stop_randomization=True,
    )

    print(f"Test dataset windows: {len(testing)}")

    train_loader = training.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    test_loader = testing.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    # 3. Build and train TFT
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,
        loss=QuantileLoss(),
    )

    trainer = Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    print("\nTraining TFT...")
    trainer.fit(model, train_loader)

    # 4. Predict holdout windows, retaining timestamp indices.
    print("\nPredicting holdout windows...")

    result = model.predict(
        test_loader,
        mode="prediction",
        return_x=True,
        return_y=True,
    )

    predictions = result.output.detach().cpu().numpy()
    actual = result.y[0] if isinstance(result.y, tuple) else result.y
    actual = actual.detach().cpu().numpy()

    decoder_indices = (
        result.x["decoder_time_idx"].detach().cpu().numpy()
    )
    encoder_targets = (
        result.x["encoder_target"].detach().cpu().numpy()
    )

    # Normalize array shapes to (windows, forecast horizon).
    predictions = np.asarray(predictions).reshape(
        -1, PREDICTION_LENGTH
    )
    actual = np.asarray(actual).reshape(
        -1, PREDICTION_LENGTH
    )
    decoder_indices = np.asarray(decoder_indices).reshape(
        -1, PREDICTION_LENGTH
    )

    if not (
        predictions.shape == actual.shape == decoder_indices.shape
    ):
        raise RuntimeError(
            "Inconsistent prediction, target, and timestamp shapes: "
            f"{predictions.shape}, {actual.shape}, "
            f"{decoder_indices.shape}"
        )

    if len(encoder_targets) != len(predictions):
        raise RuntimeError("Encoder and prediction window counts differ.")

    # Persistence baseline: last observed workload for each window.
    last_observed = encoder_targets[:, -1]
    baseline = np.repeat(
        last_observed[:, None],
        PREDICTION_LENGTH,
        axis=1,
    )

    # 5. Build a timestamp-level table from every forecast horizon.
    # For timestamps predicted more than once, keep the shortest
    # lead time (the earliest forecast available for that timestamp).
    records = []

    for window_idx in range(len(predictions)):
        for horizon_idx in range(PREDICTION_LENGTH):
            timestamp_idx = int(decoder_indices[window_idx, horizon_idx])

            if timestamp_idx < split_idx or timestamp_idx >= len(df):
                continue

            records.append({
                "time_idx": timestamp_idx,
                "lead_steps": horizon_idx + 1,
                "actual": float(actual[window_idx, horizon_idx]),
                "tft_prediction": float(
                    predictions[window_idx, horizon_idx]
                ),
                "baseline_prediction": float(
                    baseline[window_idx, horizon_idx]
                ),
            })

    if not records:
        raise RuntimeError(
            "No holdout predictions were found. "
            "Check decoder_time_idx and the holdout split."
        )

    all_forecasts = pd.DataFrame(records)

    # One row per timestamp, preferring the shortest lead time.
    unique_forecasts = (
        all_forecasts
        .sort_values(["time_idx", "lead_steps"])
        .drop_duplicates(subset=["time_idx"], keep="first")
        .sort_values("time_idx")
        .reset_index(drop=True)
    )

    expected_indices = set(range(split_idx, len(df)))
    covered_indices = set(unique_forecasts["time_idx"].astype(int))
    missing_indices = sorted(expected_indices - covered_indices)

    print(f"Unique holdout timestamps evaluated: {len(unique_forecasts)}")
    print(f"Expected unique holdout timestamps: {len(expected_indices)}")
    print(f"Missing holdout timestamps: {len(missing_indices)}")

    unique_forecasts["timestamp"] = (
        df.set_index("time_idx")
        .loc[unique_forecasts["time_idx"].to_numpy(), "timestamp"]
        .to_numpy()
    )

    # Save the timestamp-level evaluation data for auditing.
    unique_forecasts.to_csv(
        OUT_DIR / "tft_timestamp_level_predictions.csv",
        index=False,
    )

    # 6. One prediction per timestamp: overall metrics.
    y_true = unique_forecasts["actual"].to_numpy()
    y_pred = unique_forecasts["tft_prediction"].to_numpy()
    y_base = unique_forecasts["baseline_prediction"].to_numpy()

    scores = pd.DataFrame([
        {"model": "TFT", **calculate_metrics(y_true, y_pred)},
        {
            "model": "Persistence baseline",
            **calculate_metrics(y_true, y_base),
        },
    ])

    scores.to_csv(
        OUT_DIR / "tft_accuracy_metrics.csv",
        index=False,
    )

    # 7. Define high load from the training set only.
    high_threshold = float(
        train_df["invocation_rate"].quantile(0.75)
    )
    high_mask = y_true >= high_threshold

    # 8. Compare safety margins using the same timestamps.
    margin_rows = []

    for margin in SAFETY_MARGINS:
        adjusted = np.maximum(0.0, y_pred + margin)

        if high_mask.any():
            high_mae = float(
                np.mean(
                    np.abs(y_true[high_mask] - adjusted[high_mask])
                )
            )
            high_underprediction_pct = float(
                np.mean(
                    adjusted[high_mask] < y_true[high_mask]
                ) * 100
            )
        else:
            high_mae = float("nan")
            high_underprediction_pct = float("nan")

        margin_rows.append({
            "safety_margin": margin,
            "overall_MAE": float(
                np.mean(np.abs(y_true - adjusted))
            ),
            "overall_RMSE": float(
                np.sqrt(np.mean((y_true - adjusted) ** 2))
            ),
            "high_load_MAE": high_mae,
            "high_load_underprediction_pct": (
                high_underprediction_pct
            ),
        })

    margin_scores = pd.DataFrame(margin_rows)

    margin_scores.to_csv(
        OUT_DIR / "tft_safety_margin_results.csv",
        index=False,
    )

    # Save compatible arrays for analyze_tft_errors.py.
    np.savez(
        OUT_DIR / "tft_holdout_predictions.npz",
        actual=y_true,
        predicted=y_pred,
        baseline=y_base,
    )

    # 9. Plot an actual 12-step forecast window.
    plt.figure(figsize=(10, 5))
    plt.plot(actual[0], marker="o", label="Actual")
    plt.plot(predictions[0], marker="o", label="TFT prediction")
    plt.plot(
        baseline[0],
        linestyle="--",
        label="Persistence baseline",
    )
    plt.xlabel("Forecast step (5-minute intervals)")
    plt.ylabel("Invocation rate")
    plt.title("TFT Holdout Forecast vs Actual Workload")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "tft_accuracy_comparison.png",
        dpi=150,
    )
    plt.close()

    # 10. Save evaluation model separately.
    torch.save(
        model.state_dict(),
        OUT_DIR / "tft_eval_state_dict.pt",
    )

    # 11. Report results.
    print("\n=== Timestamp-level holdout accuracy ===")
    print(scores.round(4).to_string(index=False))

    print(f"\nTraining-derived high-load threshold: {high_threshold:.3f}")
    print(f"High-load timestamps: {int(high_mask.sum())}")

    print("\n=== Safety-margin experiment ===")
    print(margin_scores.round(4).to_string(index=False))

    if missing_indices:
        print(
            "\nWARNING: Some holdout timestamps have no predictions. "
            "Metrics apply only to covered timestamps."
        )
        print("First missing indices:", missing_indices[:20])
    else:
        print("\nAll holdout timestamps are covered exactly once.")

    print("\nSaved outputs:")
    print(" - tft_accuracy_metrics.csv")
    print(" - tft_safety_margin_results.csv")
    print(" - tft_timestamp_level_predictions.csv")
    print(" - tft_accuracy_comparison.png")
    print(" - tft_holdout_predictions.npz")
    print(" - tft_eval_state_dict.pt")


if __name__ == "__main__":
    main()
