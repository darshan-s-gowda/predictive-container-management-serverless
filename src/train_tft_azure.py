
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lightning.pytorch import Trainer
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "azure_serverless_workload.csv"
OUT_DIR = ROOT / "outputs"

ENCODER_LENGTH = 36
PREDICTION_LENGTH = 12
EPOCHS = 10
BATCH_SIZE = 64


def prepare_data():
    df = pd.read_csv(DATA_PATH)

    required = {"time_idx", "invocation_rate", "mean_duration_seconds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.sort_values("time_idx").reset_index(drop=True)

    if df.isna().any().any():
        raise ValueError("Dataset contains missing values.")

    if (df["invocation_rate"] < 0).any():
        raise ValueError("Invocation counts cannot be negative.")

    # Cyclical elapsed-time features. These are relative phases,
    # not claims about the original calendar time or weekday.
    t = df["time_idx"].to_numpy()
    df["time_of_day_sin"] = np.sin(2 * np.pi * (t % 288) / 288)
    df["time_of_day_cos"] = np.cos(2 * np.pi * (t % 288) / 288)
    df["weekly_phase_sin"] = np.sin(2 * np.pi * (t % 2016) / 2016)
    df["weekly_phase_cos"] = np.cos(2 * np.pi * (t % 2016) / 2016)
    df["series"] = "azure_functions"

    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    df = prepare_data()

    split_idx = int(len(df) * 0.8)
    train_df = df[df["time_idx"] < split_idx].copy()

    if len(train_df) <= ENCODER_LENGTH + PREDICTION_LENGTH:
        raise ValueError("Not enough training rows for the selected windows.")

    print(f"Azure rows: {len(df)}")
    print(f"Training rows: {len(train_df)}")
    print(f"Holdout rows: {len(df) - split_idx}")
    print(f"Training cutoff index: {split_idx}")

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

    loader = training.to_dataloader(
        train=True,
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

    trainer = Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        deterministic=True,
    )

    print("\nTraining TFT on real Azure invocation counts...")
    trainer.fit(model, loader)

    model_path = OUT_DIR / "tft_azure_state_dict.pt"
    torch.save(model.state_dict(), model_path)

    print(f"\nSaved trained model: {model_path}")
    print("Next run: python src/evaluate_tft_azure.py")


if __name__ == "__main__":
    main()
