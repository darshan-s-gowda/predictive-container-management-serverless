from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "azure_serverless_workload.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_FRACTION = 0.80
ENCODER_LENGTH = 36
MAX_EPOCHS = 15
BATCH_SIZE = 64

MODEL_PATH = OUTPUT_DIR / "tft_azure_log_state_dict.pt"
PREDICTIONS_PATH = OUTPUT_DIR / "tft_azure_log_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "tft_azure_log_metrics.csv"
PLOT_PATH = OUTPUT_DIR / "tft_azure_log_forecast.png"

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def find_column(df, candidates, description):
    columns = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]
    raise ValueError(
        f"Cannot find {description} column.\nAvailable columns: {list(df.columns)}"
    )


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if len(actual) == 0:
        return np.nan, np.nan
    error = actual - predicted
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error ** 2)))


def unpack_prediction_result(result):
    """Support common PyTorch Forecasting prediction return formats."""
    if hasattr(result, "output") and hasattr(result, "index"):
        output, index = result.output, result.index
    elif isinstance(result, (tuple, list)):
        if len(result) < 2:
            raise ValueError(f"Expected prediction and index; got {len(result)} items.")
        output, index = result[0], result[1]
    else:
        raise TypeError(f"Unexpected prediction result type: {type(result).__name__}")

    if isinstance(output, dict):
        if "prediction" not in output:
            raise KeyError(f"Unexpected prediction keys: {list(output.keys())}")
        output = output["prediction"]

    if isinstance(index, pd.DataFrame):
        index_df = index.copy()
    elif isinstance(index, dict):
        index_df = pd.DataFrame(index)
    else:
        raise TypeError(f"Unexpected prediction index type: {type(index).__name__}")

    if "time_idx" not in index_df.columns:
        raise KeyError(f"No time_idx in prediction index. Columns: {list(index_df.columns)}")

    if isinstance(output, torch.Tensor):
        predictions = output.detach().cpu().numpy()
    else:
        predictions = np.asarray(output)
    predictions = predictions.reshape(-1)
    times = pd.to_numeric(index_df["time_idx"], errors="coerce").to_numpy()

    if len(predictions) != len(times):
        raise ValueError(
            f"Prediction count ({len(predictions)}) != timestamp count ({len(times)}). "
            "This experiment expects one-step forecasts."
        )
    return predictions, times


def main():
    seed_everything(SEED, workers=True)
    print("Loading Azure workload...")
    raw = pd.read_csv(DATA_PATH)

    time_col = find_column(raw, ["time_idx", "time_index", "timestamp_idx"], "time index")
    load_col = find_column(
        raw, ["invocation_rate", "invocations", "invocation_count", "request_count", "load"],
        "invocation workload"
    )
    duration_col = find_column(
        raw, ["mean_duration_seconds", "mean_duration", "duration_seconds"], "mean duration"
    )

    df = raw[[time_col, load_col, duration_col]].copy()
    df.columns = ["time_idx", "invocation_rate", "mean_duration_seconds"]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = (
        df.dropna().sort_values("time_idx")
        .drop_duplicates("time_idx", keep="last").reset_index(drop=True)
    )

    if len(df) < 100:
        raise ValueError("Too few valid workload rows to train the model.")
    if (df["invocation_rate"] < 0).any():
        raise ValueError("Invocation workload contains negative values.")

    expected = np.arange(int(df["time_idx"].iloc[0]), int(df["time_idx"].iloc[0]) + len(df))
    if not np.array_equal(df["time_idx"].to_numpy(), expected):
        raise ValueError("time_idx must be consecutive with no gaps; check preprocessing output.")

    day_phase = 2 * np.pi * (df["time_idx"] % 288) / 288
    week_phase = 2 * np.pi * (df["time_idx"] % 2016) / 2016
    df["day_sin"], df["day_cos"] = np.sin(day_phase), np.cos(day_phase)
    df["week_sin"], df["week_cos"] = np.sin(week_phase), np.cos(week_phase)
    df["log_invocations"] = np.log1p(df["invocation_rate"])
    df["series"] = "azure"

    split = int(len(df) * TRAIN_FRACTION)
    train_df = df.iloc[:split].copy()
    validation_df = df.iloc[split - ENCODER_LENGTH:].copy()
    if len(train_df) <= ENCODER_LENGTH:
        raise ValueError("Training split is too short for the encoder.")

    high_load_threshold = float(train_df["invocation_rate"].quantile(0.80))
    print(f"Total buckets: {len(df)}")
    print(f"Training buckets: {len(train_df)}")
    print(f"Holdout buckets: {len(df) - split}")
    print(f"Encoder length: {ENCODER_LENGTH}")
    print("Forecast horizon: 1 bucket (5 minutes)")
    print(f"Training-derived high-load threshold: {high_load_threshold:.2f}")

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="log_invocations",
        group_ids=["series"],
        min_encoder_length=ENCODER_LENGTH,
        max_encoder_length=ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=["series"],
        time_varying_known_reals=[
            "time_idx", "day_sin", "day_cos", "week_sin", "week_cos"
        ],
        time_varying_unknown_reals=["log_invocations", "mean_duration_seconds"],
        target_normalizer=GroupNormalizer(groups=["series"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, validation_df, predict=False, stop_randomization=True
    )
    train_loader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    print("\nBuilding log-transformed TFT...")
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=16,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=MAE(),
        output_size=1,
        log_interval=-1,
        reduce_on_plateau_patience=3,
    )
    early_stop = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=3, mode="min")
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stop],
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=True,
        enable_progress_bar=True,
    )

    print("\nTraining model...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    torch.save(model.state_dict(), MODEL_PATH)

    print("\nGenerating holdout predictions...")
    result = model.predict(val_loader, mode="prediction", return_index=True)
    predictions, prediction_times = unpack_prediction_result(result)
    predicted_counts = np.maximum(np.expm1(predictions), 0)

    predicted_df = pd.DataFrame({
        "time_idx": prediction_times.astype(int),
        "log_tft_prediction": predictions,
        "log_tft_predicted_invocations": predicted_counts,
    })
    actual_df = df.iloc[split:][["time_idx", "invocation_rate"]].rename(
        columns={"invocation_rate": "actual_invocations"}
    )
    predicted_df = (
        predicted_df.drop_duplicates("time_idx", keep="first")
        .merge(actual_df, on="time_idx", how="inner")
        .sort_values("time_idx").reset_index(drop=True)
    )

    previous = df[["time_idx", "invocation_rate"]].copy()
    previous["time_idx"] += 1
    previous = previous.rename(columns={"invocation_rate": "persistence_prediction"})
    predicted_df = predicted_df.merge(previous, on="time_idx", how="left")
    predicted_df["high_load"] = predicted_df["actual_invocations"] >= high_load_threshold

    if predicted_df.empty:
        raise ValueError("No holdout predictions matched actual timestamps.")
    if len(predicted_df) != len(actual_df):
        print(f"WARNING: matched {len(predicted_df)} of {len(actual_df)} holdout timestamps.")

    predicted_df.to_csv(PREDICTIONS_PATH, index=False)

    metric_rows = []
    for model_name, column in [
        ("Log-transformed TFT", "log_tft_predicted_invocations"),
        ("Persistence", "persistence_prediction"),
    ]:
        for segment, subset in [
            ("Overall", predicted_df),
            ("High load", predicted_df[predicted_df["high_load"]]),
            ("Other load", predicted_df[~predicted_df["high_load"]]),
        ]:
            valid = subset.dropna(subset=["actual_invocations", column])
            mae, rmse = calculate_metrics(valid["actual_invocations"], valid[column])
            metric_rows.append({
                "model": model_name, "segment": segment,
                "MAE": mae, "RMSE": rmse, "count": len(valid)
            })

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(METRICS_PATH, index=False)

    print("\n=== High-load underprediction ===")
    high = predicted_df[predicted_df["high_load"]]
    for model_name, column in [
        ("Log-transformed TFT", "log_tft_predicted_invocations"),
        ("Persistence", "persistence_prediction"),
    ]:
        valid = high.dropna(subset=[column, "actual_invocations"])
        if len(valid):
            under_pct = (valid["actual_invocations"] > valid[column]).mean() * 100
            print(f"{model_name}: {under_pct:.2f}% ({len(valid)} high-load timestamps)")
        else:
            print(f"{model_name}: no valid high-load predictions")

    spike_position = predicted_df["actual_invocations"].idxmax()
    start = max(0, spike_position - 288)
    end = min(len(predicted_df), spike_position + 2 * 288)
    window = predicted_df.iloc[start:end]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(window["time_idx"], window["actual_invocations"], label="Actual workload", linewidth=1.5)
    ax.plot(window["time_idx"], window["log_tft_predicted_invocations"], label="Log-transformed TFT", linewidth=1.2)
    ax.plot(window["time_idx"], window["persistence_prediction"], label="Persistence", alpha=0.8)
    ax.axhline(high_load_threshold, linestyle="--", label=f"High-load threshold ({high_load_threshold:.0f})")
    ax.set_title("Azure one-step-ahead forecasting: log-TFT vs persistence")
    ax.set_xlabel("Elapsed 5-minute bucket index")
    ax.set_ylabel("Invocations per bucket")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)

    print("\n=== Forecast metrics ===")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nSaved files:")
    print(MODEL_PATH)
    print(PREDICTIONS_PATH)
    print(METRICS_PATH)
    print(PLOT_PATH)


if __name__ == "__main__":
    main()
