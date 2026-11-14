
from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_serverless_workload.csv"
WEIGHTS = ROOT / "outputs" / "tft_state_dict.pt"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# Re-create the dataset configuration used during training
df = pd.read_csv(DATA, parse_dates=["timestamp"])
df["time_idx"] = range(len(df))
df["series"] = 0

cutoff = df["time_idx"].max() - 48

train_ds = TimeSeriesDataSet(
    df[df.time_idx <= cutoff],
    time_idx="time_idx",
    target="invocation_rate",
    group_ids=["series"],
    min_encoder_length=36,
    max_encoder_length=36,
    min_prediction_length=12,
    max_prediction_length=12,
    time_varying_known_reals=["time_idx", "hour", "day_of_week"],
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

# Rebuild the same TFT architecture
model = TemporalFusionTransformer.from_dataset(
    train_ds,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,
)

# Load the trained weights
state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

# Predict the next 12 time steps from the latest available history
predict_ds = TimeSeriesDataSet.from_dataset(
    train_ds,
    df,
    predict=True,
    stop_randomization=True,
)
loader = predict_ds.to_dataloader(
    train=False, batch_size=64, num_workers=0
)

with torch.no_grad():
    result = model.predict(loader, mode="quantiles")

pred = result[0].detach().cpu().numpy()
median_index = 3  # median of the seven configured quantiles
forecast = pred[:, median_index]

future_idx = range(df["time_idx"].max() + 1,
                   df["time_idx"].max() + 13)

result_df = pd.DataFrame({
    "time_idx": list(future_idx),
    "predicted_invocation_rate": forecast,
})
result_df.to_csv(OUT / "tft_forecast.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(
    df["time_idx"].tail(60),
    df["invocation_rate"].tail(60),
    label="Observed workload",
)
plt.plot(
    result_df["time_idx"],
    result_df["predicted_invocation_rate"],
    marker="o",
    label="TFT median forecast",
)
plt.xlabel("Time index")
plt.ylabel("Invocation rate")
plt.title("Trained TFT Workload Forecast")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "tft_forecast.png", dpi=150)
plt.close()

print("Loaded trained TFT weights successfully.")
print(result_df.to_string(index=False))
print(f"Saved forecast to: {OUT / 'tft_forecast.csv'}")
print(f"Saved graph to: {OUT / 'tft_forecast.png'}")
