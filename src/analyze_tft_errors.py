
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

prediction_file = OUT / "tft_holdout_predictions.npz"

if not prediction_file.exists():
    raise FileNotFoundError(
        f"Missing {prediction_file}. Run evaluate_tft.py first."
    )

data = np.load(prediction_file)

actual = np.asarray(data["actual"]).reshape(-1)
predicted = np.asarray(data["predicted"]).reshape(-1)
baseline = np.asarray(data["baseline"]).reshape(-1)

results = pd.DataFrame({
    "actual": actual,
    "tft_prediction": predicted,
    "baseline_prediction": baseline,
})

results["tft_abs_error"] = (
    results["actual"] - results["tft_prediction"]
).abs()

results["baseline_abs_error"] = (
    results["actual"] - results["baseline_prediction"]
).abs()

threshold = results["actual"].quantile(0.75)

results["load_group"] = np.where(
    results["actual"] >= threshold,
    "High load",
    "Other load",
)

print(f"High-load threshold: {threshold:.3f}")
print("\n=== Error by load group ===")

summary = results.groupby("load_group")[
    ["tft_abs_error", "baseline_abs_error"]
].mean()

print(summary.round(3).to_string())

results.to_csv(OUT / "tft_error_analysis.csv", index=False)

n = min(240, len(results))

plt.figure(figsize=(12, 5))
plt.plot(results["actual"].iloc[:n], label="Actual")
plt.plot(
    results["tft_prediction"].iloc[:n],
    label="TFT prediction",
)
plt.plot(
    results["baseline_prediction"].iloc[:n],
    label="Persistence baseline",
    linestyle="--",
)
plt.xlabel("Flattened forecast observation")
plt.ylabel("Invocation rate")
plt.title("TFT Holdout Error Analysis")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "tft_error_analysis.png", dpi=150)
plt.close()

print("\nSaved outputs/tft_error_analysis.csv")
print("Saved outputs/tft_error_analysis.png")
# Additional high-load diagnostics
high = results["actual"] >= threshold

if high.any():
    actual_high = results.loc[high, "actual"]
    pred_high = results.loc[high, "tft_prediction"]
    base_high = results.loc[high, "baseline_prediction"]

    print("\n=== High-load diagnostics ===")
    print(f"High-load observations: {high.sum()} / {len(results)}")
    print(
        "TFT mean signed error:",
        round((pred_high - actual_high).mean(), 3),
    )
    print(
        "Baseline mean signed error:",
        round((base_high - actual_high).mean(), 3),
    )
    print(
        "TFT underprediction rate:",
        round((pred_high < actual_high).mean() * 100, 1),
        "%",
    )
    print(
        "Baseline underprediction rate:",
        round((base_high < actual_high).mean() * 100, 1),
        "%",
    )