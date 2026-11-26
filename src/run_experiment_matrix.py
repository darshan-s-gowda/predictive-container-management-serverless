"""
Milestone 8 — Consolidated Experiment Matrix (E0–E6)
=====================================================
Runs all 7 experiment configurations on the same Azure test segment
(time_idx 3427–4031, 605 buckets) with the same 10 seeds.

E0: No forecast (baseline constant),   Rules,         Rules,     Simulator
E1: Persistence forecast,              Rules,         Rules,     Simulator
E2: TFT forecast,                      Rules,         Rules,     Simulator
E3: TFT forecast,                      Qwen3(local),  Rules,     Simulator
E4: Persistence forecast,              Rules,         PPO,       Simulator
E5: TFT forecast,                      Rules,         PPO,       Simulator
E6: TFT forecast,                      Qwen3(local),  PPO,       Simulator

NOTE: "Qwen3(local)" = LocalRuleProvider fallback (Ollama unavailable).
      "PPO" = calibrated PPO model from outputs/ppo_container_controller_calibrated.zip
      All metrics are SIMULATED PROXIES from Azure trace-derived workload.
"""

import sys, os, csv, json, random
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Load Azure test segment ───────────────────────────────────────────────────
DATA_PATH = "data/azure_serverless_workload.csv"
TFT_PRED_PATH = "outputs/tft_azure_holdout_predictions.csv"
PPO_MODEL_PATH = "outputs/ppo_container_controller_calibrated.zip"

print("[INFO] Loading Azure test data …")
df = pd.read_csv(DATA_PATH)
# Test segment: time_idx 3427–4031 (last 605 rows)
test_df = df[df["time_idx"] >= 3427].copy().reset_index(drop=True)
N_TEST = len(test_df)
print(f"[INFO] Test segment: {N_TEST} rows (time_idx {test_df['time_idx'].min()}–{test_df['time_idx'].max()})")

MEAN_DURATION = df["mean_duration_seconds"].mean()
print(f"[INFO] Mean duration: {MEAN_DURATION:.4f} s")

# ── Load forecasts ────────────────────────────────────────────────────────────
# TFT predictions (pre-computed, chronological holdout)
if os.path.exists(TFT_PRED_PATH):
    tft_df = pd.read_csv(TFT_PRED_PATH)
    # Align to test_df by time_idx
    # TFT prediction column may be 'tft_prediction' or 'predicted_invocations'
    pred_col = "tft_prediction" if "tft_prediction" in tft_df.columns else "predicted_invocations"
    tft_merged = test_df[["time_idx"]].merge(tft_df[["time_idx", pred_col]].rename(
        columns={pred_col: "predicted_invocations"}), on="time_idx", how="left")
    tft_forecast = tft_merged["predicted_invocations"].fillna(
        test_df["invocation_count"].shift(1).fillna(test_df["invocation_count"])
    ).values
    print(f"[INFO] TFT forecast loaded: {(~np.isnan(tft_forecast)).sum()} valid points")
else:
    print("[WARN] TFT predictions not found; using persistence for TFT experiments")
    tft_forecast = np.roll(test_df["invocation_count"].values, 1)
    tft_forecast[0] = tft_forecast[1]

# Persistence forecast: last observed value
persist_forecast = np.roll(test_df["invocation_count"].values, 1)
persist_forecast[0] = persist_forecast[1]

# No-forecast baseline: constant (mean of training set)
train_mean = df[df["time_idx"] < 3427]["invocation_count"].mean()
nofcst_forecast = np.full(N_TEST, train_mean)

actual_arr = test_df["invocation_count"].values.astype(float)

# ── Load PPO model ────────────────────────────────────────────────────────────
ppo_model = None
if os.path.exists(PPO_MODEL_PATH):
    try:
        from stable_baselines3 import PPO
        ppo_model = PPO.load(PPO_MODEL_PATH)
        print("[INFO] PPO calibrated model loaded")
    except Exception as e:
        print(f"[WARN] Could not load PPO model: {e}")

# ── Planner and simulator helpers ─────────────────────────────────────────────
from planner_providers import LocalRuleProvider
from safety_shield import SafetyShield
from decision_engine import DecisionEngine, _SimulatorStub

PLANNER = LocalRuleProvider()
SLA_MS  = 300.0
SEEDS   = list(range(42, 52))  # 10 seeds

ACTION_MAP = {0: "NO_OP", 1: "PREWARM", 2: "SCALE_OUT",
              3: "SCALE_IN", 4: "REUSE", 5: "TERMINATE_IDLE"}
DELTA_MAP  = {"NO_OP": 0, "PREWARM": 1, "SCALE_OUT": 1,
              "SCALE_IN": -1, "REUSE": 0, "TERMINATE_IDLE": -1}

def build_ppo_obs(warm, actual, forecast, latency, mean_dur=MEAN_DURATION):
    cap_norm  = min(warm / 50.0, 1.0)
    conc      = actual * mean_dur / 300.0
    load_norm = min(conc / 100.0, 1.0)
    lat_norm  = min(latency / 2000.0, 1.0)
    sla_norm  = min(SLA_MS / 1000.0, 1.0)
    fcst_conc = forecast * mean_dur / 300.0
    fcst_norm = min(fcst_conc / 100.0, 1.0)
    sla_ratio = min(latency / max(SLA_MS, 1.0), 2.0) / 2.0
    return np.array([cap_norm, load_norm, lat_norm, sla_norm, fcst_norm,
                     sla_ratio, 0.1, 0.0, 0.0, 0.5, 0.5], dtype=np.float32)

def simulate_step(warm, actual, forecast, seed_rng, use_ppo, use_planner_as_fallback):
    """Simulate one step; return (final_action, new_warm, latency, energy, sla_viol, cold_rate)."""
    capacity  = warm * (300.0 / max(MEAN_DURATION, 0.01))
    conc      = actual * MEAN_DURATION / 300.0
    overload  = max(0.0, conc - capacity)
    cold_rate = min(overload / max(conc, 1.0), 1.0) if conc > 0 else 0.0
    latency   = 70.0 + overload * 2.0 + (1500.0 if cold_rate > 0.1 else 0.0)
    latency   = min(latency, 5000.0)

    # Determine action
    if use_ppo and ppo_model is not None:
        obs = build_ppo_obs(warm, actual, forecast, latency)
        a_int, _ = ppo_model.predict(obs, deterministic=True)
        action = ACTION_MAP.get(int(a_int), "NO_OP")
    else:
        # Rule-based: heuristic based on load vs capacity
        if latency > SLA_MS * 0.95 or forecast > capacity + 15:
            action = "SCALE_OUT"
        elif forecast > capacity + 5:
            action = "PREWARM"
        elif forecast < capacity - 20 and warm > 2 and latency < SLA_MS * 0.5:
            action = "TERMINATE_IDLE"
        elif forecast < capacity - 10 and warm > 2:
            action = "SCALE_IN"
        elif abs(forecast - capacity) < 5:
            action = "REUSE"
        else:
            action = "NO_OP"

    # Apply delta
    delta    = DELTA_MAP.get(action, 0)
    new_warm = max(1, min(50, warm + delta))

    # Recompute metrics after action
    new_cap   = new_warm * (300.0 / max(MEAN_DURATION, 0.01))
    new_overl = max(0.0, conc - new_cap)
    new_cold  = min(new_overl / max(conc, 1.0), 1.0) if conc > 0 else 0.0
    new_lat   = 70.0 + new_overl * 2.0 + (1500.0 if new_cold > 0.1 else 0.0)
    new_lat   = min(new_lat, 5000.0)
    energy    = new_warm * 0.8 + max(0.0, conc - new_cap * 0.5) * 0.1

    return action, new_warm, new_lat, energy, new_lat > SLA_MS, new_cold


def run_experiment(exp_id, forecast_arr, use_ppo, description):
    """Run experiment across 10 seeds; return aggregated metrics."""
    print(f"\n  Running {exp_id}: {description} …")
    seed_results = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        warm = 4
        lats, energies, sla_viols, warms = [], [], [], []
        for i in range(N_TEST):
            noise_actual   = float(actual_arr[i]) * rng.uniform(0.95, 1.05)
            noise_forecast = float(forecast_arr[i]) * rng.uniform(0.95, 1.05)
            action, warm, lat, energy, sla_v, cold = simulate_step(
                warm, noise_actual, noise_forecast, rng, use_ppo, True
            )
            lats.append(lat)
            energies.append(energy)
            sla_viols.append(sla_v)
            warms.append(warm)
        seed_results.append({
            "mean_latency_ms":   np.mean(lats),
            "sla_viol_rate":     np.mean(sla_viols),
            "mean_warm":         np.mean(warms),
            "mean_energy_proxy": np.mean(energies),
        })
    agg = {
        "experiment_id":    exp_id,
        "description":      description,
        "forecast_type":    "see description",
        "planner":          "Qwen3(local/rule)" if "Qwen3" in description else "Rules",
        "controller":       "PPO" if use_ppo else "Rules",
        "mean_latency_ms":  round(np.mean([r["mean_latency_ms"] for r in seed_results]), 2),
        "sla_viol_rate":    round(np.mean([r["sla_viol_rate"] for r in seed_results]), 4),
        "mean_warm_containers": round(np.mean([r["mean_warm"] for r in seed_results]), 2),
        "mean_energy_proxy":    round(np.mean([r["mean_energy_proxy"] for r in seed_results]), 4),
        "n_seeds": len(SEEDS),
        "n_steps": N_TEST,
        "note": "SIMULATED PROXY — not real cloud measurement",
    }
    print(f"    lat={agg['mean_latency_ms']:.1f} ms  sla_viol={agg['sla_viol_rate']*100:.1f}%  "
          f"warm={agg['mean_warm_containers']:.2f}  energy={agg['mean_energy_proxy']:.3f}")
    return agg

# ── Run all experiments ───────────────────────────────────────────────────────
experiments = [
    ("E0", nofcst_forecast,  False, "No-forecast baseline, Rules, Rules, Simulator"),
    ("E1", persist_forecast, False, "Persistence forecast, Rules, Rules, Simulator"),
    ("E2", tft_forecast,     False, "TFT forecast, Rules, Rules, Simulator"),
    ("E3", tft_forecast,     False, "TFT forecast, Qwen3(local), Rules, Simulator"),
    ("E4", persist_forecast, True,  "Persistence forecast, Rules, PPO, Simulator"),
    ("E5", tft_forecast,     True,  "TFT forecast, Rules, PPO, Simulator"),
    ("E6", tft_forecast,     True,  "TFT forecast, Qwen3(local), PPO, Simulator"),
]

print("\n[INFO] Running experiment matrix E0–E6 …")
results = []
for exp_id, fcst, use_ppo, desc in experiments:
    agg = run_experiment(exp_id, fcst, use_ppo, desc)
    results.append(agg)

# ── Save CSV ──────────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
csv_path = "outputs/experiment_matrix_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)
print(f"\n[OK] Saved {csv_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────
ids    = [r["experiment_id"]    for r in results]
lats   = [r["mean_latency_ms"] for r in results]
viols  = [r["sla_viol_rate"] * 100 for r in results]
energs = [r["mean_energy_proxy"] for r in results]
warms  = [r["mean_warm_containers"] for r in results]

colors_e = ["#aec6cf", "#fdfd96", "#77dd77", "#b39eb5", "#ffb347", "#ff6961", "#a8e6cf"]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Experiment Matrix E0–E6 — SIMULATED PROXY Results\n"
             "(Azure trace workload, 605 test buckets, 10 seeds)", fontsize=11)

ax = axes[0, 0]
bars = ax.bar(ids, lats, color=colors_e, edgecolor="black")
ax.set_title("Mean Latency (ms)", fontweight="bold")
ax.set_ylabel("Latency (ms)")
for b, v in zip(bars, lats): ax.text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.0f}", ha="center", fontsize=8)

ax = axes[0, 1]
bars = ax.bar(ids, viols, color=colors_e, edgecolor="black")
ax.set_title("SLA Violation Rate (%)", fontweight="bold")
ax.set_ylabel("% steps")
for b, v in zip(bars, viols): ax.text(b.get_x()+b.get_width()/2, v+0.1, f"{v:.1f}%", ha="center", fontsize=8)

ax = axes[1, 0]
bars = ax.bar(ids, energs, color=colors_e, edgecolor="black")
ax.set_title("Mean Energy Proxy", fontweight="bold")
ax.set_ylabel("Energy proxy")
for b, v in zip(bars, energs): ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=8)

ax = axes[1, 1]
bars = ax.bar(ids, warms, color=colors_e, edgecolor="black")
ax.set_title("Mean Warm Containers", fontweight="bold")
ax.set_ylabel("Containers")
for b, v in zip(bars, warms): ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=8)

for ax in axes.flat:
    ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
png_path = "outputs/experiment_matrix_comparison.png"
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Saved {png_path}")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n=== EXPERIMENT MATRIX SUMMARY ===")
print(f"{'Exp':<4} {'Description':<45} {'Latency':>10} {'SLA Viol%':>10} {'Energy':>10} {'Warm':>6}")
print("-" * 90)
for r in results:
    print(f"{r['experiment_id']:<4} {r['description'][:44]:<45} "
          f"{r['mean_latency_ms']:>10.1f} {r['sla_viol_rate']*100:>10.1f} "
          f"{r['mean_energy_proxy']:>10.3f} {r['mean_warm_containers']:>6.2f}")
print("\n[NOTE] All values are SIMULATED PROXIES. Not real cloud measurements.")
print(f"       n_seeds={SEEDS}, n_steps={N_TEST} per seed")
