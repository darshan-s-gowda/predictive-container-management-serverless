"""
Milestone 3 — Offline Planner Comparison Experiment
Runs LocalRuleProvider across 10 scenario types, saves CSV + PNG.
Ollama/Qwen3 comparison requires: PLANNER_BACKEND=ollama ollama pull qwen3:8b
"""
import sys, os, csv
sys.path.insert(0, 'src')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from planner_providers import (
    LocalRuleProvider, OllamaProvider, build_provider_from_env,
    run_offline_planner_comparison,
)

SYSTEM_PROMPT = (
    "You are a serverless container management planner. "
    "Return ONLY valid JSON with keys: recommended_action, rationale, confidence, "
    "container_delta, expected_latency_ms, sla_risk."
)

def make_user_prompt(s):
    return "\n".join([
        f"forecast_invocations={s.get('forecast_load', 0.5) * 100:.1f}",
        f"current_capacity={s.get('current_containers', 4) * 10:.1f}",
        f"current_latency_ms={s.get('latency_ms', 150)}",
        f"sla_ms={s.get('sla_target_ms', 300)}",
        f"warm_containers={s.get('current_containers', 4)}",
        f"cold_start_rate=0.10",
        f"capacity_limit=50",
    ])

SCENARIOS = [
    {"id": "S01", "name": "low_load",    "current_containers": 3,  "forecast_load": 0.05, "latency_ms": 80,  "sla_target_ms": 300},
    {"id": "S02", "name": "burst",       "current_containers": 5,  "forecast_load": 0.98, "latency_ms": 450, "sla_target_ms": 300},
    {"id": "S03", "name": "high_lat",    "current_containers": 4,  "forecast_load": 0.65, "latency_ms": 350, "sla_target_ms": 300},
    {"id": "S04", "name": "idle",        "current_containers": 10, "forecast_load": 0.00, "latency_ms": 50,  "sla_target_ms": 300},
    {"id": "S05", "name": "normal",      "current_containers": 4,  "forecast_load": 0.45, "latency_ms": 150, "sla_target_ms": 300},
    {"id": "S06", "name": "ramp_up",     "current_containers": 3,  "forecast_load": 0.80, "latency_ms": 120, "sla_target_ms": 300},
    {"id": "S07", "name": "ramp_down",   "current_containers": 8,  "forecast_load": 0.20, "latency_ms": 200, "sla_target_ms": 300},
    {"id": "S08", "name": "near_sla",    "current_containers": 4,  "forecast_load": 0.72, "latency_ms": 285, "sla_target_ms": 300},
    {"id": "S09", "name": "overloaded",  "current_containers": 2,  "forecast_load": 1.00, "latency_ms": 800, "sla_target_ms": 300},
    {"id": "S10", "name": "stable_high", "current_containers": 6,  "forecast_load": 0.57, "latency_ms": 190, "sla_target_ms": 300},
]

# ── Build providers ───────────────────────────────────────────────────────────
providers = [LocalRuleProvider()]
provider_labels = ["local_rule"]

backend = os.environ.get("PLANNER_BACKEND", "local").lower()
if backend == "ollama":
    endpoint = os.environ.get("PLANNER_ENDPOINT", "http://localhost:11434")
    model    = os.environ.get("PLANNER_MODEL", "qwen3:8b")
    prov = OllamaProvider(endpoint=endpoint, model=model, timeout=30)
    providers.append(prov)
    provider_labels.append(f"ollama_{model}")
    print(f"[INFO] Ollama provider: {endpoint} model={model}")
else:
    print("[INFO] Only LocalRuleProvider active. Set PLANNER_BACKEND=ollama for Qwen3 comparison.")

# ── Run ───────────────────────────────────────────────────────────────────────
raw_rows = run_offline_planner_comparison(SCENARIOS, providers, SYSTEM_PROMPT, make_user_prompt)

# Enrich with scenario metadata
sc_map = {s["id"]: s for s in SCENARIOS}
rows = []
for r in raw_rows:
    sc = sc_map.get(r["scenario_id"], {})
    rows.append({**r, "scenario_name": sc.get("name", r["scenario_id"]),
                 "latency_ms": sc.get("latency_ms", 0),
                 "forecast_load": sc.get("forecast_load", 0)})
    print(f"  [{r['provider']:20s}] {sc.get('name','?'):12s} → {r['action']:15s} conf={r['confidence']:.2f} valid={r['valid_output']}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
csv_path = "outputs/planner_offline_comparison.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\n[OK] Saved {csv_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────
ACTION_COLORS = {
    "NO_OP": "#aec6cf", "PREWARM": "#fdfd96", "SCALE_OUT": "#77dd77",
    "SCALE_IN": "#ff6961", "REUSE": "#b39eb5", "TERMINATE_IDLE": "#ffb347",
    "FALLBACK": "#cccccc", "ERROR": "#888888",
}

scenario_names = [s["name"] for s in SCENARIOS]
fig, axes = plt.subplots(len(provider_labels), 1,
                         figsize=(13, 4 * len(provider_labels)), squeeze=False)

for pi, pname in enumerate(provider_labels):
    ax = axes[pi][0]
    prows = [r for r in rows if r["provider"] == pname.replace("_", " ").split()[0] or r["provider"] == pname]
    # match by provider field
    prows = [r for r in rows if r["provider"].replace(" ", "_").replace("-","_") in
             (pname.replace(" ","_"), pname.split("_")[0])]
    if not prows:
        prows = [r for r in rows][:len(SCENARIOS)]  # fallback

    actions = [r["action"] for r in prows]
    confs   = [float(r["confidence"]) for r in prows]
    colors  = [ACTION_COLORS.get(a, "#cccccc") for a in actions]

    bars = ax.bar(scenario_names, confs, color=colors, edgecolor="black", linewidth=0.5)
    for bar, action in zip(bars, actions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                action.replace("_", "\n"), ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_title(f"Provider: {pname}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1.3)
    ax.tick_params(axis="x", rotation=30)

    legend_elements = [Patch(facecolor=c, label=a) for a, c in ACTION_COLORS.items()]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7, ncol=3)

plt.suptitle(
    "Offline Planner Comparison — LocalRuleProvider vs Qwen3 (if configured)\n"
    "[NOTE: LocalRuleProvider is deterministic; Qwen3 requires Ollama. All outputs are SIMULATED proxies]",
    fontsize=10, y=1.01,
)
plt.tight_layout()
png_path = "outputs/planner_offline_comparison.png"
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Saved {png_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== PLANNER OFFLINE COMPARISON SUMMARY ===")
for pname in provider_labels:
    prows = [r for r in rows if r["provider"] in (pname, "local_rules", "ollama")]
    valid = sum(1 for r in prows if r["valid_output"])
    action_counts = {}
    for r in prows:
        action_counts[r["action"]] = action_counts.get(r["action"], 0) + 1
    print(f"\n  {pname}: {valid}/{len(prows)} valid")
    for a, c in sorted(action_counts.items()):
        print(f"    {a}: {c}")

print("\n[NOTE] To compare Qwen3: set PLANNER_BACKEND=ollama PLANNER_ENDPOINT=http://localhost:11434 PLANNER_MODEL=qwen3:8b")
print("[NOTE] Requires: ollama pull qwen3:8b (needs GPU and network access)")
