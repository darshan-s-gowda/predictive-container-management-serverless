"""
src/generate_replay_report.py
==============================
Milestone 5 — Step-by-Step Replay Viewer.

Loads a JSONL decision log, generates:
  1. outputs/decision_replay.html  — interactive HTML card-per-step viewer
  2. outputs/decision_replay_summary.csv
  3. outputs/decision_replay.png   — static multi-panel plot

Usage:
  python src/generate_replay_report.py [path/to/decision_log.jsonl]

If no path given, generates a fresh 60-step trace using LocalRuleProvider
and saves it first (useful for demo / CI).
"""

from __future__ import annotations
import json, os, sys, csv
sys.path.insert(0, 'src')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Helpers ────────────────────────────────────────────────────────────────────

ACTION_COLORS = {
    "NO_OP":         "#aec6cf",
    "PREWARM":       "#fdfd96",
    "SCALE_OUT":     "#77dd77",
    "SCALE_IN":      "#ff6961",
    "REUSE":         "#b39eb5",
    "TERMINATE_IDLE":"#ffb347",
    "N/A":           "#cccccc",
}

def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def generate_or_load(log_path: str | None) -> tuple[list[dict], str]:
    """Return (records, path). Generate demo trace if log_path is None."""
    if log_path and os.path.exists(log_path):
        return load_records(log_path), log_path

    print("[INFO] No log path given; generating 60-step demo trace …")
    from decision_engine import DecisionEngine, _SimulatorStub
    from planner_providers import LocalRuleProvider
    from safety_shield import SafetyShield

    rng = np.random.default_rng(42)
    # Simulate a workload: low → burst → decay → stable
    actual = np.concatenate([
        rng.uniform(5, 15, 15),
        rng.uniform(40, 80, 15),
        rng.uniform(20, 35, 15),
        rng.uniform(8, 18, 15),
    ]).tolist()
    forecast = [a * rng.uniform(0.9, 1.1) for a in actual]

    sim     = _SimulatorStub()
    planner = LocalRuleProvider()
    shield  = SafetyShield()
    engine  = DecisionEngine(sim, planner, shield, run_id="demo",
                              log_dir="outputs/decision_logs")
    engine.run_trace(actual, forecast, sla_target_ms=300.0, initial_containers=4)
    path = engine.log_path
    engine.close()
    return load_records(path), path

# ── HTML Report ───────────────────────────────────────────────────────────────

def build_html(records: list[dict]) -> str:
    SLA_MS = 300.0

    rows_html = []
    for r in records:
        step      = r.get("timestamp_step", 0)
        final     = r.get("final_action", "?")
        color     = ACTION_COLORS.get(final, "#eeeeee")
        lat       = r.get("latency_ms", 0.0)
        sla_ok    = "✅" if not r.get("sla_violation") else "❌"
        overridden = "⚠️ OVERRIDDEN" if r.get("shield_overridden") else ""
        fallback   = "⚠️ FALLBACK"   if r.get("planner_fallback") else ""

        rows_html.append(f"""
        <div class="card" style="border-left: 6px solid {color};">
          <div class="card-header">
            <span class="step-badge">Step {step}</span>
            <span class="action-badge" style="background:{color};">{final}</span>
            <span class="sla-badge">{sla_ok} Latency: {lat:.0f} ms / SLA {SLA_MS:.0f} ms</span>
            {"<span class='warn-badge'>" + overridden + "</span>" if overridden else ""}
            {"<span class='warn-badge'>" + fallback + "</span>"   if fallback   else ""}
          </div>
          <div class="card-body">
            <div class="col">
              <b>📊 Workload</b><br>
              Actual load: <code>{r.get('actual_load', 0):.2f}</code><br>
              Forecast:    <code>{r.get('forecast_load', 0):.2f}</code><br>
              Warm containers: <code>{r.get('warm_containers', '?')}</code><br>
              Energy proxy: <code>{r.get('energy_proxy', 0):.3f}</code><br>
              Cold-start proxy: <code>{r.get('cold_start_proxy', 0):.3f}</code>
            </div>
            <div class="col">
              <b>🤖 Planner ({r.get('planner_provider','?')})</b><br>
              Suggested: <code>{r.get('planner_action','?')}</code>
              (conf={r.get('planner_confidence', 0):.2f})<br>
              SLA risk: <code>{r.get('planner_sla_risk','?')}</code><br>
              Rationale: <i>{r.get('planner_rationale','')[:120]}</i>
            </div>
            <div class="col">
              <b>🛡️ Safety Shield</b><br>
              Input: <code>{r.get('shield_input_action','?')}</code><br>
              Approved: <code>{r.get('shield_approved','?')}</code><br>
              Rule: <code>{r.get('shield_reason_code','?')}</code><br>
              <small>{r.get('shield_details','')[:100]}</small>
            </div>
            <div class="col">
              <b>🎯 PPO Policy</b><br>
              Action: <code>{r.get('ppo_action_str','N/A')}</code>
              (int={r.get('ppo_action_int',-1)})<br>
              <b>🔑 Correlation ID</b><br>
              <small><code>{r.get('correlation_id','?')}</code></small>
            </div>
          </div>
        </div>""")

    n = len(records)
    sla_viol = sum(1 for r in records if r.get("sla_violation"))
    action_counts = {}
    for r in records:
        a = r.get("final_action", "?")
        action_counts[a] = action_counts.get(a, 0) + 1

    ac_html = "".join(
        f"<li><span style='background:{ACTION_COLORS.get(a,'#eee')};padding:2px 6px;"
        f"border-radius:3px;'>{a}</span>: {c} ({c/n*100:.1f}%)</li>"
        for a, c in sorted(action_counts.items(), key=lambda x: -x[1])
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Decision Replay Viewer</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background:#f5f7fa; color:#222; }}
  h1   {{ color:#2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
  .summary {{ background:#fff; padding:16px; border-radius:8px; margin-bottom:20px;
              box-shadow:0 1px 4px rgba(0,0,0,0.1); }}
  .card {{ background:#fff; border-radius:8px; margin-bottom:12px; padding:12px 16px;
           box-shadow:0 1px 3px rgba(0,0,0,0.08); }}
  .card-header {{ display:flex; align-items:center; gap:10px; margin-bottom:8px; flex-wrap:wrap; }}
  .step-badge   {{ background:#2c3e50; color:#fff; border-radius:4px; padding:2px 8px; font-weight:bold; }}
  .action-badge {{ border-radius:4px; padding:2px 10px; font-weight:bold; color:#333; }}
  .sla-badge    {{ color:#555; font-size:0.9em; }}
  .warn-badge   {{ background:#e74c3c; color:#fff; border-radius:4px; padding:2px 8px; font-size:0.85em; }}
  .card-body    {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; font-size:0.88em; }}
  .col          {{ background:#f8f9fa; border-radius:4px; padding:8px; }}
  code          {{ background:#edf2f7; padding:1px 4px; border-radius:3px; font-size:0.9em; }}
  ul            {{ list-style:none; padding:0; }}
  li            {{ margin-bottom:4px; }}
  .filter-bar   {{ margin-bottom:16px; }}
  input[type=text] {{ padding:6px 10px; border:1px solid #ccc; border-radius:4px; width:280px; font-size:1em; }}
  @media(max-width:800px) {{ .card-body {{ grid-template-columns:1fr 1fr; }} }}
  @media(max-width:500px) {{ .card-body {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<h1>🔍 Decision Replay Viewer</h1>
<div class="summary">
  <b>Trace summary</b>: {n} steps &nbsp;|&nbsp;
  SLA violations: {sla_viol} ({sla_viol/max(n,1)*100:.1f}%) &nbsp;|&nbsp;
  Avg latency: {sum(r.get('latency_ms',0) for r in records)/max(n,1):.1f} ms &nbsp;|&nbsp;
  Avg warm containers: {sum(r.get('warm_containers',0) for r in records)/max(n,1):.2f}<br><br>
  <b>Action distribution:</b>
  <ul style="display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;">{ac_html}</ul>
  <p style="font-size:0.8em;color:#888;">⚠️ All metrics are SIMULATED PROXIES derived from the Azure trace workload.
  Not real cloud measurements.</p>
</div>

<div class="filter-bar">
  <input type="text" id="searchBox" placeholder="Filter steps by action, risk, or text…"
         onkeyup="filterCards()" />
</div>

<div id="cards">
{''.join(rows_html)}
</div>

<script>
function filterCards() {{
  var val = document.getElementById('searchBox').value.toLowerCase();
  document.querySelectorAll('.card').forEach(function(c) {{
    c.style.display = val === '' || c.textContent.toLowerCase().includes(val) ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""

# ── Static plot ───────────────────────────────────────────────────────────────

def build_plot(records: list[dict], png_path: str):
    n      = len(records)
    steps  = [r.get("timestamp_step", i) for i, r in enumerate(records)]
    actual = [r.get("actual_load", 0.0)    for r in records]
    fcst   = [r.get("forecast_load", 0.0)  for r in records]
    warm   = [r.get("warm_containers", 4)  for r in records]
    lat    = [r.get("latency_ms", 0.0)     for r in records]
    sla_v  = [1 if r.get("sla_violation") else 0 for r in records]
    energy = [r.get("energy_proxy", 0.0)   for r in records]
    shield_override = [1 if r.get("shield_overridden") else 0 for r in records]

    actions = [r.get("final_action", "NO_OP") for r in records]
    action_counts = {}
    for a in actions:
        action_counts[a] = action_counts.get(a, 0) + 1

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    # Panel 1: Workload + containers
    ax = axes[0]
    ax.plot(steps, actual, label="Actual load", color="#3498db", linewidth=1.5)
    ax.plot(steps, fcst,   label="Forecast",    color="#e67e22", linewidth=1.2, linestyle="--")
    ax2 = ax.twinx()
    ax2.step(steps, warm, label="Warm containers", color="#27ae60", linewidth=1.2, where="post")
    ax2.set_ylabel("Warm containers", color="#27ae60")
    ax.set_ylabel("Invocations (normalized)")
    ax.set_title("Workload and Container Count", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    # Panel 2: Latency + SLA violations
    ax = axes[1]
    ax.plot(steps, lat, color="#8e44ad", linewidth=1.5, label="Latency ms")
    ax.axhline(300.0, color="red", linestyle="--", linewidth=1.0, label="SLA 300ms")
    sla_steps = [s for s, v in zip(steps, sla_v) if v]
    sla_lats  = [lat[i] for i, v in enumerate(sla_v) if v]
    ax.scatter(sla_steps, sla_lats, color="red", s=25, zorder=5, label="SLA violation")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency and SLA Violations", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 3: Energy proxy + shield overrides
    ax = axes[2]
    ax.fill_between(steps, energy, alpha=0.4, color="#f39c12", label="Energy proxy")
    ax.plot(steps, energy, color="#f39c12", linewidth=1.0)
    ax3 = ax.twinx()
    shield_steps = [s for s, v in zip(steps, shield_override) if v]
    ax3.scatter(shield_steps, [1] * len(shield_steps), color="red", marker="x", s=40,
                label="Shield override")
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["", "Override"])
    ax.set_ylabel("Energy proxy")
    ax.set_title("Energy Proxy and Safety Shield Overrides", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax3.legend(loc="upper right", fontsize=8)

    # Panel 4: Action frequency bar
    ax = axes[3]
    action_names = list(action_counts.keys())
    action_vals  = [action_counts[a] for a in action_names]
    colors       = [ACTION_COLORS.get(a, "#cccccc") for a in action_names]
    bars = ax.bar(action_names, action_vals, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, action_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(val), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Final Action Distribution", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    ax.set_xlabel("Step")

    plt.suptitle("Decision Replay Report — Simulated Proxy Metrics (Azure Trace)\n"
                 "⚠️ All values are SIMULATED, not real cloud measurements",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {png_path}")

# ── CSV summary ───────────────────────────────────────────────────────────────

def build_csv(records: list[dict], csv_path: str):
    if not records:
        return
    keys = ["timestamp_step", "actual_load", "forecast_load", "warm_containers",
            "latency_ms", "energy_proxy", "sla_violation", "cold_start_proxy",
            "final_action", "planner_action", "planner_confidence", "planner_sla_risk",
            "shield_overridden", "shield_reason_code", "planner_fallback", "correlation_id"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"[OK] Saved {csv_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else None
    records, source_path = generate_or_load(log_path)
    print(f"[INFO] Loaded {len(records)} decision records from {source_path}")

    os.makedirs("outputs", exist_ok=True)

    html_path = "outputs/decision_replay.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html(records))
    print(f"[OK] Saved {html_path}")

    build_csv(records, "outputs/decision_replay_summary.csv")
    build_plot(records, "outputs/decision_replay.png")

    print(f"\n=== REPLAY SUMMARY ===")
    n = len(records)
    sla_viol = sum(1 for r in records if r.get("sla_violation"))
    shield_ov = sum(1 for r in records if r.get("shield_overridden"))
    print(f"Steps:            {n}")
    print(f"SLA violations:   {sla_viol} ({sla_viol/max(n,1)*100:.1f}%)")
    print(f"Shield overrides: {shield_ov} ({shield_ov/max(n,1)*100:.1f}%)")
    avg_lat = sum(r.get("latency_ms",0) for r in records) / max(n, 1)
    print(f"Avg latency:      {avg_lat:.1f} ms")
    action_counts = {}
    for r in records:
        a = r.get("final_action","?")
        action_counts[a] = action_counts.get(a, 0) + 1
    for a, c in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {a}: {c} ({c/n*100:.1f}%)")
