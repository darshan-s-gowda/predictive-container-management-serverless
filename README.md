# Predictive Container Management for Energy-Efficient Serverless Computing

> **Academic research project** — All latency, energy, and cold-start figures are **simulated proxy metrics**, not measured in a production cloud environment.

---

## Overview

This project investigates whether **workload forecasting** and **intelligent container-management decisions** can reduce cold starts and resource waste while meeting latency/SLA requirements in serverless computing.

The system forecasts workload using a **Temporal Fusion Transformer (TFT)** and a **persistence baseline**, then drives container actions through:
- A **structured agentic planner** (rule-based with an OpenAI-compatible LLM hook for Qwen3)  
- A **PPO reinforcement-learning controller**  
- Classic **rule-based** and **fixed-container** baselines

---

## Architecture

```
Azure Functions Trace (Jan 2021)
        │
        ▼
  Preprocessing (5-min buckets)
        │
        ▼
  Workload Forecasting
  ├── Original TFT
  ├── Log-transformed TFT
  └── Persistence baseline  ← best on this trace
        │
        ▼
  Agentic Planner (structured JSON, safe fallback, LLM-ready)
        │
        ▼
  PPO Controller (Stable-Baselines3, diverse workload training)
        │
        ▼
  Python Simulator (container lifecycle, cold-start, energy proxy)
        │
        ▼
  Function Services (user_info, search, recommend)
        │
        ▼
  Monitoring & Feedback (JSONL logs, Prometheus exposition, dashboards)
        │
        ▼
  End-to-End Evaluation (6 policies × 10 seeds)
```

---

## Dataset

| Property | Value |
|---|---|
| Source | [Azure Functions Invocation Trace, Jan 2021](https://github.com/Azure/AzurePublicDataset) |
| Buckets | 4,032 (five-minute intervals, ~14 days) |
| Recorded invocations | ~1.98 million |
| Mean invocations/bucket | 491.31 |
| Max | 11,020 |

---

## Key Results

### Forecasting (matched one-step, test set — 606 buckets)

| Model | Overall MAE | Overall RMSE | High-load underprediction |
|---|---|---|---|
| Original TFT | 226.7 | 637.5 | 100% |
| Log-TFT | 182.6 | 489.6 | 98% |
| **Persistence** | **153.2** | **379.8** | **77%** |

Persistence outperforms both TFT variants on this trace. Log-TFT improves ~19% over original TFT.

### End-to-End Policy Comparison (Azure test segment, simulated)

| Policy | Mean latency (ms) | SLA violation rate | Avg energy proxy |
|---|---|---|---|
| Fixed-4 containers | 4,313 | 97.3% | 4.68 |
| Rule-based | 1,609 | 36.0% | 25.33 |
| Persistence + Rule | 1,686 | 36.5% | 22.99 |
| TFT + Rule | 1,609 | 36.0% | 25.33 |
| **Planner-assisted** | **1,465** | **28.3%** | 29.25 |
| **PPO** | **1,270** | **20.7%** | 37.00 |

PPO achieves the lowest SLA violation rate and latency at the cost of higher energy proxy (more containers). Planner-assisted is the best deterministic policy.

### PPO Validation (synthetic workloads, 20 seeds × 3 workload types)

| Policy | Avg latency (ms) | SLA violation rate | No-op fraction |
|---|---|---|---|
| Fixed-4 | 86.5 | 3.8% | 100% |
| Rule-based | 82.4 | 0.3% | 70% |
| **PPO** | **70.2** | **0.0%** | **42%** |

PPO average no-op fraction: **42%** ✅ (healthy; not stuck in no-op collapse)

---

## Repository Structure

```
major_1/
├── data/
│   ├── azure_serverless_workload.csv       ← preprocessed trace
│   ├── synthetic_serverless_workload.csv
│   └── azure_trace_raw/                    ← raw Azure trace (not committed)
├── src/
│   ├── preprocess_azure.py                 ← raw trace → 5-min buckets
│   ├── train_tft_azure.py                  ← original TFT training
│   ├── experiment_tft_azure_log.py         ← log-TFT training
│   ├── matched_horizon_comparison.py       ← Phase A: fair forecast comparison
│   ├── agentic_planner.py                  ← Phase C: structured JSON planner
│   ├── serverless_simulator.py             ← Phase D: execution layer simulator
│   ├── function_services.py                ← Phase E: example function handlers
│   ├── monitoring.py                       ← Phase F: JSONL logs + dashboards
│   ├── retrain_ppo.py                      ← Phase B: PPO retraining & validation
│   ├── ppo_env.py                          ← PPO Gymnasium environment
│   ├── end_to_end_evaluation.py            ← Phase G: 6-policy comparison
│   └── final_demo.py                       ← Phase H: integrated pipeline demo
├── outputs/                                ← generated results (plots, CSVs, models)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Environment Setup

```powershell
cd D:\major_1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Preprocess Azure Trace (if raw trace present)

```powershell
python src\preprocess_azure.py
```

### 3. Run All Phases in Order

```powershell
# Phase A — Matched-horizon forecast comparison
python src\matched_horizon_comparison.py

# Phase C — Planner smoke test (all 4 cases should PASS)
python src\agentic_planner.py

# Phase D — Simulator smoke test
python src\serverless_simulator.py

# Phase E — Function services demo (200 workload buckets)
python src\function_services.py

# Phase F — Monitoring dashboard demo
python src\monitoring.py

# Phase B — PPO retraining + validation (takes ~2 min on CPU)
python src\retrain_ppo.py

# Phase G — End-to-end 6-policy comparison
python src\end_to_end_evaluation.py

# Phase H — Full integrated demo (single-command pipeline)
python src\final_demo.py
```

### 4. Use Qwen3 (Optional)

Set environment variables before running any script:

```powershell
$env:PLANNER_ENDPOINT = "http://localhost:8000/v1"   # vLLM or similar
$env:PLANNER_MODEL    = "Qwen/Qwen3-8B"
$env:PLANNER_API_KEY  = "your-key-or-empty"
python src\final_demo.py
```

The planner falls back to deterministic rules if the endpoint is unavailable.

---

## Output Files

| File | Description |
|---|---|
| `outputs/matched_horizon_metrics.csv` | Forecast MAE/RMSE on test set |
| `outputs/matched_horizon_comparison.png` | Forecast comparison plot |
| `outputs/ppo_container_controller.zip` | Trained PPO model |
| `outputs/ppo_retrain_comparison.png` | PPO vs baselines (synthetic) |
| `outputs/end_to_end_comparison.png` | 6-policy evaluation chart |
| `outputs/final_demo.png` | Integrated pipeline demo plot |
| `outputs/monitoring_dashboard_*.png` | Per-policy monitoring dashboards |
| `outputs/monitoring_logs/*.jsonl` | Structured step-level event logs |

---

## Limitations

1. All latency/energy/cold-start numbers are **simulated** proxy metrics.
2. Azure trace provides completion timestamps, not arrival timestamps.
3. TFT systematically underpredicts high-load spikes (100% underprediction rate for original TFT).
4. PPO is trained on synthetic workloads; generalisation to real distributions is not verified.
5. Qwen3 LLM planning requires a running inference server.
6. No real Prometheus/Grafana integration verified.

---

## Requirements

```
numpy>=1.26
pandas>=2.1
matplotlib>=3.8
scikit-learn>=1.4
torch>=2.2
lightning>=2.2
pytorch-forecasting>=1.2
gymnasium>=0.29
stable-baselines3>=2.3
```

---

## License

MIT — for academic and research use.

---

## Author

Darshan S Gowda · darshandarshu1115@gmail.com
