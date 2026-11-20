# FINAL MAJOR PROJECT — HANDOFF / CONTINUATION GUIDE

Project:
Predictive Container Management for Energy-Efficient Serverless Computing

Purpose of this file:
This document is the single handoff source for continuing the project with Gemini or another coding assistant. It records what is already implemented, what is verified, what is only a proxy/simulation, and the exact point from which work must continue.

============================================================
1. PROJECT LOCATION AND ENVIRONMENT
============================================================

Project root:
D:\major_1

Platform:
Windows PowerShell

Python:
3.11.4

Virtual environment:
.venv

Run commands from:
D:\major_1

Main source directory:
src\

Main output directory:
outputs\

============================================================
2. PROJECT ARCHITECTURE
============================================================

Intended architecture:

Azure workload data
        |
        v
Workload preprocessing
        |
        v
Forecasting (TFT / persistence)
        |
        v
Agentic planning layer
        |
        v
PPO reinforcement-learning controller
        |
        v
Serverless container simulator
        |
        v
Latency / energy / SLA / warm-container metrics
        |
        v
End-to-end evaluation

Important truth boundaries:

- The serverless simulator produces estimated/proxy latency, energy, cold-start and capacity metrics.
- Azure trace data is workload input, not proof of production controller performance.
- There is no verified production Docker/Kubernetes deployment.
- The planner currently uses a deterministic local fallback unless a real configured LLM endpoint is available. Logs have shown fallback=True.
- PPO has NOT yet been successfully integrated into the calibrated Azure end-to-end comparison.
- Do not claim that Qwen3 is actually performing inference unless a configured Qwen3 endpoint is demonstrated.
- Do not claim real cloud energy measurements or real production cold-start counts.

============================================================
3. WHAT IS COMPLETED
============================================================

A. Azure workload ingestion and preprocessing
----------------------------------------------

Implemented and working.

The project reads the Azure workload CSV and validates:

- time_idx
- invocation_count
- mean_duration_seconds

Invalid/missing rows are removed.

The workload is sorted by time_idx and duplicate timestamps are removed.

B. Chronological train/test split
---------------------------------

Implemented in:

src\end_to_end_evaluation.py

Current split:

TEST_FRAC = 0.15

Therefore:

- first 85% = training period
- final 15% = untouched test period

This is chronological, not random.

This must be preserved to avoid temporal leakage.

C. Azure workload -> concurrency conversion
--------------------------------------------

Current conversion:

estimated concurrency =
    invocation_count
    * mean_duration_seconds
    / 300 seconds

BUCKET_SECONDS = 300.0

The final evaluation uses this estimated concurrency as simulator workload.

Important:
This is an estimated average concurrency measure, not measured instantaneous concurrency.

D. Forecast alignment
---------------------

Implemented.

Forecast files are aligned to test timestamps using time_idx.

Forecasts are not padded or shifted if timestamps are missing.

Currently supported forecast sources:

- persistence
- original TFT
- log TFT

TFT invocation-count predictions are converted to concurrency using the TRAINING-period mean execution duration.

This is intentional and must not be changed to test-period duration.

E. Serverless simulator
-----------------------

Implemented:

src\serverless_simulator.py

Main characteristics:

- 10 workload units capacity per warm container
- initial warm containers = 4
- SLA = 200 ms
- seeded noise
- container scaling actions
- latency calculation
- energy proxy
- cold-start/load proxy
- SLA violation calculation
- throughput/capacity metrics

Policy actions include:

- NO_OP
- PREWARM
- SCALE_OUT
- SCALE_IN
- REUSE
- TERMINATE_IDLE

The simulator exposes a common interface used by policy evaluation.

F. Existing non-PPO policies
-----------------------------

Implemented and evaluated:

1. Fixed-4
2. Reactive Rule-based
3. Persistence + Rule
4. Original TFT + Rule
5. Log TFT + Rule
6. Planner-assisted

These policies use the same calibrated simulator.

G. End-to-end evaluation
------------------------

Implemented:

src\end_to_end_evaluation.py

Outputs:

outputs\end_to_end_results.csv
outputs\end_to_end_summary.csv
outputs\end_to_end_comparison.png

The evaluation runs multiple seeds and aggregates results.

Current evaluation metrics include:

- mean_latency_ms
- p95_latency_ms
- mean_bucket_p95_latency_ms
- cold_start_load_proxy
- cold_start_rate
- sla_violations
- sla_violation_rate
- avg_warm_containers
- peak_warm_containers
- avg_energy_proxy
- action counts

Important metric definitions:

p95_latency_ms:
95th percentile across per-bucket mean latency observations.

mean_bucket_p95_latency_ms:
mean of the simulator's per-bucket P95 estimates.

These are NOT request-level measured production P95 metrics.

cold_start_load_proxy:
a simulated/estimated load proxy. Do NOT describe it as actual measured cold-start event count.

H. Final demo
-------------

Implemented:

src\final_demo.py

Successful run generated:

outputs\final_demo_results.csv
outputs\final_demo.png
outputs\monitoring_dashboard_Planner-assisted.png

and a monitoring JSONL log under:

outputs\monitoring_logs\

Latest successful final-demo run summary:

- 288 buckets
- Raw invocation mean ≈ 281.21
- Raw invocation max ≈ 1329
- Scaled concurrency mean ≈ 23.43
- Scaled concurrency max ≈ 110.75
- Mean latency ≈ 225.23 ms
- P95 across bucket means ≈ 1095.96 ms
- Mean bucket P95 estimate ≈ 281.54 ms
- SLA violation rate ≈ 21.18%
- Cold-start load proxy ≈ 1104.35
- Average warm containers ≈ 2.86
- Average energy proxy ≈ 3.0058
- Actions:
  NO_OP 85
  SCALE_OUT 62
  REUSE 57
  TERMINATE_IDLE 43
  SCALE_IN 41

Planner logs showed fallback=True, so the planner result must be described as planner-assisted/local fallback unless a real LLM endpoint is configured.

I. Monitoring
-------------

Monitoring logger/dashboard infrastructure exists.

It records controller decisions and simulator metrics into JSONL logs and produces dashboard output.

============================================================
4. VERIFIED END-TO-END RESULTS BEFORE PPO
============================================================

Latest calibrated end-to-end comparison was run with 10 seeds per policy.

PPO was intentionally SKIPPED because the existing PPO model was trained on a different raw workload scale/environment and was not valid for this calibrated comparison.

Results:

Policy                  Mean Latency    Temporal P95    Mean Bucket P95
-----------------------------------------------------------------------
Fixed-4                 210.761         70.000           263.450
Log TFT + Rule          213.221         270.028          266.526
Original TFT + Rule     213.221         270.028          266.526
Persistence + Rule      212.315         203.023          265.395
Planner-assisted        213.299         280.546          266.624
Reactive Rule-based     211.632         179.079          264.538

Other averages:

Fixed-4:
- SLA violation ≈ 3.0%
- warm containers = 4.0
- energy proxy ≈ 3.519

Log TFT + Rule:
- SLA violation ≈ 5.3%
- warm containers ≈ 1.456
- energy proxy ≈ 1.483

Original TFT + Rule:
- SLA violation ≈ 5.3%
- warm containers ≈ 1.455
- energy proxy ≈ 1.483

Persistence + Rule:
- SLA violation ≈ 5.1%
- warm containers ≈ 1.465
- energy proxy ≈ 1.491

Planner-assisted:
- SLA violation ≈ 5.3%
- warm containers ≈ 1.454
- energy proxy ≈ 1.483

Reactive Rule-based:
- SLA violation ≈ 5.0%
- warm containers ≈ 1.674
- energy proxy ≈ 1.658

Interpretation:
No single policy dominates every metric.

- Fixed-4 has the lowest SLA violation but highest energy.
- Reactive Rule has the lowest mean latency among adaptive policies.
- Persistence + Rule has the best average bucket P95 among adaptive policies.
- TFT/planner policies achieve low energy/warm-container usage but have higher SLA violation than Fixed-4.

Do not oversell the result.

============================================================
5. CURRENT PPO STATUS — NOT COMPLETE
============================================================

This is the exact point where the project must continue.

Existing files:

src\ppo_env.py
src\retrain_ppo.py

The current PPO implementation is NOT ready for the final calibrated comparison.

Why:

The current src\ppo_env.py is a separate simplified simulator.

Its observation has 9 values:

0 load
1 predicted load
2 cpu
3 memory
4 warm containers
5 cold
6 latency
7 energy
8 risk

Its action space has 6 actions:

0 = no-op
1 = +1 warm
2 = +2 warm
3 = -1 warm
4 = no-op
5 = -2 warm

Its latency/energy/reward calculations are different from src\serverless_simulator.py.

Therefore:
DO NOT simply retrain the current PPO environment and claim that it is comparable to the calibrated end-to-end simulator.

The existing retrain_ppo.py also trains on synthetic workloads such as:

- mixed
- daily
- bursty
- steady

and evaluates those synthetic workloads.

Those old PPO results are useful only as preliminary/synthetic experiments.

They are NOT the final calibrated Azure results.

Old synthetic PPO behavior:
PPO generally reduced latency/SLA violations but used more warm containers and more energy proxy than rule/fixed policies.

Do not present those numbers as Azure-calibrated final results.

============================================================
6. EXACT NEXT WORK — PPO CALIBRATION
============================================================

The next developer/assistant MUST do the following in order.

STEP 1:
Create a PPO environment/adapter that uses the SAME calibrated simulator logic as:

src\serverless_simulator.py

Do not create a second incompatible latency/energy model.

STEP 2:
Use the SAME workload units as end_to_end_evaluation.py:

estimated concurrency =
invocation_count * mean_duration_seconds / 300

STEP 3:
Preserve the chronological 85/15 split.

PPO training must use ONLY the first 85% training segment.

The final 15% test segment must remain untouched until evaluation.

STEP 4:
Train PPO on the calibrated workload scale.

It is acceptable to create multiple training episodes from the training segment and/or carefully generated variations around the training distribution, but no test actual values may be used to train the policy.

STEP 5:
The PPO observation must contain only information available at decision time.

At a decision point, PPO may use:

- previous/current observed simulator state
- previous observed workload
- current/next forecast
- warm container count
- previous latency
- energy proxy
- cold-start proxy
- SLA information

It MUST NOT use the upcoming actual workload as an observation.

STEP 6:
Map PPO actions to the simulator's action strings.

Recommended mapping must be documented explicitly, for example:

0 -> NO_OP
1 -> PREWARM
2 -> SCALE_OUT
3 -> SCALE_IN
4 -> REUSE
5 -> TERMINATE_IDLE

If the simulator action semantics differ, adapt the mapping to the actual simulator implementation instead of guessing.

STEP 7:
Use a reward aligned with the calibrated simulator objectives:

- latency
- SLA violation
- energy
- cold-start/load proxy
- overprovisioning

The reward should be documented.

Do not invent a reward that makes PPO look artificially better.

STEP 8:
Save the trained PPO model, for example:

outputs\ppo_container_controller_calibrated.zip

STEP 9:
Integrate PPO into:

src\end_to_end_evaluation.py

PPO should become another policy alongside:

Fixed-4
Reactive Rule-based
Persistence + Rule
Original TFT + Rule
Log TFT + Rule
Planner-assisted

The existing six policy results must remain unchanged.

STEP 10:
Evaluate PPO on the SAME untouched Azure test segment.

Use the SAME:

- actual concurrency
- forecast
- simulator
- SLA
- seeds
- metrics
- aggregation

used by the existing calibrated evaluation.

STEP 11:
Generate updated:

outputs\end_to_end_results.csv
outputs\end_to_end_summary.csv
outputs\end_to_end_comparison.png

with PPO included.

============================================================
7. IMPORTANT PPO EVALUATION FAIRNESS RULE
============================================================

The PPO model must not be evaluated with raw invocation counts if the other policies use estimated concurrency.

Everything must use the same workload unit.

Bad:
raw invocation_count for PPO
+
estimated concurrency for other policies

Good:
estimated concurrency for ALL policies.

============================================================
8. FORECAST CHOICE FOR PPO
============================================================

For the first calibrated PPO integration, use ONE clearly defined forecast.

Recommended:
Log TFT forecast converted to concurrency.

Reason:
It already exists in the calibrated evaluation and provides a consistent forecast input to the controller.

Alternatively, persistence can be used for a clean controller-only comparison, but document the choice.

Do not silently switch forecasts between training and testing.

============================================================
9. DATA LEAKAGE RULES
============================================================

Never use test actual workload to:

- train PPO
- tune PPO hyperparameters
- select reward coefficients based on test performance
- choose the best PPO checkpoint based on test metrics

The test period is for final evaluation only.

Training data:
first 85%.

Testing data:
final 15%.

============================================================
10. EXISTING FILES TO PRESERVE
============================================================

Do not unnecessarily rewrite:

src\serverless_simulator.py
src\end_to_end_evaluation.py
src\final_demo.py

unless integration requires a targeted change.

Prefer creating:

src\ppo_calibrated_env.py

and replacing/updating:

src\retrain_ppo.py

Then make a minimal targeted modification to:

src\end_to_end_evaluation.py

to load and evaluate the trained PPO controller.

============================================================
11. COMMANDS / VERIFICATION
============================================================

Environment activation:

.venv\Scripts\Activate.ps1

Run final demo:

python src\final_demo.py

Run end-to-end evaluation:

python src\end_to_end_evaluation.py

After PPO calibration is implemented, the intended sequence is:

1. train PPO
2. verify model file exists
3. run calibrated end-to-end evaluation
4. inspect PPO metrics
5. compare all seven policies
6. regenerate plots/CSV
7. prepare project explanation/report

============================================================
12. WHAT MUST NOT BE CLAIMED IN THE FINAL REPORT
============================================================

Do NOT claim:

- real production deployment
- real Kubernetes deployment
- real Docker deployment unless separately verified
- actual production energy consumption
- actual measured cold-start events
- real cloud SLA measurements
- real Qwen3 inference if fallback=True
- PPO superiority before calibrated evaluation proves it
- request-level P95 when the metric is simulator-derived bucket P95
- cold_start_load_proxy as actual cold-start event count

Correct wording:

"simulated"
"estimated"
"proxy"
"Azure trace-derived workload"
"calibrated simulator"
"offline evaluation"

============================================================
13. FINAL PROJECT STORY AFTER PPO IS COMPLETE
============================================================

The final explanation should follow this order:

1. Problem:
   Serverless workloads vary over time, so static container provisioning wastes
   energy or causes latency/SLA problems.

2. Workload:
   Azure trace data is processed and converted into estimated concurrency.

3. Forecasting:
   Persistence and TFT models predict upcoming workload.

4. Planning:
   An agentic planner can recommend scaling actions based on structured state.
   Current implementation may use a deterministic fallback when an external LLM
   endpoint is unavailable.

5. Reinforcement learning:
   PPO learns a container-management policy from the simulator.

6. Execution:
   The simulator applies container actions and estimates latency, energy,
   warm containers and SLA violations.

7. Monitoring:
   Controller actions and metrics are logged and visualized.

8. Evaluation:
   PPO and baseline controllers are evaluated on the same untouched Azure test
   period using the same simulator and metrics.

9. Result:
   Report trade-offs across latency, SLA violations, energy proxy and
   container utilization. Do not claim one controller is best unless the
   measured results support that conclusion.

============================================================
14. IMMEDIATE CONTINUATION POINT
============================================================

START HERE.

The current task is:

"Build calibrated PPO using the existing ServerlessSimulator and integrate it
into the existing end-to-end evaluation."

First inspect the complete implementation of:

src\serverless_simulator.py

and the remaining main/evaluation section of:

src\end_to_end_evaluation.py

Then implement:

1. src\ppo_calibrated_env.py
2. calibrated src\retrain_ppo.py
3. minimal PPO integration into src\end_to_end_evaluation.py

Then run training.

Do not stop after designing the solution.
Actually create/update the files and run the commands.

After every major change, verify the generated output before proceeding.

============================================================
END OF HANDOFF DOCUMENT
============================================================
