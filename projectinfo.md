# Predictive Container Management for Energy-Efficient Serverless Computing

> **Status updated: 29 October 2026.** Data ingestion and multiple TFT
> experiments have been run. On the evaluated Azure trace holdout,
> persistence currently outperforms both TFT variants. PPO code exists
> but needs validation; Qwen3 planning and the remaining integrated
> execution/monitoring flow are not yet demonstrated as complete.

## 1. Project objective

Investigate whether workload forecasting and intelligent
container-management decisions can reduce cold starts and resource waste
while meeting latency/SLA requirements in serverless computing.

The system is intended to forecast workload, choose safe container
actions, and evaluate trade-offs among: - response latency and SLA/SLO
violations; - cold starts; - resource and energy proxies; -
over-provisioning and idle capacity; - responsiveness to sudden workload
spikes.

**Measurement boundary:** the current latency and energy figures in
controller experiments are simulated/proxy metrics, not production-cloud
measurements. The Azure trace supplies workload data; it does not
measure this project's controller latency or energy.

## 2. Target architecture

1.  **Users / clients** --- request sources and workload generators.
2.  **Workload ingestion** --- clean, aggregate, and expose historical
    workload data.
3.  **Workload prediction (TFT)** --- forecast workload at future time
    steps.
4.  **Agentic AI planning (Qwen3)** --- interpret forecast, system
    state, energy goals and SLA requirements; propose structured
    recommendations.
5.  **Reinforcement-learning controller (PPO)** --- choose container
    actions based on state and reward.
6.  **Serverless execution layer** --- apply actions to a simulator or
    optional local container runtime.
7.  **Applications / services** --- representative functions receiving
    work.
8.  **Monitoring and feedback** --- collect metrics and feed
    observations back to planning/control.
9.  **Evaluation** --- compare policies for latency, cold starts, SLA
    violations and resource/energy proxies.

``` text
Workload trace -> preprocessing -> TFT/baseline forecasts
               -> Qwen3 planning (planned)
               -> PPO controller (validate policy)
               -> simulator / optional local runtime
               -> metrics, plots and feedback
```

This is the **target architecture**, not a claim that all modules are
implemented. Module 4 onward is pending or needs validation as noted
below.

## 3. Repository layout

Working directory: `D:\major_1` on Windows. Files identified during
development:

``` text
D:\major_1\
├── README.md
├── requirements.txt
├── data\
│   ├── synthetic_serverless_workload.csv
│   ├── azure_serverless_workload.csv
│   └── azure_trace_raw\
│       └── AzureFunctionsInvocationTraceForTwoWeeksJan2021\
│           └── AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt
├── src\
│   ├── generate_data.py
│   ├── train_tft.py
│   ├── predict_tft.py
│   ├── ppo_env.py
│   ├── train_ppo.py
│   ├── run_demo.py
│   ├── evaluate_ppo.py
│   ├── evaluate_ppo_multi.py
│   ├── evaluate_tft.py
│   ├── analyze_tft_errors.py
│   ├── preprocess_azure.py
│   ├── train_tft_azure.py
│   ├── evaluate_tft_azure.py
│   ├── analyze_azure_forecasts.py
│   ├── experiment_tft_azure_log.py
│   ├── compare_azure_tft_models.py
│   └── finalize_azure_tft_comparison.py
└── outputs\  # trained weights, predictions, metric CSVs and plots
```

Check the actual directory before assuming this list is exhaustive.
Outputs created by different experiments may use different baseline
definitions or forecast horizons; do not compare metrics without
checking alignment.

## 4. Environment setup

Known environment: - Windows PowerShell - Python 3.11.4 - virtual
environment `.venv` - PyTorch Forecasting and Lightning installed -
latest training run used CPU (`GPU available: False`)

``` powershell
cd D:\major_1
.\.venv\Scripts\Activate.ps1
python --version
```

Inspect `requirements.txt` and the working environment before upgrading
dependencies; the project has encountered version-specific PyTorch
Forecasting prediction APIs. If requirements are pinned and complete:

``` powershell
python -m pip install -r requirements.txt
python -c "import torch, pandas, numpy, matplotlib, lightning, pytorch_forecasting; print('Imports OK'); print('torch:', torch.__version__)"
```

Run scripts from the repository root, for example
`python src\preprocess_azure.py`. Do not commit `.venv`, large raw
traces, or model binaries unless there is a clear reason and
data/license/storage policy allows it.

## 5. Data and preprocessing

The raw Azure Functions trace fields observed are: - `app`: application
identifier; - `func`: function identifier; - `end_timestamp`: elapsed
time in seconds; - `duration`: execution duration in seconds.

`src\preprocess_azure.py` aggregates recorded completions into
five-minute buckets and writes `data\azure_serverless_workload.csv`.

Known preprocessing summary:

  Property                    Observed value
  ------------------------- ----------------
  Buckets                              4,032
  Approximate coverage               14 days
  Recorded invocations             1,980,951
  Mean invocations/bucket             491.31
  Median                                 323
  95th percentile                    1,334.9
  99th percentile                   3,977.38
  Maximum                             11,020
  Mean function duration      4.8816 seconds

**Caveat:** timestamps are completion timestamps, not necessarily
request-arrival timestamps. Describe bucket counts as recorded
completions per five-minute bucket, not exact future arrivals.
`end_timestamp` is elapsed time, not calendar time. Synthetic workload
results must be labeled separately from Azure-trace results.

## 6. Current module status

Status labels mean: **Implemented** = code exists and has run;
**Evaluated** = experiment/metrics exist; **Needs validation** =
effectiveness or correctness is not sufficiently demonstrated;
**Pending** = implementation/integration not demonstrated.

  -----------------------------------------------------------------------
  Module                  Status                  Evidence / remaining
                                                  work
  ----------------------- ----------------------- -----------------------
  1\. Users / clients     Conceptually defined    Architecture identifies
                                                  request sources; full
                                                  client UI not
                                                  established.

  2\. Workload ingestion  Implemented             Azure trace
                                                  preprocessed into 4,032
                                                  buckets; synthetic data
                                                  also exist.

  3\. TFT prediction      Implemented and         Original TFT, log-TFT,
                          evaluated; decision     persistence, holdout
                          still needs closure     metrics and plots
                                                  exist. Persistence
                                                  currently wins.

  4\. Qwen3 agentic       Pending / not           Define structured
  planning                demonstrated            planner interface and
                                                  tests.

  5\. PPO controller      Code exists; policy     Environment, training
                          needs validation        and evaluation scripts
                                                  exist. A prior
                                                  diagnostic showed a
                                                  policy selecting no-op
                                                  for most steps; retrain
                                                  and revalidate.

  6\. Serverless          Pending / scope         Current latency/energy
  execution               undecided               behavior is simulated;
                                                  local container runtime
                                                  not established.

  7\.                     Partially represented   Add representative
  Applications/services   by workload/function    functions/routing if
                          IDs                     required by final demo.

  8\. Monitoring and      Partial                 Evaluation scripts log
  feedback                                        metrics;
                                                  Prometheus/Grafana and
                                                  closed-loop feedback
                                                  not verified.

  9\. End-to-end          Partial                 Forecast evaluation
  evaluation                                      exists; integrated
                                                  multi-policy experiment
                                                  remains.

  Final report/demo       In progress             Finalize after
                                                  reproducible integrated
                                                  results.
  -----------------------------------------------------------------------

## 7. Module 3: forecasting experiments

Experiments completed: 1. Original TFT on Azure workload. 2. Persistence
baseline: forecast at time `t` using the observed workload at `t-1`. 3.
Previous-day seasonal baseline. 4. Log-transformed TFT: train on
`log1p(invocation_rate)` and convert predictions back with `expm1`. 5.
Common-holdout comparison using saved original and log-TFT predictions.

The chronological split used 3,225 training buckets and 807 holdout
buckets. The training-derived high-load threshold is the training
workload's 80th percentile, `721.20` invocations/bucket. There are 111
high-load holdout timestamps by this definition.

### Latest consistent results

Source outputs: `outputs\azure_tft_final_metrics.csv` and
`outputs\azure_tft_final_safety.csv`.

  Model                           Segment            MAE        RMSE   Count
  ------------------------------- ------------ --------- ----------- -------
  Original TFT                    Overall        223.442     570.659     807
  Original TFT                    High load      842.436   1,475.985     111
  Original TFT                    Other load     124.723     173.636     696
  Log-transformed TFT             Overall        178.085     440.465     807
  Log-transformed TFT             High load      636.816   1,118.358     111
  Log-transformed TFT             Other load     104.926     159.631     696
  Persistence (previous bucket)   Overall        159.511     354.478     807
  Persistence (previous bucket)   High load      410.378     701.068     111
  Persistence (previous bucket)   Other load     119.501     259.440     696

High-load underprediction (actual \> forecast):

  Model                     Rate
  --------------------- --------
  Original TFT            90.99%
  Log-transformed TFT     83.78%
  Persistence             65.77%

Conclusions supported by evidence: - Log transformation improved overall
TFT MAE by about 20.3% and RMSE by about 22.8% relative to the original
TFT. - Persistence still performs better than both TFT variants on
overall MAE/RMSE and high-load MAE. - TFT predictions
smooth/underpredict some abrupt spikes. - Do **not** claim that TFT
beats persistence on this Azure trace.

### Evaluation caveats

-   The original TFT experiment used a 12-step horizon and selected the
    shortest lead per timestamp; the log-TFT experiment is
    one-step-ahead. The saved outputs are timestamp-aligned, but this is
    not a perfectly matched horizon/architecture comparison. Run a
    matched one-step experiment if making a strong model-to-model claim.
-   Earlier evaluation scripts used slightly different persistence
    alignment. The final comparison recalculates one canonical baseline
    from the source workload: forecast at `t` = actual workload at
    `t-1`. Use the final comparison outputs for the latest comparison.
-   The holdout has already been inspected repeatedly. For final claims,
    use an untouched test segment or chronological train/validation/test
    split. Use validation data for tuning and keep the final test set
    untouched.
-   Underprediction matters for container safety, but safety margins can
    increase over-provisioning and resource use. Report forecasting
    accuracy and controller outcomes together.

## 8. Commands for the current forecasting pipeline

Run from `D:\major_1` with `.venv` activated.

``` powershell
# Preprocess Azure trace (rerun only if needed)
python src\preprocess_azure.py

# Train original Azure TFT
python src\train_tft_azure.py

# Evaluate original Azure TFT
python src\evaluate_tft_azure.py

# Analyze existing original forecasts
python src\analyze_azure_forecasts.py

# Train/evaluate log-TFT (re-trains model)
python src\experiment_tft_azure_log.py

# Generate latest consistent comparison
python src\finalize_azure_tft_comparison.py
```

Latest canonical outputs: - `outputs\azure_tft_final_metrics.csv` -
`outputs\azure_tft_final_safety.csv` -
`outputs\azure_tft_final_predictions.csv` -
`outputs\azure_tft_final_comparison.png`

`analyze_azure_forecasts.py` and `compare_azure_tft_models.py` are
earlier/exploratory comparison scripts. Use
`finalize_azure_tft_comparison.py` for the latest canonical one-step
persistence comparison while documenting the horizon mismatch.

## 9. Remaining work: execution plan

Work in order; don't implement all modules simultaneously.

### Phase A --- close Module 3

1.  Freeze current outputs and record metrics above.
2.  Run a matched-horizon comparison for original TFT, log-TFT and
    persistence on identical timestamps, or use a new chronological
    train/validation/test split.
3.  If attempting improvements, use validation data for model selection,
    hyperparameters and safety margins; keep final test untouched.
4.  Evaluate high-load underprediction, MAE, RMSE and
    overprediction/resource implications.
5.  Record the decision. If persistence remains best, retain it as
    operational fallback and document TFT as implemented/evaluated but
    not better than baseline.
6.  Freeze a forecast interface with timestamp, predicted value, horizon
    and model name.

**Definition of done:** reproducible metrics, baseline definition,
honest model-selection decision, limitations and stable output format.

### Phase B --- validate PPO (Module 5) before integration

Inspect `src\ppo_env.py`, `src\train_ppo.py`, `src\evaluate_ppo.py` and
`src\evaluate_ppo_multi.py` before rewriting.

1.  Check observation/action definitions, reward terms, termination,
    scaling and future-data leakage.
2.  Train on varied training traces/seeds.
3.  Ensure action counts are not dominated by no-op and actions affect
    container count when appropriate.
4.  Compare at least fixed-container, rule-based, persistence-driven,
    and PPO policies.
5.  Evaluate on workloads not used in training; report average and
    variability across seeds.

A previous diagnostic of an earlier policy showed action counts
`[800, 0, 6, 0, 0, 0]`, mostly no-op. This is a warning to revalidate,
not proof that a newer policy behaves identically.

**Definition of done:** saved trained policy, sensible action
distribution, reproducible evaluation, fair baselines, no
future-information leakage.

### Phase C --- implement Qwen3 planner (Module 4)

1.  Define typed inputs: forecast, current warm-container count,
    latency/SLA status, capacity limits, resource/energy proxy.
2.  Define an allow-listed action schema such as `NO_OP`, `PREWARM`,
    `SCALE_OUT`, `SCALE_IN`, `REUSE`, `TERMINATE_IDLE`.
3.  Require structured JSON output containing recommended action, brief
    rationale and relevant constraints.
4.  Validate output; reject malformed or unsafe actions and fall back to
    a deterministic safe rule.
5.  Define one clear responsibility: Qwen3 advises PPO, selects among
    approved strategies, or explains decisions. Avoid conflicting
    independent controllers.
6.  Test invalid JSON, model unavailability, excessive scale requests,
    SLA-critical states and missing fields.
7.  Log model/version, prompt/template version, input, parsed action and
    fallback reason.
8.  Check Qwen3 availability, inference requirements and licensing.
    Never commit API keys.

**Definition of done:** structured recommendations, schema validation,
safe fallback, unit tests and clear Qwen3/PPO responsibility.

### Phase D --- execution layer (Module 6)

**Recommended first: Python simulator.** - Simulate warm/idle containers
and lifecycle actions. - Simulate cold-start and service-time effects. -
Enforce container bounds and apply allowed actions. - Collect latency,
cold-start, capacity and energy-proxy metrics.

Optional second step: local Docker demonstration with a small function
service. Measure actual runtime events where feasible; don't label
simulated energy as measured energy. Kubernetes is optional unless
required by project scope.

**Definition of done:** actions produce observable state transitions and
failures are handled safely.

### Phase E --- applications/services (Module 7)

1.  Implement two or three small example functions (e.g. user info,
    search, recommendation).
2.  Replay/generate requests from the workload driver.
3.  Tag requests with function ID and time where possible.
4.  Route workload to the right function and collect per-function
    metrics.
5.  Keep scope small; the research focus is container management.

### Phase F --- monitoring/feedback (Module 8)

Log at least: - time index, actual and predicted workload; -
warm-container count and chosen action; - simulated/measured response
latency; - cold-start count/proxy; - CPU/memory or resource proxy; -
energy proxy and its formula; - SLA threshold and violation flag; -
policy/model version and random seed.

Prometheus/Grafana are optional enhancements unless required. Don't
claim integration until the data flow and dashboard have actually been
run and verified.

### Phase G --- end-to-end evaluation (Module 9)

Run the same held-out workload and initial conditions through: 1. fixed
container count; 2. rule-based controller; 3. persistence forecast +
controller; 4. selected TFT forecast + controller; 5. PPO controller; 6.
optional Qwen3-assisted strategy.

Report: - mean/P95 latency and units; label simulated vs measured; -
cold-start count/rate; - SLA violations/rate; - average/peak warm
containers; - resource/energy proxy and formula; - forecast MAE/RMSE; -
action counts and policy runtime; - mean and standard deviation across
multiple seeds/traces.

Keep trace, initial state, SLA and scoring identical for fair
comparison. Add ablations to isolate TFT, Qwen3 and PPO contribution.
Discuss trade-offs between energy proxy and SLA/latency.

### Phase H --- final report/demo

Include problem statement, objectives, related work, architecture,
dataset/preprocessing, forecasting and baseline methodology, PPO
state/action/reward design, Qwen3 safety/role, experiment setup, result
tables/plots, limitations, conclusion, future work and reproducible
commands/software versions.

The demo should show workload -\> forecast -\> decision -\> container
state transition -\> resulting metrics.

## 10. Engineering rules

1.  Label simulated results as simulated; don't claim cloud measurements
    unless actually collected.
2.  Prevent data leakage: a policy may only use information available at
    the decision time.
3.  Use chronological splits, not random time-series shuffling.
4.  Document the exact baseline and align timestamps/horizons.
5.  Tune on validation data; keep the final test untouched.
6.  Record seeds, package versions, configs, input trace and output
    paths.
7.  Enforce safe action bounds and deterministic fallback behavior.
8.  Run and test each module before integrating the next.
9.  Save experiments to separate filenames; preserve known-good outputs.
10. Finish the research pipeline before adding optional
    dashboard/Kubernetes complexity.

## 11. Immediate next action

**Phase A is the next task:** run a matched-horizon one-step comparison
for both TFT variants and persistence, or create a fresh chronological
train/validation/test split. This is necessary for a strong comparison
because the original TFT and log-TFT experiments used different
horizons.

Then validate PPO before connecting it to Qwen3 or the execution layer.

## 12. Quick status summary

-   Data ingestion: implemented and run.
-   Azure workload: 4,032 five-minute buckets, about 1.98 million
    recorded invocations.
-   Original TFT: implemented and evaluated.
-   Log-TFT: implemented and evaluated; better than original TFT, worse
    than persistence.
-   Persistence: strongest forecast among the current evaluated
    baselines.
-   PPO: code exists; effectiveness needs revalidation.
-   Qwen3 planner: implementation/verification pending.
-   Execution: simulation behavior exists in controller-related code;
    real container execution not established.
-   Monitoring: evaluation outputs exist; full Prometheus/Grafana
    feedback loop not established.
-   End-to-end project: not complete.
