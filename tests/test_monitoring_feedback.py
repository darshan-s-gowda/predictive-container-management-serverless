"""
Milestone 7 — Monitoring & Feedback Integration Test
Runs a 20-step end-to-end pipeline with DecisionEngine → MonitoringLogger → record_feedback.
Saves feedback_log.jsonl and monitoring dashboard.
"""
import sys, os, json
sys.path.insert(0, 'src')

import numpy as np

from decision_engine import DecisionEngine, _SimulatorStub
from planner_providers import LocalRuleProvider
from safety_shield import SafetyShield
from monitoring import MonitoringLogger

PASS = []
FAIL = []

def test(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  [PASS] {name}")
    except Exception as e:
        import traceback; traceback.print_exc()
        FAIL.append(name)
        print(f"  [FAIL] {name}: {e}")

def t_feedback_integration():
    rng = np.random.default_rng(7)
    actual   = rng.uniform(5, 30, 20).tolist()
    forecast = [a * rng.uniform(0.9, 1.1) for a in actual]

    sim     = _SimulatorStub()
    planner = LocalRuleProvider()
    shield  = SafetyShield()
    engine  = DecisionEngine(sim, planner, shield, run_id="m7test",
                              log_dir="outputs/decision_logs")

    logger = MonitoringLogger(
        policy_name="DecisionEngine+LocalRule",
        policy_version="m7",
        seed=7,
        forecast_model="mock",
        sla_ms=300.0,
    )

    records = engine.run_trace(actual, forecast, sla_target_ms=300.0, initial_containers=4)
    engine.close()

    feedback_records = []
    for i, rec in enumerate(records):
        step_metrics = {
            "latency_ms":       rec.latency_ms,
            "sla_violation":    rec.sla_violation,
            "energy_proxy":     rec.energy_proxy,
            "warm_containers":  rec.warm_containers,
            "cold_start_proxy": rec.cold_start_proxy,
            "feedback_note":    "m7_integration_test",
        }
        fb = logger.record_feedback(rec, step_metrics)
        feedback_records.append(fb)

        logger.log(
            time_idx=i,
            actual_invocations=actual[i],
            predicted_invocations=forecast[i],
            warm_containers=rec.warm_containers,
            action=rec.final_action,
            container_delta=rec.planner_delta,
            latency_ms=rec.latency_ms,
            p95_latency_ms=rec.latency_ms * 1.3,
            cold_start_count=rec.cold_start_proxy * actual[i],
            cold_start_proxy=rec.cold_start_proxy,
            cpu_pct=min(actual[i] / 50.0 * 100, 100.0),
            memory_pct=min(rec.warm_containers * 2.0, 100.0),
            energy_proxy=rec.energy_proxy,
            sla_violated=rec.sla_violation,
            sla_excess_ms=max(0.0, rec.latency_ms - 300.0),
        )

    logger.close()

    assert len(records) == 20, f"Expected 20 records, got {len(records)}"
    assert len(feedback_records) == 20

    # Verify feedback log exists
    assert os.path.exists("outputs/feedback_log.jsonl")
    with open("outputs/feedback_log.jsonl") as f:
        lines = [l.strip() for l in f if l.strip()]
    # At least 20 lines (may be more from previous runs)
    assert len(lines) >= 20, f"feedback_log has {len(lines)} lines"
    fb = json.loads(lines[-1])
    assert "decision_correlation_id" in fb
    assert "observed_latency_ms" in fb
    assert fb["feedback_note"] == "m7_integration_test"

test("M7: 20-step pipeline with feedback logging", t_feedback_integration)

def t_prometheus_snapshot():
    logger = MonitoringLogger("prometheus_test", sla_ms=300.0)
    logger.log(0, 10.0, 11.0, 4, "REUSE", 0, 150.0, 195.0, 0.5, 0.05, 50.0, 40.0, 3.5, False, 0.0)
    snap = logger.prometheus_snapshot()
    logger.close()
    assert "serverless_warm_containers" in snap
    assert "serverless_latency_ms" in snap
    assert "serverless_sla_violated" in snap
    assert "serverless_energy_proxy" in snap

test("M7: prometheus_snapshot contains expected metric names", t_prometheus_snapshot)

def t_feedback_fields():
    from decision_engine import DecisionRecord
    import uuid
    rec = DecisionRecord(
        correlation_id=str(uuid.uuid4()),
        timestamp_step=5,
        actual_load=15.0, forecast_load=16.0,
        planner_provider="local_rules", planner_action="SCALE_OUT",
        planner_confidence=0.85, planner_sla_risk="high",
        planner_delta=1, planner_rationale="test",
        planner_fallback=False,
        ppo_obs_vec=[], ppo_action_int=-1, ppo_action_str="N/A",
        shield_input_action="SCALE_OUT", shield_approved="SCALE_OUT",
        shield_reason_code="PASS", shield_overridden=False, shield_details="",
        final_action="SCALE_OUT", latency_ms=200.0, energy_proxy=4.0,
        sla_violation=False, warm_containers=5, cold_start_proxy=0.02,
    )
    logger = MonitoringLogger("feedback_field_test", sla_ms=300.0)
    fb = logger.record_feedback(rec, {"latency_ms": 200.0, "sla_violation": False,
                                      "energy_proxy": 4.0, "warm_containers": 5,
                                      "cold_start_proxy": 0.02})
    logger.close()
    assert fb["decision_correlation_id"] == rec.correlation_id
    assert fb["final_action"] == "SCALE_OUT"
    assert fb["observed_latency_ms"] == 200.0
    assert not fb["observed_sla_violation"]

test("M7: record_feedback returns correct fields", t_feedback_fields)

print()
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
else:
    print("ALL PASS")
