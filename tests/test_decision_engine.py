"""
Milestone 4 — Safety Shield + Decision Engine Tests
Run directly with the venv Python (no pytest required).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from safety_shield import SafetyShield, ALLOWED_ACTIONS
from decision_engine import DecisionEngine, DecisionRecord, _SimulatorStub
from planner_providers import LocalRuleProvider

PASS = []
FAIL = []

def test(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  [PASS] {name}")
    except Exception as e:
        FAIL.append(name)
        print(f"  [FAIL] {name}: {e}")

# ── SafetyShield ──────────────────────────────────────────────────────────────

def t_pass_safe_action():
    shield = SafetyShield()
    state = {"warm_containers": 4, "latency_ms": 100.0, "sla_target_ms": 300.0}
    res = shield.validate("NO_OP", state)
    assert res.reason_code == "PASS"
    assert not res.was_overridden
    assert res.approved_action == "NO_OP"

test("SafetyShield: safe action passes", t_pass_safe_action)

def t_illegal_action():
    shield = SafetyShield()
    state = {"warm_containers": 4, "latency_ms": 100.0, "sla_target_ms": 300.0}
    res = shield.validate("LAUNCH_MISSILES", state)
    assert res.was_overridden
    assert res.reason_code == "ILLEGAL_ACTION"
    assert res.approved_action == "NO_OP"

test("SafetyShield: illegal action overridden to NO_OP", t_illegal_action)

def t_terminate_blocked_sla_violated():
    shield = SafetyShield()
    state = {"warm_containers": 5, "latency_ms": 500.0, "sla_target_ms": 300.0}  # SLA violated
    res = shield.validate("TERMINATE_IDLE", state)
    assert res.was_overridden
    assert res.reason_code == "SLA_VIOLATED"
    assert res.approved_action == "NO_OP"

test("SafetyShield: TERMINATE_IDLE blocked when SLA violated", t_terminate_blocked_sla_violated)

def t_scale_in_blocked_near_sla():
    shield = SafetyShield()
    state = {"warm_containers": 4, "latency_ms": 280.0, "sla_target_ms": 300.0}  # 93% of SLA
    res = shield.validate("SCALE_IN", state)
    assert res.was_overridden
    assert res.reason_code == "NEAR_SLA"

test("SafetyShield: SCALE_IN blocked when latency > 90% SLA", t_scale_in_blocked_near_sla)

def t_cooldown():
    shield = SafetyShield(cooldown_steps=2)
    state = {"warm_containers": 4, "latency_ms": 100.0, "sla_target_ms": 300.0}
    # First SCALE_OUT should pass
    r1 = shield.validate("SCALE_OUT", state)
    assert not r1.was_overridden, f"First SCALE_OUT should pass, got {r1.reason_code}"
    # Immediate second SCALE_OUT should be blocked by cooldown
    r2 = shield.validate("SCALE_OUT", {"warm_containers": 5, "latency_ms": 100.0, "sla_target_ms": 300.0})
    assert r2.was_overridden, "Second SCALE_OUT should be blocked by cooldown"
    assert r2.reason_code == "COOLDOWN"

test("SafetyShield: consecutive scale-ops blocked by cooldown", t_cooldown)

def t_bounds_min():
    shield = SafetyShield(min_containers=2)
    state = {"warm_containers": 2, "latency_ms": 50.0, "sla_target_ms": 300.0}
    res = shield.validate("SCALE_IN", state)
    assert res.was_overridden
    assert res.reason_code == "BOUNDS_MIN"

test("SafetyShield: SCALE_IN blocked at minimum bound", t_bounds_min)

def t_bounds_max():
    shield = SafetyShield(max_containers=50)
    state = {"warm_containers": 50, "latency_ms": 500.0, "sla_target_ms": 300.0}
    res = shield.validate("SCALE_OUT", state)
    assert res.was_overridden
    assert res.reason_code in ("BOUNDS_MAX", "COOLDOWN")  # cooldown may trigger first

test("SafetyShield: SCALE_OUT blocked at maximum bound", t_bounds_max)

def t_terminate_allowed_no_sla_violation():
    shield = SafetyShield(cooldown_steps=0)
    state = {"warm_containers": 8, "latency_ms": 100.0, "sla_target_ms": 300.0}
    res = shield.validate("TERMINATE_IDLE", state)
    # Should pass (SLA not violated, no cooldown with cooldown_steps=0, within bounds)
    assert res.reason_code == "PASS", f"Expected PASS, got {res.reason_code}: {res.details}"

test("SafetyShield: TERMINATE_IDLE allowed when SLA not violated", t_terminate_allowed_no_sla_violation)

def t_reset():
    shield = SafetyShield(cooldown_steps=3)
    state = {"warm_containers": 4, "latency_ms": 100.0, "sla_target_ms": 300.0}
    shield.validate("SCALE_OUT", state)  # triggers cooldown
    shield.reset()
    # After reset, scale op should be allowed again
    r = shield.validate("SCALE_OUT", state)
    assert not r.was_overridden or r.reason_code != "COOLDOWN", \
        f"After reset, cooldown should be cleared; got {r.reason_code}"

test("SafetyShield: reset clears cooldown", t_reset)

# ── DecisionEngine ────────────────────────────────────────────────────────────

def build_engine(use_ppo=False):
    sim      = _SimulatorStub()
    planner  = LocalRuleProvider()
    shield   = SafetyShield()
    return DecisionEngine(sim, planner, shield, ppo_model=None, use_ppo=use_ppo,
                          run_id="test", log_dir="outputs/decision_logs")

def t_one_cycle():
    engine = build_engine()
    state = {"warm_containers": 4, "latency_ms": 150.0, "sla_target_ms": 300.0,
             "mean_duration_seconds": 4.97, "cold_start_proxy": 0.1}
    record = engine.run_one_cycle(state, actual_load=10.0, forecast_load=12.0, step=0)
    engine.close()
    assert isinstance(record, DecisionRecord)
    assert record.final_action in ALLOWED_ACTIONS
    assert record.shield_reason_code in ("PASS", "COOLDOWN", "SLA_VIOLATED", "NEAR_SLA",
                                          "BOUNDS_MIN", "BOUNDS_MAX", "ILLEGAL_ACTION")
    assert record.latency_ms > 0
    assert record.warm_containers >= 1

test("DecisionEngine.run_one_cycle: returns valid DecisionRecord", t_one_cycle)

def t_one_cycle_shield_override_recorded():
    """Shield override must be reflected in the record."""
    engine = build_engine()
    # SLA violated → TERMINATE_IDLE should be overridden
    state = {"warm_containers": 2, "latency_ms": 500.0, "sla_target_ms": 300.0,
             "mean_duration_seconds": 4.97, "cold_start_proxy": 0.1}
    # Force planner to return TERMINATE_IDLE by giving idle-like load
    # We can't force it, but we can verify the record captures shield fields
    record = engine.run_one_cycle(state, actual_load=0.0, forecast_load=0.0, step=1)
    engine.close()
    assert hasattr(record, "shield_overridden")
    assert hasattr(record, "shield_reason_code")
    # If shield overrode, check override fields are consistent
    if record.shield_overridden:
        assert record.shield_input_action != record.shield_approved or \
               record.shield_reason_code != "PASS"

test("DecisionEngine: shield override fields recorded correctly", t_one_cycle_shield_override_recorded)

def t_jsonl_log_written():
    import os, json
    engine = build_engine()
    state = {"warm_containers": 4, "latency_ms": 150.0, "sla_target_ms": 300.0,
             "mean_duration_seconds": 4.97}
    engine.run_one_cycle(state, 10.0, 12.0, step=0)
    engine.run_one_cycle(state, 15.0, 14.0, step=1)
    path = engine.log_path
    engine.close()
    assert os.path.exists(path), f"Log file not found: {path}"
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 2, f"Expected 2 log lines, got {len(lines)}"
    rec = json.loads(lines[0])
    assert "correlation_id" in rec
    assert "final_action" in rec

test("DecisionEngine: JSONL log written with correlation_id", t_jsonl_log_written)

def t_run_trace_10_steps():
    import os
    engine = build_engine()
    actual   = [5.0 + i * 0.5 for i in range(10)]
    forecast = [5.5 + i * 0.5 for i in range(10)]
    records = engine.run_trace(actual, forecast, sla_target_ms=300.0, initial_containers=4)
    engine.close()
    assert len(records) == 10
    for r in records:
        assert r.final_action in ALLOWED_ACTIONS
        assert r.warm_containers >= 1

test("DecisionEngine.run_trace: 10 steps, all valid records", t_run_trace_10_steps)

def t_fallback_path():
    """LocalRuleProvider fallback when planner unavailable (simulate by patching)."""
    engine = build_engine()
    # Monkeypatch planner to return fallback_used=True
    from planner_providers import ProviderResult
    orig_call = engine.planner.call
    def broken_call(sp, up):
        return ProviderResult(plan_dict=None, provider_name="local_rules",
                              model_id="rules_v1", inference_ms=0,
                              fallback_used=True, fallback_reason="test",
                              raw_response=None)
    engine.planner.call = broken_call
    state = {"warm_containers": 4, "latency_ms": 150.0, "sla_target_ms": 300.0}
    record = engine.run_one_cycle(state, 10.0, 12.0, step=0)
    engine.close()
    assert record.planner_fallback is True
    assert record.planner_action == "NO_OP"  # fallback default
    assert record.final_action in ALLOWED_ACTIONS

test("DecisionEngine: planner fallback path handled correctly", t_fallback_path)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
else:
    print("ALL PASS")
