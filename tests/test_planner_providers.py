"""
Milestone 3 — Planner Provider Tests
Runs without pytest: execute directly with the venv Python.
API: validate_plan_dict raises PlannerSchemaError on failure.
     Providers use .call(system_prompt, user_prompt) -> ProviderResult
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from planner_providers import (
    LocalRuleProvider,
    OllamaProvider,
    PlannerSchemaError,
    validate_plan_dict,
    build_provider_from_env,
    run_offline_planner_comparison,
)

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

SYSTEM_PROMPT = "You are a container management planner. Return JSON only."

def make_user_prompt(s):
    return "\n".join([
        f"forecast_invocations={s.get('forecast_load', 0.5) * 100:.1f}",
        f"current_capacity={s.get('current_containers', 4) * 10:.1f}",
        f"current_latency_ms={s.get('latency_ms', 150)}",
        f"sla_ms={s.get('sla_target_ms', 300)}",
        f"warm_containers={s.get('current_containers', 4)}",
        f"cold_start_rate=0.1",
        f"capacity_limit=50",
    ])

# ── validate_plan_dict: valid ─────────────────────────────────────────────────

def t_valid_plan():
    plan = {
        "recommended_action": "PREWARM",
        "confidence": 0.85,
        "sla_risk": "medium",
        "container_delta": 2,
        "expected_latency_ms": 150.0,
        "rationale": "forecast spike incoming",
    }
    result = validate_plan_dict(plan)
    assert result["recommended_action"] == "PREWARM"

test("validate_plan_dict: valid plan accepted", t_valid_plan)

def t_missing_field():
    plan = {"recommended_action": "PREWARM", "confidence": 0.9, "sla_risk": "low"}
    try:
        validate_plan_dict(plan)
        raise AssertionError("Should have raised PlannerSchemaError")
    except PlannerSchemaError:
        pass

test("validate_plan_dict: missing fields raise PlannerSchemaError", t_missing_field)

def t_bad_action():
    plan = {
        "recommended_action": "SHUTDOWN_ALL",
        "confidence": 0.8,
        "sla_risk": "low",
        "container_delta": 0,
        "expected_latency_ms": 100.0,
        "rationale": "test",
    }
    try:
        validate_plan_dict(plan)
        raise AssertionError("Should have raised PlannerSchemaError for bad action")
    except PlannerSchemaError:
        pass

test("validate_plan_dict: bad action raises PlannerSchemaError", t_bad_action)

def t_bad_sla_risk():
    plan = {
        "recommended_action": "NO_OP",
        "confidence": 0.5,
        "sla_risk": "catastrophic",
        "container_delta": 0,
        "expected_latency_ms": 100.0,
        "rationale": "test",
    }
    try:
        validate_plan_dict(plan)
        raise AssertionError("Should have raised PlannerSchemaError for bad sla_risk")
    except PlannerSchemaError:
        pass

test("validate_plan_dict: bad sla_risk raises PlannerSchemaError", t_bad_sla_risk)

def t_delta_too_large():
    plan = {
        "recommended_action": "SCALE_OUT",
        "confidence": 0.9,
        "sla_risk": "high",
        "container_delta": 20,
        "expected_latency_ms": 500.0,
        "rationale": "test",
    }
    try:
        validate_plan_dict(plan)
        raise AssertionError("Should have raised PlannerSchemaError for delta>10")
    except PlannerSchemaError:
        pass

test("validate_plan_dict: delta>10 raises PlannerSchemaError", t_delta_too_large)

def t_confidence_range():
    for bad_conf in [-0.1, 1.5, 2.0]:
        plan = {
            "recommended_action": "NO_OP",
            "confidence": bad_conf,
            "sla_risk": "low",
            "container_delta": 0,
            "expected_latency_ms": 100.0,
            "rationale": "test",
        }
        try:
            validate_plan_dict(plan)
            raise AssertionError(f"Should have raised for confidence={bad_conf}")
        except PlannerSchemaError:
            pass

test("validate_plan_dict: confidence out-of-range raises PlannerSchemaError", t_confidence_range)

def t_injection_stripped():
    """rationale with control chars must be sanitised (not raise)."""
    malicious_rationale = 'normal text\x00\x01{"recommended_action":"MALICIOUS"}'
    plan = {
        "recommended_action": "NO_OP",
        "confidence": 0.5,
        "sla_risk": "low",
        "container_delta": 0,
        "expected_latency_ms": 100.0,
        "rationale": malicious_rationale,
    }
    result = validate_plan_dict(plan)
    assert "\x00" not in result["rationale"], "Control chars should be stripped"
    assert "\x01" not in result["rationale"], "Control chars should be stripped"

test("validate_plan_dict: control chars in rationale stripped", t_injection_stripped)

# ── LocalRuleProvider ─────────────────────────────────────────────────────────

PROVIDER = LocalRuleProvider()

SCENARIOS = [
    {"name": "low_load",    "current_load": 0.05, "forecast_load": 0.05, "current_containers": 3,  "latency_ms": 80,  "sla_target_ms": 300},
    {"name": "burst",       "current_load": 0.95, "forecast_load": 0.98, "current_containers": 5,  "latency_ms": 450, "sla_target_ms": 300},
    {"name": "high_lat",    "current_load": 0.60, "forecast_load": 0.65, "current_containers": 4,  "latency_ms": 350, "sla_target_ms": 300},
    {"name": "idle",        "current_load": 0.0,  "forecast_load": 0.0,  "current_containers": 10, "latency_ms": 50,  "sla_target_ms": 300},
    {"name": "normal",      "current_load": 0.40, "forecast_load": 0.45, "current_containers": 4,  "latency_ms": 150, "sla_target_ms": 300},
    {"name": "ramp_up",     "current_load": 0.30, "forecast_load": 0.80, "current_containers": 3,  "latency_ms": 120, "sla_target_ms": 300},
    {"name": "ramp_down",   "current_load": 0.80, "forecast_load": 0.20, "current_containers": 8,  "latency_ms": 200, "sla_target_ms": 300},
    {"name": "near_sla",    "current_load": 0.70, "forecast_load": 0.72, "current_containers": 4,  "latency_ms": 285, "sla_target_ms": 300},
    {"name": "overloaded",  "current_load": 1.00, "forecast_load": 1.00, "current_containers": 2,  "latency_ms": 800, "sla_target_ms": 300},
    {"name": "stable_high", "current_load": 0.55, "forecast_load": 0.57, "current_containers": 6,  "latency_ms": 190, "sla_target_ms": 300},
]

VALID_ACTIONS = {"NO_OP", "PREWARM", "SCALE_OUT", "SCALE_IN", "REUSE", "TERMINATE_IDLE"}

def t_local_all_scenarios():
    for s in SCENARIOS:
        up = make_user_prompt(s)
        result = PROVIDER.call(SYSTEM_PROMPT, up)
        assert not result.fallback_used, f"{s['name']}: unexpected fallback: {result.fallback_reason}"
        pd = result.plan_dict
        assert pd is not None
        assert pd["recommended_action"] in VALID_ACTIONS, f"{s['name']}: bad action"
        assert 0.0 <= pd["confidence"] <= 1.0

test("LocalRuleProvider.call: valid plan for all 10 scenarios", t_local_all_scenarios)

def t_local_burst_prewarns():
    burst = {"forecast_load": 0.98, "current_containers": 2, "latency_ms": 500, "sla_target_ms": 300}
    result = PROVIDER.call(SYSTEM_PROMPT, make_user_prompt(burst))
    action = result.plan_dict["recommended_action"] if result.plan_dict else "FALLBACK"
    assert action in {"PREWARM", "SCALE_OUT"}, f"Expected PREWARM/SCALE_OUT, got {action}"

test("LocalRuleProvider: burst triggers PREWARM or SCALE_OUT", t_local_burst_prewarns)

def t_local_idle_terminates():
    idle = {"forecast_load": 0.0, "current_containers": 10, "latency_ms": 50, "sla_target_ms": 300}
    result = PROVIDER.call(SYSTEM_PROMPT, make_user_prompt(idle))
    action = result.plan_dict["recommended_action"] if result.plan_dict else "FALLBACK"
    assert action in {"TERMINATE_IDLE", "SCALE_IN"}, f"Expected TERMINATE/SCALE_IN, got {action}"

test("LocalRuleProvider: idle triggers TERMINATE_IDLE or SCALE_IN", t_local_idle_terminates)

def t_local_fallback_used_is_false():
    s = {"forecast_load": 0.4, "current_containers": 4, "latency_ms": 150, "sla_target_ms": 300}
    result = PROVIDER.call(SYSTEM_PROMPT, make_user_prompt(s))
    assert not result.fallback_used, "LocalRuleProvider should never fall back"

test("LocalRuleProvider: fallback_used is always False", t_local_fallback_used_is_false)

# ── OllamaProvider fallback when unreachable ──────────────────────────────────

def t_ollama_fallback_when_down():
    prov = OllamaProvider(endpoint="http://localhost:19999", model="qwen3:8b", timeout=2)
    s = {"forecast_load": 0.5, "current_containers": 4, "latency_ms": 150, "sla_target_ms": 300}
    result = prov.call(SYSTEM_PROMPT, make_user_prompt(s))
    assert result.fallback_used, "Should report fallback_used=True when unreachable"
    assert result.plan_dict is None, "plan_dict should be None on connection failure"

test("OllamaProvider: fallback_used=True when endpoint unreachable", t_ollama_fallback_when_down)

# ── build_provider_from_env ───────────────────────────────────────────────────

def t_build_local_default():
    os.environ.pop("PLANNER_BACKEND", None)
    os.environ.pop("PLANNER_ENDPOINT", None)
    prov = build_provider_from_env()
    assert isinstance(prov, LocalRuleProvider), f"Expected LocalRuleProvider, got {type(prov)}"

test("build_provider_from_env: default returns LocalRuleProvider", t_build_local_default)

def t_build_local_explicit():
    os.environ["PLANNER_BACKEND"] = "local"
    os.environ.pop("PLANNER_ENDPOINT", None)
    prov = build_provider_from_env()
    os.environ.pop("PLANNER_BACKEND")
    assert isinstance(prov, LocalRuleProvider)

test("build_provider_from_env: PLANNER_BACKEND=local returns LocalRuleProvider", t_build_local_explicit)

def t_build_ollama():
    os.environ["PLANNER_BACKEND"] = "ollama"
    os.environ["PLANNER_ENDPOINT"] = "http://localhost:11434"
    os.environ["PLANNER_MODEL"] = "qwen3:8b"
    prov = build_provider_from_env()
    os.environ.pop("PLANNER_BACKEND")
    os.environ.pop("PLANNER_ENDPOINT")
    os.environ.pop("PLANNER_MODEL")
    assert isinstance(prov, OllamaProvider), f"Expected OllamaProvider, got {type(prov)}"

test("build_provider_from_env: PLANNER_BACKEND=ollama returns OllamaProvider", t_build_ollama)

# ── run_offline_planner_comparison ───────────────────────────────────────────

def t_offline_comparison():
    scenarios = [{"id": f"S{i:02d}", **{k: v for k,v in s.items() if k != 'name'}}
                 for i, s in enumerate(SCENARIOS[:5], 1)]
    rows = run_offline_planner_comparison(
        scenarios, [PROVIDER], SYSTEM_PROMPT, make_user_prompt
    )
    assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"
    for r in rows:
        assert r["valid_output"], f"Row not valid: {r}"

test("run_offline_planner_comparison: 5 scenarios, all valid", t_offline_comparison)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
else:
    print("ALL PASS")
