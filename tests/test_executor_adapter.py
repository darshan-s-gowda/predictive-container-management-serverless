"""
Milestone 6 — Executor Adapter Tests
Run directly with venv Python (no pytest required).
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from executor_adapter import DryRunAdapter, KubernetesAdapter, ExecutionResult, ALLOWED_ACTIONS

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

AUDIT_PATH = "outputs/test_executor_audit.jsonl"
STATE = {"warm_containers": 4, "latency_ms": 150.0, "sla_target_ms": 300.0, "invocation_count": 10.0}

def make_adapter():
    return DryRunAdapter(audit_log_path=AUDIT_PATH)

# ── DryRunAdapter: all allowed actions ────────────────────────────────────────

def t_all_allowed_actions():
    adapter = make_adapter()
    for action in ALLOWED_ACTIONS:
        result = adapter.execute(action, STATE, correlation_id="test-cid")
        assert result.approved,     f"{action}: approved should be True"
        assert result.dry_run,      f"{action}: dry_run should be True"
        assert result.error is None, f"{action}: error should be None"
        assert result.action == action
    adapter.close()

test("DryRunAdapter: all allowed actions succeed", t_all_allowed_actions)

def t_illegal_action_rejected():
    adapter = make_adapter()
    result = adapter.execute("WIPE_CLUSTER", STATE, correlation_id="test-bad")
    assert not result.approved
    assert result.error == "ILLEGAL_ACTION"
    assert result.action == "WIPE_CLUSTER"
    adapter.close()

test("DryRunAdapter: illegal action rejected", t_illegal_action_rejected)

def t_dry_run_flag():
    adapter = make_adapter()
    result = adapter.execute("SCALE_OUT", STATE)
    assert result.dry_run is True
    adapter.close()

test("DryRunAdapter: dry_run flag is True", t_dry_run_flag)

def t_audit_log_written():
    # Clear audit log
    if os.path.exists(AUDIT_PATH):
        os.remove(AUDIT_PATH)
    adapter = make_adapter()
    adapter.execute("NO_OP",    STATE, correlation_id="cid-1")
    adapter.execute("PREWARM",  STATE, correlation_id="cid-2")
    adapter.execute("WIPE",     STATE, correlation_id="cid-3")  # bad action
    adapter.close()
    with open(AUDIT_PATH) as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 3, f"Expected 3 audit lines, got {len(lines)}"
    rec = json.loads(lines[0])
    assert rec["correlation_id"] == "cid-1"
    assert "timestamp" in rec
    assert "state_snapshot" in rec

test("DryRunAdapter: audit log written for every execute call", t_audit_log_written)

def t_correlation_id_in_result():
    adapter = make_adapter()
    result = adapter.execute("REUSE", STATE, correlation_id="my-unique-id")
    assert result.correlation_id == "my-unique-id"
    adapter.close()

test("DryRunAdapter: correlation_id preserved in result", t_correlation_id_in_result)

def t_auto_generates_correlation_id():
    adapter = make_adapter()
    result = adapter.execute("NO_OP", STATE)
    assert result.correlation_id is not None and len(result.correlation_id) > 0
    adapter.close()

test("DryRunAdapter: auto-generates correlation_id when not provided", t_auto_generates_correlation_id)

def t_adapter_name():
    adapter = make_adapter()
    result = adapter.execute("SCALE_IN", STATE)
    assert result.adapter_name == "dry_run"
    adapter.close()

test("DryRunAdapter: adapter_name is 'dry_run'", t_adapter_name)

# ── KubernetesAdapter: stub ───────────────────────────────────────────────────

def t_kubernetes_raises_not_implemented():
    k8s = KubernetesAdapter(audit_log_path="outputs/test_k8s_audit.jsonl")
    result = k8s.execute("SCALE_OUT", STATE)
    # Should NOT raise; wraps NotImplementedError and returns non-approved result
    assert not result.approved
    assert result.error == "NOT_IMPLEMENTED"
    k8s.close()

test("KubernetesAdapter: raises NotImplementedError stub → non-approved result", t_kubernetes_raises_not_implemented)

def t_kubernetes_dry_run():
    k8s = KubernetesAdapter(audit_log_path="outputs/test_k8s_audit.jsonl")
    result = k8s.execute("PREWARM", STATE)
    assert result.dry_run is True, "KubernetesAdapter should always be dry_run=True"
    k8s.close()

test("KubernetesAdapter: dry_run is True (never touches cluster)", t_kubernetes_dry_run)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
else:
    print("ALL PASS")
