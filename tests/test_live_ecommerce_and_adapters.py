"""
Tests for DockerAdapter, upgraded KubernetesAdapter, and Live E-Commerce Worker Pool.
Run directly with venv Python.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from executor_adapter import DockerAdapter, KubernetesAdapter
from ecommerce_app.orchestrator import LiveServerlessOrchestrator

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


def t_docker_adapter_dry_run():
    adapter = DockerAdapter(dry_run=True, audit_log_path="outputs/test_docker_audit.jsonl")
    for act in ("PREWARM", "SCALE_OUT", "SCALE_IN", "TERMINATE_IDLE", "REUSE", "NO_OP"):
        res = adapter.execute(act, {"warm_containers": 3})
        assert res.approved, f"{act} should be approved in dry_run"
        assert res.dry_run is True
        assert res.adapter_name == "docker"
    bad = adapter.execute("DROP_ALL", {"warm_containers": 3})
    assert not bad.approved and bad.error == "ILLEGAL_ACTION"
    adapter.close()


test("DockerAdapter: dry_run actions and allowlist", t_docker_adapter_dry_run)


def t_k8s_adapter_guarded_and_cmd():
    k8s_default = KubernetesAdapter(audit_log_path="outputs/test_k8s_audit.jsonl")
    res = k8s_default.execute("SCALE_OUT", {"warm_containers": 3})
    assert not res.approved and res.error == "NOT_IMPLEMENTED"
    k8s_default.close()

    k8s_cmd = KubernetesAdapter(namespace="prod", deployment="web", dry_run=True, allow_live=True,
                                audit_log_path="outputs/test_k8s_audit.jsonl")
    cmd = k8s_cmd._build_scale_cmd(5)
    assert cmd == ["kubectl", "scale", "deployment/web", "--replicas=5", "--namespace=prod", "--dry-run=client"]
    k8s_cmd.close()


test("KubernetesAdapter: default guard + kubectl scale command builder", t_k8s_adapter_guarded_and_cmd)


def t_live_orchestrator_ecommerce():
    orch = LiveServerlessOrchestrator(initial_workers=1)
    try:
        # 1. Warm catalog request
        res_cat = orch.invoke("catalog", {"category": "All"})
        assert res_cat["cold_start"] is False, f"Expected warm reuse, got cold_start={res_cat['cold_start']}"
        assert res_cat["data"]["count"] == 8

        # 2. Search request
        res_search = orch.invoke("search", {"q": "Laptop"})
        assert res_search["data"]["count"] >= 1

        # 3. Checkout request
        res_chk = orch.invoke("checkout", {"items": [{"price": 100.0, "qty": 2}]})
        assert res_chk["data"]["subtotal"] == 200.0
        assert res_chk["data"]["tax"] == 16.0
        assert res_chk["data"]["total"] == 216.0

        # 4. Mode switching
        assert orch.set_mode("reactive") == "reactive"
        assert orch.set_mode("predictive") == "predictive"

        # 5. Telemetry & Prometheus metrics
        tel = orch.get_telemetry()
        assert tel["total_requests"] >= 3
        assert tel["warm_containers"] >= 1
        prom = orch.prometheus_metrics()
        assert "shoppulse_warm_containers" in prom
        assert "shoppulse_mean_latency_ms" in prom
        assert "shoppulse_cold_starts_total" in prom
    finally:
        orch.shutdown()


test("LiveServerlessOrchestrator: real worker pool, catalog, search, checkout, /metrics", t_live_orchestrator_ecommerce)

print()
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
else:
    print("ALL PASS")
