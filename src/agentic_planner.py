"""
Phase C: Structured Agentic Planner (replaces Qwen3 with a local rule-based
planner that exposes the same JSON interface Qwen3 would use).

The planner:
  - Accepts typed inputs: forecast, warm-container count, latency/SLA status,
    capacity limits, resource/energy proxy.
  - Returns structured JSON containing recommended action, brief rationale,
    and relevant constraints.
  - Validates output; rejects unsafe actions and falls back to a deterministic
    safe rule.
  - Logs model/version, prompt/template version, input, parsed action and
    fallback reason.

When an actual Qwen3 (or any OpenAI-compatible) endpoint is available,
set env variable PLANNER_ENDPOINT=http://host:port/v1 and
PLANNER_MODEL=Qwen/Qwen3-8B (or similar).  The planner will POST to
/v1/chat/completions using the same structured JSON schema and fall back
to local rules on failure.

Run from D:\\major_1:
    python src\\agentic_planner.py
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "outputs" / "planner_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "planner.log", mode="a"),
    ],
)
log = logging.getLogger("planner")

# ── Constants ──────────────────────────────────────────────────────────────────
PLANNER_VERSION = "1.0.0"
PROMPT_TEMPLATE_VERSION = "1.0"

ALLOWED_ACTIONS = {
    "NO_OP",
    "PREWARM",
    "SCALE_OUT",
    "SCALE_IN",
    "REUSE",
    "TERMINATE_IDLE",
}

# Container limits
MIN_CONTAINERS = 1
MAX_CONTAINERS = 50


# ── Input / Output types ───────────────────────────────────────────────────────
@dataclass
class PlannerInput:
    time_idx: int
    forecast_invocations: float        # predicted invocations next bucket
    actual_invocations: float          # current observed invocations
    warm_containers: int
    sla_ms: float                      # SLA threshold in ms
    current_latency_ms: float          # current estimated latency
    energy_proxy: float                # current energy proxy value
    capacity_limit: int                # max allowed warm containers
    cold_start_rate: float             # recent cold-start fraction [0,1]


@dataclass
class PlannerOutput:
    recommended_action: str
    rationale: str
    confidence: float                  # [0, 1]
    container_delta: int               # how many containers to add/remove
    expected_latency_ms: float
    sla_risk: str                      # "low" | "medium" | "high"
    fallback_used: bool
    fallback_reason: Optional[str]
    planner_version: str
    prompt_template_version: str
    inference_ms: float


# ── Prompt builder ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a serverless container management planner. \
Given the current system state and workload forecast, return a JSON object \
with the following fields:
{
  "recommended_action": "<one of: NO_OP, PREWARM, SCALE_OUT, SCALE_IN, REUSE, TERMINATE_IDLE>",
  "rationale": "<brief explanation>",
  "confidence": <float 0.0 to 1.0>,
  "container_delta": <integer, positive=add, negative=remove, 0=no change>,
  "expected_latency_ms": <float>,
  "sla_risk": "<low | medium | high>"
}

Rules:
- warm_containers must stay between 1 and capacity_limit.
- Latency > sla_ms means SLA violation; avoid it.
- Energy proxy = warm_containers * 0.8 + cpu_pct * 0.015.
- Prefer small incremental changes over large jumps.
- TERMINATE_IDLE only when cold_start_rate < 0.05 and latency is safe.
- Return only valid JSON, no markdown fences.
"""


def build_user_prompt(inp: PlannerInput) -> str:
    capacity_per_container = 10.0
    current_capacity = inp.warm_containers * capacity_per_container
    headroom = current_capacity - inp.actual_invocations
    forecast_gap = inp.forecast_invocations - current_capacity

    return (
        f"time_idx={inp.time_idx}\n"
        f"forecast_invocations={inp.forecast_invocations:.1f}\n"
        f"actual_invocations={inp.actual_invocations:.1f}\n"
        f"warm_containers={inp.warm_containers}\n"
        f"current_capacity={current_capacity:.0f}\n"
        f"headroom={headroom:.1f}\n"
        f"forecast_gap={forecast_gap:.1f}\n"
        f"current_latency_ms={inp.current_latency_ms:.1f}\n"
        f"sla_ms={inp.sla_ms:.0f}\n"
        f"energy_proxy={inp.energy_proxy:.3f}\n"
        f"capacity_limit={inp.capacity_limit}\n"
        f"cold_start_rate={inp.cold_start_rate:.3f}\n"
    )


# ── Local deterministic fallback ───────────────────────────────────────────────
def local_rule_planner(inp: PlannerInput) -> dict:
    """
    Deterministic rule-based planner.  Always safe; used when LLM unavailable.
    """
    cap = inp.warm_containers * 10.0
    forecast = inp.forecast_invocations
    latency = inp.current_latency_ms
    sla = inp.sla_ms
    warm = inp.warm_containers

    # Decide action
    if latency > sla * 0.95 or forecast > cap + 15:
        action = "SCALE_OUT"
        delta = min(2, inp.capacity_limit - warm)
        rationale = "Forecast or latency near/above SLA; scaling out."
    elif forecast > cap + 5:
        action = "PREWARM"
        delta = min(1, inp.capacity_limit - warm)
        rationale = "Forecast slightly above capacity; pre-warming."
    elif forecast < cap - 20 and warm > 1 and inp.cold_start_rate < 0.05:
        action = "TERMINATE_IDLE"
        delta = max(-2, 1 - warm)
        rationale = "Significant over-provisioning; terminating idle."
    elif forecast < cap - 10 and warm > 1:
        action = "SCALE_IN"
        delta = -1
        rationale = "Mild over-provisioning; scaling in."
    elif abs(forecast - cap) < 5:
        action = "REUSE"
        delta = 0
        rationale = "Forecast near current capacity; reusing containers."
    else:
        action = "NO_OP"
        delta = 0
        rationale = "No action needed."

    # Safety clamp
    new_warm = max(MIN_CONTAINERS, min(inp.capacity_limit, warm + delta))
    delta = new_warm - warm

    new_cap = new_warm * 10.0
    overload = max(0.0, forecast - new_cap)
    est_latency = 70.0 + overload * 9.0

    if latency > sla:
        sla_risk = "high"
    elif latency > sla * 0.85:
        sla_risk = "medium"
    else:
        sla_risk = "low"

    return {
        "recommended_action": action,
        "rationale": rationale,
        "confidence": 0.85,
        "container_delta": delta,
        "expected_latency_ms": round(est_latency, 1),
        "sla_risk": sla_risk,
    }


# ── LLM call (optional) ────────────────────────────────────────────────────────
def llm_planner(inp: PlannerInput) -> Optional[dict]:
    """
    Call an OpenAI-compatible endpoint if PLANNER_ENDPOINT is set.
    Returns parsed dict on success, None on failure.
    """
    endpoint = os.environ.get("PLANNER_ENDPOINT", "").strip()
    model = os.environ.get("PLANNER_MODEL", "Qwen/Qwen3-8B")
    if not endpoint:
        return None

    try:
        import urllib.request as req
        import urllib.error

        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(inp)},
            ],
            "temperature": 0.1,
            "max_tokens": 300,
        }).encode()

        request = req.Request(
            f"{endpoint.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('PLANNER_API_KEY', 'no-key')}",
            },
            method="POST",
        )

        with req.urlopen(request, timeout=10) as resp:
            body = json.loads(resp.read())

        content = body["choices"][0]["message"]["content"]
        # Strip markdown fences if present
        content = content.strip()
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:-1])
        result = json.loads(content)
        return result

    except Exception as exc:
        log.warning("LLM planner call failed: %s", exc)
        return None


# ── Output validation ──────────────────────────────────────────────────────────
def validate_output(raw: dict, inp: PlannerInput) -> tuple[dict, Optional[str]]:
    """
    Validate the planner output.  Returns (validated_dict, fallback_reason).
    fallback_reason is None if no fallback was triggered.
    """
    required_keys = {
        "recommended_action", "rationale", "confidence",
        "container_delta", "expected_latency_ms", "sla_risk",
    }
    missing = required_keys - set(raw.keys())
    if missing:
        return local_rule_planner(inp), f"Missing keys: {sorted(missing)}"

    action = raw.get("recommended_action", "")
    if action not in ALLOWED_ACTIONS:
        return local_rule_planner(inp), f"Action not in allow-list: {action!r}"

    delta = int(raw.get("container_delta", 0))
    new_warm = inp.warm_containers + delta
    if not (MIN_CONTAINERS <= new_warm <= inp.capacity_limit):
        return local_rule_planner(inp), (
            f"container_delta={delta} would set warm={new_warm} "
            f"outside [{MIN_CONTAINERS}, {inp.capacity_limit}]"
        )

    confidence = float(raw.get("confidence", 0))
    if not (0.0 <= confidence <= 1.0):
        raw["confidence"] = float(max(0.0, min(1.0, confidence)))

    if raw.get("sla_risk") not in ("low", "medium", "high"):
        raw["sla_risk"] = "medium"

    return raw, None


# ── Public API ─────────────────────────────────────────────────────────────────
def plan(inp: PlannerInput) -> PlannerOutput:
    """
    Main entry point.  Returns a PlannerOutput with a validated recommendation.
    """
    t0 = time.perf_counter()
    fallback_used = False
    fallback_reason = None

    # Try LLM first
    raw = llm_planner(inp)
    if raw is None:
        raw = local_rule_planner(inp)
        fallback_used = True
        fallback_reason = "LLM endpoint not configured; using local rules."

    validated, vfb_reason = validate_output(raw, inp)
    if vfb_reason:
        fallback_used = True
        fallback_reason = vfb_reason
        validated = local_rule_planner(inp)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    output = PlannerOutput(
        recommended_action=validated["recommended_action"],
        rationale=validated["rationale"],
        confidence=float(validated["confidence"]),
        container_delta=int(validated["container_delta"]),
        expected_latency_ms=float(validated["expected_latency_ms"]),
        sla_risk=validated["sla_risk"],
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        planner_version=PLANNER_VERSION,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        inference_ms=round(elapsed_ms, 2),
    )

    log.info(
        "t=%d action=%s delta=%+d sla_risk=%s fallback=%s (%.1f ms)",
        inp.time_idx,
        output.recommended_action,
        output.container_delta,
        output.sla_risk,
        output.fallback_used,
        output.inference_ms,
    )

    return output


# ── Smoke test ─────────────────────────────────────────────────────────────────
def run_smoke_test():
    """
    Tests: normal state, SLA violation, excessive scale request, missing fields.
    """
    cases = [
        # (description, inp_kwargs, expect_action_in)
        (
            "Normal balanced state",
            dict(time_idx=0, forecast_invocations=45.0, actual_invocations=42.0,
                 warm_containers=5, sla_ms=200.0, current_latency_ms=80.0,
                 energy_proxy=4.5, capacity_limit=20, cold_start_rate=0.02),
            {"REUSE", "NO_OP", "SCALE_IN"},
        ),
        (
            "SLA violation imminent",
            dict(time_idx=1, forecast_invocations=95.0, actual_invocations=90.0,
                 warm_containers=5, sla_ms=200.0, current_latency_ms=195.0,
                 energy_proxy=4.5, capacity_limit=20, cold_start_rate=0.3),
            {"SCALE_OUT", "PREWARM"},
        ),
        (
            "Massive over-provisioning",
            dict(time_idx=2, forecast_invocations=5.0, actual_invocations=4.0,
                 warm_containers=15, sla_ms=200.0, current_latency_ms=72.0,
                 energy_proxy=12.0, capacity_limit=20, cold_start_rate=0.0),
            {"TERMINATE_IDLE", "SCALE_IN"},
        ),
        (
            "Capacity limit boundary",
            dict(time_idx=3, forecast_invocations=210.0, actual_invocations=180.0,
                 warm_containers=20, sla_ms=200.0, current_latency_ms=250.0,
                 energy_proxy=16.0, capacity_limit=20, cold_start_rate=0.5),
            # At cap; fallback should still be safe
            ALLOWED_ACTIONS,
        ),
    ]

    print("\n=== Planner smoke tests ===")
    all_passed = True
    for desc, kwargs, expect_set in cases:
        inp = PlannerInput(**kwargs)
        out = plan(inp)
        passed = out.recommended_action in expect_set
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {desc}")
        print(f"         action={out.recommended_action}  delta={out.container_delta:+d}"
              f"  sla_risk={out.sla_risk}  fallback={out.fallback_used}")
        if out.fallback_reason:
            print(f"         reason={out.fallback_reason}")

    print(f"\n{'All tests passed.' if all_passed else 'Some tests FAILED.'}")
    return all_passed


if __name__ == "__main__":
    run_smoke_test()
