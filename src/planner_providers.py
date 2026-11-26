"""
src/planner_providers.py
========================
Provider abstraction for the Qwen3 agentic planner.

Supports:
  - OllamaProvider  : local Ollama inference (recommended for development)
  - OpenAIProvider  : any OpenAI-compatible endpoint (vLLM, llama.cpp server, cloud)
  - LocalRuleProvider: deterministic fallback, always available

Configuration via environment variables (no secrets in source):
  PLANNER_BACKEND   : "ollama" | "openai" | "local"  (default: "local")
  PLANNER_ENDPOINT  : base URL   e.g. http://localhost:11434
  PLANNER_MODEL     : model tag  e.g. qwen3:8b  or  Qwen/Qwen3-8B
  PLANNER_TIMEOUT   : seconds    (default: 10)

Design rules (per IMPLEMENTATION_AGENT_PROMPT.md):
  - Strict structured output schema; reject anything outside it.
  - Allowlisted actions only.
  - Timeout, parse failure, refusal → deterministic fallback.
  - No secrets committed or logged.
  - LLM cannot override safety shield.
  - Prompt injection from workload metadata is tested explicitly.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("planner.providers")

ALLOWED_ACTIONS = {
    "NO_OP", "PREWARM", "SCALE_OUT", "SCALE_IN", "REUSE", "TERMINATE_IDLE",
}

REQUIRED_FIELDS = {
    "recommended_action", "rationale", "confidence",
    "container_delta", "expected_latency_ms", "sla_risk",
}

ALLOWED_SLA_RISKS = {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class PlannerSchemaError(ValueError):
    """Raised when LLM output fails schema or allowlist validation."""


def validate_plan_dict(raw: dict) -> dict:
    """
    Validate and sanitise a raw plan dict from an LLM response.

    Raises PlannerSchemaError on any violation so the caller can fall back.
    Never passes through arbitrary strings from LLM output.
    """
    missing = REQUIRED_FIELDS - set(raw.keys())
    if missing:
        raise PlannerSchemaError(f"Missing fields: {missing}")

    action = str(raw["recommended_action"]).strip().upper()
    if action not in ALLOWED_ACTIONS:
        raise PlannerSchemaError(
            f"Disallowed action: {raw['recommended_action']!r}. "
            f"Allowed: {ALLOWED_ACTIONS}"
        )

    try:
        confidence = float(raw["confidence"])
    except (TypeError, ValueError):
        raise PlannerSchemaError("confidence must be a float")
    if not (0.0 <= confidence <= 1.0):
        raise PlannerSchemaError(f"confidence out of range [0,1]: {confidence}")

    try:
        delta = int(raw["container_delta"])
    except (TypeError, ValueError):
        raise PlannerSchemaError("container_delta must be an integer")
    if abs(delta) > 10:
        raise PlannerSchemaError(f"container_delta too large: {delta} (max ±10)")

    try:
        latency = float(raw["expected_latency_ms"])
    except (TypeError, ValueError):
        raise PlannerSchemaError("expected_latency_ms must be a float")
    if latency < 0 or latency > 1_000_000:
        raise PlannerSchemaError(f"expected_latency_ms out of range: {latency}")

    sla_risk = str(raw["sla_risk"]).strip().lower()
    if sla_risk not in ALLOWED_SLA_RISKS:
        raise PlannerSchemaError(f"sla_risk must be one of {ALLOWED_SLA_RISKS}")

    # Sanitise rationale: truncate and strip control chars (prompt-injection defence)
    rationale = str(raw.get("rationale", ""))[:500]
    rationale = "".join(c for c in rationale if c.isprintable() or c in " \n")

    return {
        "recommended_action": action,
        "rationale": rationale,
        "confidence": confidence,
        "container_delta": delta,
        "expected_latency_ms": latency,
        "sla_risk": sla_risk,
    }


# ---------------------------------------------------------------------------
# Provider base class
# ---------------------------------------------------------------------------

@dataclass
class ProviderResult:
    plan_dict: Optional[dict]      # validated dict or None on failure
    provider_name: str
    model_id: str
    inference_ms: float
    fallback_used: bool
    fallback_reason: Optional[str]
    raw_response: Optional[str]    # first 500 chars, for audit


class BasePlannerProvider(ABC):
    """Abstract base; concrete providers implement `_call`."""

    name: str = "base"
    model_id: str = "none"

    def call(self, system_prompt: str, user_prompt: str) -> ProviderResult:
        t0 = time.perf_counter()
        raw = None
        try:
            raw = self._call(system_prompt, user_prompt)
            parsed = json.loads(raw)
            validated = validate_plan_dict(parsed)
            return ProviderResult(
                plan_dict=validated,
                provider_name=self.name,
                model_id=self.model_id,
                inference_ms=(time.perf_counter() - t0) * 1000,
                fallback_used=False,
                fallback_reason=None,
                raw_response=raw[:500] if raw else None,
            )
        except (urllib.error.URLError, ConnectionRefusedError, OSError) as e:
            reason = f"Connection error: {type(e).__name__}: {e}"
        except TimeoutError as e:
            reason = f"Timeout: {e}"
        except json.JSONDecodeError as e:
            reason = f"JSON parse error: {e} | raw={str(raw)[:200]}"
        except PlannerSchemaError as e:
            reason = f"Schema validation error: {e}"
        except Exception as e:
            reason = f"Unexpected error: {type(e).__name__}: {e}"

        log.warning("Provider %s failed: %s", self.name, reason)
        return ProviderResult(
            plan_dict=None,
            provider_name=self.name,
            model_id=self.model_id,
            inference_ms=(time.perf_counter() - t0) * 1000,
            fallback_used=True,
            fallback_reason=reason,
            raw_response=str(raw)[:500] if raw else None,
        )

    @abstractmethod
    def _call(self, system_prompt: str, user_prompt: str) -> str:
        """Return raw string response from the LLM."""


# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------

class OllamaProvider(BasePlannerProvider):
    """
    Local Ollama inference via /api/chat endpoint.

    Model must be pulled first: `ollama pull qwen3:8b`
    Set PLANNER_MODEL=qwen3:8b (or qwen3:4b for smaller hardware).
    Verify current availability at https://ollama.com/library/qwen3
    """

    name = "ollama"

    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        timeout: int = 10,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model_id = model
        self.timeout = timeout

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        payload = json.dumps({
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 300},
        }).encode()

        url = f"{self.endpoint}/api/chat"
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read())

        return body["message"]["content"]


# ---------------------------------------------------------------------------
# OpenAI-compatible provider (vLLM, llama.cpp, cloud)
# ---------------------------------------------------------------------------

class OpenAICompatibleProvider(BasePlannerProvider):
    """
    Any endpoint that speaks the OpenAI /v1/chat/completions API.
    Compatible with: vLLM, llama.cpp server, together.ai, cloud proxies.

    Set PLANNER_ENDPOINT=http://host:port/v1
    Set PLANNER_MODEL=Qwen/Qwen3-8B  (or whatever the endpoint serves)
    """

    name = "openai_compat"

    def __init__(
        self,
        endpoint: str,
        model: str = "Qwen/Qwen3-8B",
        timeout: int = 10,
        api_key: str = "",
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model_id = model
        self.timeout = timeout
        # API key from env only; never log it
        self._api_key = api_key

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        payload = json.dumps({
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 300,
        }).encode()

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self.endpoint}/chat/completions"
        req = urllib.request.Request(
            url, data=payload, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read())

        return body["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Local rule provider (deterministic fallback, always available)
# ---------------------------------------------------------------------------

class LocalRuleProvider(BasePlannerProvider):
    """
    Deterministic rule-based planner. Always available, zero latency.
    Used when no LLM endpoint is configured or reachable.
    """

    name = "local_rules"
    model_id = "rules_v1"

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        # Parse the structured user_prompt back into values
        kv: dict = {}
        for line in user_prompt.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                kv[k.strip()] = v.strip()

        def flt(k: str, default: float = 0.0) -> float:
            try:
                return float(kv.get(k, default))
            except ValueError:
                return default

        def intt(k: str, default: int = 0) -> int:
            try:
                return int(float(kv.get(k, default)))
            except ValueError:
                return default

        forecast = flt("forecast_invocations")
        cap      = flt("current_capacity", 40.0)
        latency  = flt("current_latency_ms")
        sla_ms   = flt("sla_ms", 200.0)
        warm     = intt("warm_containers", 4)
        cold_rate = flt("cold_start_rate")
        cap_limit = intt("capacity_limit", 50)

        if latency > sla_ms * 0.95 or forecast > cap + 15:
            action, delta = "SCALE_OUT", min(2, cap_limit - warm)
            rationale = "Forecast or latency near/above SLA; scaling out."
        elif forecast > cap + 5:
            action, delta = "PREWARM", min(1, cap_limit - warm)
            rationale = "Forecast slightly above capacity; pre-warming."
        elif forecast < cap - 20 and warm > 1 and cold_rate < 0.05:
            action, delta = "TERMINATE_IDLE", max(-2, 1 - warm)
            rationale = "Significant over-provisioning; terminating idle."
        elif forecast < cap - 10 and warm > 1:
            action, delta = "SCALE_IN", -1
            rationale = "Mild over-provisioning; scaling in."
        elif abs(forecast - cap) < 5:
            action, delta = "REUSE", 0
            rationale = "Forecast near current capacity; reusing."
        else:
            action, delta = "NO_OP", 0
            rationale = "No action needed."

        new_warm = max(1, min(cap_limit, warm + delta))
        delta = new_warm - warm
        new_cap = new_warm * 10.0
        overload = max(0.0, forecast - new_cap)
        est_latency = round(70.0 + overload * 9.0, 1)

        if latency > sla_ms:
            sla_risk = "high"
        elif latency > sla_ms * 0.85:
            sla_risk = "medium"
        else:
            sla_risk = "low"

        return json.dumps({
            "recommended_action": action,
            "rationale": rationale,
            "confidence": 0.85,
            "container_delta": delta,
            "expected_latency_ms": est_latency,
            "sla_risk": sla_risk,
        })


# ---------------------------------------------------------------------------
# Factory: build provider from environment
# ---------------------------------------------------------------------------

def build_provider_from_env() -> BasePlannerProvider:
    """
    Select and configure a provider from environment variables.

    PLANNER_BACKEND: ollama | openai | local  (default: local)
    PLANNER_ENDPOINT: URL base
    PLANNER_MODEL: model tag/name
    PLANNER_TIMEOUT: seconds (default 10)
    PLANNER_API_KEY: API key (never logged)
    """
    backend  = os.environ.get("PLANNER_BACKEND", "local").lower().strip()
    endpoint = os.environ.get("PLANNER_ENDPOINT", "").strip()
    model    = os.environ.get("PLANNER_MODEL", "")
    timeout  = int(os.environ.get("PLANNER_TIMEOUT", "10"))
    api_key  = os.environ.get("PLANNER_API_KEY", "")

    if backend == "ollama":
        ep = endpoint or "http://localhost:11434"
        m  = model or "qwen3:8b"
        log.info("Using OllamaProvider: endpoint=%s model=%s", ep, m)
        return OllamaProvider(endpoint=ep, model=m, timeout=timeout)

    if backend == "openai" or (backend == "local" and endpoint):
        ep = endpoint or "http://localhost:8000/v1"
        m  = model or "Qwen/Qwen3-8B"
        log.info("Using OpenAICompatibleProvider: endpoint=%s model=%s", ep, m)
        return OpenAICompatibleProvider(endpoint=ep, model=m, timeout=timeout, api_key=api_key)

    log.info("Using LocalRuleProvider (deterministic fallback)")
    return LocalRuleProvider()


# ---------------------------------------------------------------------------
# Offline provider experiment (compare rule vs LLM over a scenario)
# ---------------------------------------------------------------------------

def run_offline_planner_comparison(
    scenarios: list[dict],
    providers: list[BasePlannerProvider],
    system_prompt: str,
    user_prompt_builder,
) -> list[dict]:
    """
    Run each provider over all scenarios, return a list of result dicts.
    Used for offline planner evaluation without a live cluster.
    """
    rows = []
    for sc in scenarios:
        prompt = user_prompt_builder(sc)
        for prov in providers:
            t0 = time.perf_counter()
            result = prov.call(system_prompt, prompt)
            elapsed = (time.perf_counter() - t0) * 1000
            rows.append({
                "scenario_id": sc.get("id", ""),
                "provider": prov.name,
                "model_id": prov.model_id,
                "action": result.plan_dict["recommended_action"] if result.plan_dict else "FALLBACK",
                "confidence": result.plan_dict["confidence"] if result.plan_dict else 0.0,
                "sla_risk": result.plan_dict["sla_risk"] if result.plan_dict else "unknown",
                "fallback_used": result.fallback_used,
                "fallback_reason": result.fallback_reason or "",
                "inference_ms": round(elapsed, 1),
                "valid_output": not result.fallback_used,
            })
    return rows
