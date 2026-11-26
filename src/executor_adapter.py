"""
src/executor_adapter.py
========================
Milestone 6 — Execution Layer (Dry-Run Adapter).

Design rules (per IMPLEMENTATION_AGENT_PROMPT.md):
  - DryRunAdapter NEVER touches a real cluster.
  - Every execution attempt is recorded in an append-only audit log.
  - Actions must pass the ALLOWED_ACTIONS allowlist before execution.
  - Kubernetes adapter interface is stubbed; raises NotImplementedError by default.
  - audit log: outputs/executor_audit.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("executor.adapter")

ALLOWED_ACTIONS = {"NO_OP", "PREWARM", "SCALE_OUT", "SCALE_IN", "REUSE", "TERMINATE_IDLE"}


@dataclass
class ExecutionResult:
    action:          str
    approved:        bool          # passed allowlist check
    dry_run:         bool          # True for DryRunAdapter
    timestamp:       str           # ISO-8601 UTC
    correlation_id:  str           # links back to DecisionRecord
    reason:          str           # why approved or rejected
    error:           Optional[str] # None on success
    adapter_name:    str = "unknown"


class ExecutorAdapter(ABC):
    """Abstract base; concrete adapters implement `_execute`."""

    name: str = "abstract"
    dry_run: bool = True

    def __init__(self, audit_log_path: str = "outputs/executor_audit.jsonl"):
        os.makedirs(os.path.dirname(audit_log_path) if os.path.dirname(audit_log_path) else ".", exist_ok=True)
        self._audit_path = audit_log_path
        self._audit_fh   = open(audit_log_path, "a")

    def execute(self, action: str, state: dict, correlation_id: Optional[str] = None) -> ExecutionResult:
        """
        Execute `action` after allowlist check. Always writes to audit log.

        Parameters
        ----------
        action         : one of ALLOWED_ACTIONS (or will be rejected)
        state          : current system state dict (for logging context)
        correlation_id : links to the DecisionRecord that produced this action
        """
        cid = correlation_id or str(uuid.uuid4())[:8]
        ts  = datetime.now(timezone.utc).isoformat()
        err = None

        # ── Allowlist check ───────────────────────────────────────────────────
        if action not in ALLOWED_ACTIONS:
            reason = f"REJECTED: action {action!r} not in allowlist {sorted(ALLOWED_ACTIONS)}"
            log.warning("Executor rejected action %r (cid=%s)", action, cid)
            result = ExecutionResult(
                action=action, approved=False, dry_run=self.dry_run,
                timestamp=ts, correlation_id=cid, reason=reason,
                error="ILLEGAL_ACTION", adapter_name=self.name,
            )
            self._audit(result, state)
            return result

        # ── Dispatch to implementation ─────────────────────────────────────────
        try:
            reason = self._execute(action, state)
            approved = True
        except NotImplementedError as e:
            reason   = f"STUB: {e}"
            approved = False
            err      = "NOT_IMPLEMENTED"
        except Exception as e:
            reason   = f"ERROR: {type(e).__name__}: {e}"
            approved = False
            err      = str(e)
            log.exception("Executor._execute raised unexpected error (cid=%s)", cid)

        result = ExecutionResult(
            action=action, approved=approved, dry_run=self.dry_run,
            timestamp=ts, correlation_id=cid, reason=reason,
            error=err, adapter_name=self.name,
        )
        self._audit(result, state)
        return result

    @abstractmethod
    def _execute(self, action: str, state: dict) -> str:
        """Implement actual execution. Return reason string on success."""

    def _audit(self, result: ExecutionResult, state: dict) -> None:
        record = {**asdict(result), "state_snapshot": {
            k: state.get(k) for k in
            ("warm_containers", "latency_ms", "sla_target_ms", "invocation_count")
        }}
        self._audit_fh.write(json.dumps(record) + "\n")
        self._audit_fh.flush()

    def close(self):
        if self._audit_fh and not self._audit_fh.closed:
            self._audit_fh.close()


class DryRunAdapter(ExecutorAdapter):
    """
    Dry-run adapter: logs every action but NEVER touches a real cluster.
    Returns a simulated success result describing what would happen.
    """
    name    = "dry_run"
    dry_run = True

    # What each action "would do" in a real cluster
    _ACTION_DESCRIPTIONS = {
        "NO_OP":         "No change to cluster; idle monitoring tick.",
        "PREWARM":       "Would pre-warm +1 container (pull image, init runtime).",
        "SCALE_OUT":     "Would add +1 container to the warm pool.",
        "SCALE_IN":      "Would remove -1 container from the warm pool.",
        "REUSE":         "Would route next invocation to an existing warm container.",
        "TERMINATE_IDLE":"Would terminate the longest-idle container in the pool.",
    }

    def _execute(self, action: str, state: dict) -> str:
        desc = self._ACTION_DESCRIPTIONS.get(action, "Unknown action.")
        warm = state.get("warm_containers", "?")
        log.info("[DRY-RUN] %s | warm=%s | %s", action, warm, desc)
        return f"DRY-RUN OK: {desc}"


class KubernetesAdapter(ExecutorAdapter):
    """
    Kubernetes adapter interface stub.
    Raises NotImplementedError — do NOT remove this guard.
    Real implementation requires: cluster kubeconfig, RBAC, dry_run=False flag.
    """
    name    = "kubernetes"
    dry_run = True  # Always dry-run until explicitly overridden with care

    def __init__(self, namespace: str = "serverless", **kwargs):
        super().__init__(**kwargs)
        self.namespace = namespace

    def _execute(self, action: str, state: dict) -> str:
        raise NotImplementedError(
            f"KubernetesAdapter._execute not implemented for action={action!r}. "
            "This is an intentional stub to prevent accidental cluster modifications. "
            "See IMPLEMENTATION_PLAN.md §Executor for how to implement this safely."
        )
