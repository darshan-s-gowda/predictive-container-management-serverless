"""
src/executor_adapter.py
========================
Milestone 6 + Live Execution Layer — Executor Adapters.

Design rules (per IMPLEMENTATION_AGENT_PROMPT.md):
  - DryRunAdapter NEVER touches a real cluster.
  - DockerAdapter and KubernetesAdapter default to dry_run=True / allow_live=False
    to prevent accidental cluster/container modifications unless explicitly enabled.
  - Every execution attempt is recorded in an append-only audit log.
  - Actions must pass the ALLOWED_ACTIONS allowlist before execution.
  - audit log: outputs/executor_audit.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("executor.adapter")

ALLOWED_ACTIONS = {"NO_OP", "PREWARM", "SCALE_OUT", "SCALE_IN", "REUSE", "TERMINATE_IDLE"}


@dataclass
class ExecutionResult:
    action:          str
    approved:        bool          # passed allowlist check
    dry_run:         bool          # True unless live execution enabled
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
        parent = os.path.dirname(audit_log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._audit_path = audit_log_path
        self._audit_fh   = open(audit_log_path, "a", encoding="utf-8")

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


class DockerAdapter(ExecutorAdapter):
    """
    Docker container lifecycle adapter using the `docker` CLI.
    Defaults to dry_run=True; set dry_run=False when Docker daemon is running
    to manage real containers labeled `app=serverless-worker`.
    """
    name = "docker"

    def __init__(
        self,
        image: str = "python:3.11-slim",
        container_prefix: str = "serverless-worker",
        dry_run: bool = True,
        min_containers: int = 1,
        max_containers: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image = image
        self.container_prefix = container_prefix
        self.dry_run = dry_run
        self.min_containers = min_containers
        self.max_containers = max_containers

    def _list_workers(self) -> list[str]:
        proc = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"name={self.container_prefix}"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return [cid.strip() for cid in proc.stdout.splitlines() if cid.strip()]

    def _execute(self, action: str, state: dict) -> str:
        warm = int(state.get("warm_containers", 1))

        if action in ("NO_OP", "REUSE"):
            return f"{'DRY-RUN ' if self.dry_run else ''}DOCKER OK: {action} (warm={warm})"

        if action in ("PREWARM", "SCALE_OUT"):
            if warm >= self.max_containers:
                return f"DOCKER SKIP: already at max_containers={self.max_containers}"
            name = f"{self.container_prefix}-{uuid.uuid4().hex[:6]}"
            cmd = ["docker", "run", "-d", "--name", name, "--label", "app=serverless-worker",
                   self.image, "sleep", "3600"]
            if self.dry_run:
                return f"DRY-RUN DOCKER: would run {' '.join(cmd)}"
            subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
            return f"DOCKER LIVE OK: started container {name}"

        if action in ("SCALE_IN", "TERMINATE_IDLE"):
            if warm <= self.min_containers:
                return f"DOCKER SKIP: already at min_containers={self.min_containers}"
            if self.dry_run:
                return f"DRY-RUN DOCKER: would stop/remove 1 {self.container_prefix} container"
            workers = self._list_workers()
            if not workers:
                return "DOCKER LIVE SKIP: no running worker containers found"
            target = workers[-1]
            subprocess.run(["docker", "rm", "-f", target],
                           capture_output=True, text=True, timeout=15, check=True)
            return f"DOCKER LIVE OK: removed container {target}"

        return f"DOCKER OK: {action}"


class KubernetesAdapter(ExecutorAdapter):
    """
    Kubernetes Deployment scaling adapter using `kubectl`.
    By default (`allow_live=False`), raises NotImplementedError as a safety guard
    to prevent accidental cluster modifications.
    Pass `allow_live=True, dry_run=False` to execute real `kubectl scale` commands.
    """
    name = "kubernetes"

    def __init__(
        self,
        namespace: str = "serverless",
        deployment: str = "serverless-worker",
        dry_run: bool = True,
        allow_live: bool = False,
        min_replicas: int = 1,
        max_replicas: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.namespace = namespace
        self.deployment = deployment
        self.dry_run = dry_run
        self.allow_live = allow_live
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas

    def _build_scale_cmd(self, target_replicas: int) -> list[str]:
        cmd = [
            "kubectl", "scale",
            f"deployment/{self.deployment}",
            f"--replicas={target_replicas}",
            f"--namespace={self.namespace}",
        ]
        if self.dry_run:
            cmd.append("--dry-run=client")
        return cmd

    def _execute(self, action: str, state: dict) -> str:
        if not self.allow_live:
            raise NotImplementedError(
                f"KubernetesAdapter._execute live execution disabled for action={action!r}. "
                "Pass allow_live=True (and dry_run=False for real scaling) when connected to a cluster."
            )

        warm = int(state.get("warm_containers", 1))
        if action in ("PREWARM", "SCALE_OUT"):
            target = min(self.max_replicas, warm + 1)
        elif action in ("SCALE_IN", "TERMINATE_IDLE"):
            target = max(self.min_replicas, warm - 1)
        else:
            return f"K8S OK: {action} (replicas={warm})"

        cmd = self._build_scale_cmd(target)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=True)
        return f"K8S {'DRY-RUN' if self.dry_run else 'LIVE'} OK: {proc.stdout.strip() or ' '.join(cmd)}"
