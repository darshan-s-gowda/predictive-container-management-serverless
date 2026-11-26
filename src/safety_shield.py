"""
src/safety_shield.py
====================
Safety Shield for the container management decision pipeline.

Rules (applied in priority order):
  1. Action must be in ALLOWED_ACTIONS allowlist.
  2. warm_containers must remain within [MIN_CONTAINERS, MAX_CONTAINERS].
  3. No TERMINATE_IDLE if SLA is currently violated (latency > sla_target_ms).
  4. No SCALE_IN if latency > 90% of SLA target.
  5. Cooldown: min COOLDOWN_STEPS between consecutive scale operations
     (SCALE_OUT, SCALE_IN, PREWARM, TERMINATE_IDLE).

Design:
  - Shield is deterministic; never raises.
  - Returns (approved_action, reason_code, was_overridden).
  - approved_action is always in ALLOWED_ACTIONS.
  - All overrides are logged to the caller via reason_code.
  - LLM/PPO output is passed through; shield has final say.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import logging

log = logging.getLogger("safety.shield")

ALLOWED_ACTIONS = {"NO_OP", "PREWARM", "SCALE_OUT", "SCALE_IN", "REUSE", "TERMINATE_IDLE"}
SCALE_OPS       = {"SCALE_OUT", "SCALE_IN", "PREWARM", "TERMINATE_IDLE"}

MIN_CONTAINERS  = 1
MAX_CONTAINERS  = 50
COOLDOWN_STEPS  = 2    # min steps between consecutive scale ops


@dataclass
class ShieldResult:
    approved_action: str
    was_overridden:  bool
    reason_code:     str    # e.g. "PASS", "ILLEGAL_ACTION", "SLA_VIOLATED", "COOLDOWN", "BOUNDS"
    original_action: str
    details:         str = ""


class SafetyShield:
    """
    Stateful safety shield tracking cooldown across steps.
    Call validate() once per decision cycle.
    """

    def __init__(
        self,
        min_containers: int = MIN_CONTAINERS,
        max_containers: int = MAX_CONTAINERS,
        cooldown_steps: int = COOLDOWN_STEPS,
    ):
        self.min_containers = min_containers
        self.max_containers = max_containers
        self.cooldown_steps = cooldown_steps
        self._steps_since_scale: int = cooldown_steps  # start ready

    def validate(self, action: str, state: dict) -> ShieldResult:
        """
        Validate `action` given current `state`.

        state keys (all optional with safe defaults):
          - warm_containers : int   (current warm container count)
          - latency_ms      : float (current observed latency)
          - sla_target_ms   : float (SLA threshold)
          - step            : int   (current time step, for logging)

        Returns ShieldResult with approved_action (always safe).
        """
        warm    = int(state.get("warm_containers", 4))
        latency = float(state.get("latency_ms", 0.0))
        sla_ms  = float(state.get("sla_target_ms", 300.0))
        step    = int(state.get("step", 0))

        # ── Rule 1: allowlist ────────────────────────────────────────────────
        if action not in ALLOWED_ACTIONS:
            log.warning("step=%d: ILLEGAL action %r → NO_OP", step, action)
            self._update_cooldown("NO_OP")
            return ShieldResult(
                approved_action="NO_OP",
                was_overridden=True,
                reason_code="ILLEGAL_ACTION",
                original_action=action,
                details=f"Action {action!r} not in allowlist; substituted NO_OP",
            )

        # ── Rule 3: no TERMINATE_IDLE if SLA violated ────────────────────────
        sla_violated = latency > sla_ms
        if action == "TERMINATE_IDLE" and sla_violated:
            safe = "NO_OP"
            log.warning("step=%d: TERMINATE_IDLE blocked (SLA violated lat=%.0f ms > %.0f ms) → %s",
                        step, latency, sla_ms, safe)
            self._update_cooldown(safe)
            return ShieldResult(
                approved_action=safe,
                was_overridden=True,
                reason_code="SLA_VIOLATED",
                original_action=action,
                details=f"TERMINATE_IDLE blocked: latency {latency:.0f}ms > SLA {sla_ms:.0f}ms",
            )

        # ── Rule 4: no SCALE_IN if latency > 90% SLA ────────────────────────
        near_sla = latency > sla_ms * 0.9
        if action == "SCALE_IN" and near_sla:
            safe = "NO_OP"
            log.warning("step=%d: SCALE_IN blocked (latency near SLA %.0f ms > %.0f ms) → %s",
                        step, latency, sla_ms * 0.9, safe)
            self._update_cooldown(safe)
            return ShieldResult(
                approved_action=safe,
                was_overridden=True,
                reason_code="NEAR_SLA",
                original_action=action,
                details=f"SCALE_IN blocked: latency {latency:.0f}ms > 90% SLA {sla_ms*0.9:.0f}ms",
            )

        # ── Rule 5: cooldown between scale operations ─────────────────────────
        if action in SCALE_OPS and self._steps_since_scale < self.cooldown_steps:
            safe = "NO_OP"
            log.warning("step=%d: %s blocked (cooldown %d/%d) → %s",
                        step, action, self._steps_since_scale, self.cooldown_steps, safe)
            self._update_cooldown(safe)
            return ShieldResult(
                approved_action=safe,
                was_overridden=True,
                reason_code="COOLDOWN",
                original_action=action,
                details=f"Scale-op cooldown: {self._steps_since_scale}/{self.cooldown_steps} steps elapsed",
            )

        # ── Rule 2: container bounds ─────────────────────────────────────────
        # Predict resulting container count
        delta_map = {"SCALE_OUT": 1, "PREWARM": 1, "SCALE_IN": -1, "TERMINATE_IDLE": -1,
                     "REUSE": 0, "NO_OP": 0}
        delta = delta_map.get(action, 0)
        new_warm = warm + delta
        if new_warm < self.min_containers:
            safe = "NO_OP"
            log.warning("step=%d: %s blocked (would reach %d < min %d) → NO_OP",
                        step, action, new_warm, self.min_containers)
            self._update_cooldown(safe)
            return ShieldResult(
                approved_action=safe,
                was_overridden=True,
                reason_code="BOUNDS_MIN",
                original_action=action,
                details=f"Would reduce containers to {new_warm} < min {self.min_containers}",
            )
        if new_warm > self.max_containers:
            safe = "NO_OP"
            log.warning("step=%d: %s blocked (would reach %d > max %d) → NO_OP",
                        step, action, new_warm, self.max_containers)
            self._update_cooldown(safe)
            return ShieldResult(
                approved_action=safe,
                was_overridden=True,
                reason_code="BOUNDS_MAX",
                original_action=action,
                details=f"Would exceed containers to {new_warm} > max {self.max_containers}",
            )

        # ── All rules passed ─────────────────────────────────────────────────
        self._update_cooldown(action)
        return ShieldResult(
            approved_action=action,
            was_overridden=False,
            reason_code="PASS",
            original_action=action,
            details="All safety rules passed",
        )

    def _update_cooldown(self, approved: str) -> None:
        if approved in SCALE_OPS:
            self._steps_since_scale = 0
        else:
            self._steps_since_scale = min(
                self._steps_since_scale + 1, self.cooldown_steps + 1
            )

    def reset(self) -> None:
        """Reset cooldown counter (call between independent traces)."""
        self._steps_since_scale = self.cooldown_steps
