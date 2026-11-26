"""
src/decision_engine.py
======================
Decision Engine — integrates TFT forecast, agentic planner, PPO policy,
and safety shield into one auditable decision cycle.

Each cycle produces a DecisionRecord with full provenance:
  actual_load → forecast_load → PlannerOutput → PPO action → ShieldResult → final_action → metrics

Writes JSONL audit logs to outputs/decision_logs/decision_{run_id}.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any

import numpy as np

log = logging.getLogger("decision.engine")

# ── DecisionRecord ─────────────────────────────────────────────────────────────

@dataclass
class DecisionRecord:
    correlation_id:   str
    timestamp_step:   int
    actual_load:      float
    forecast_load:    float

    # Planner fields
    planner_provider: str
    planner_action:   str
    planner_confidence: float
    planner_sla_risk: str
    planner_delta:    int
    planner_rationale: str
    planner_fallback: bool

    # PPO fields
    ppo_obs_vec:      List[float]
    ppo_action_int:   int
    ppo_action_str:   str

    # Shield fields
    shield_input_action: str
    shield_approved:     str
    shield_reason_code:  str
    shield_overridden:   bool
    shield_details:      str

    # Final outcome (from simulator step)
    final_action:     str
    latency_ms:       float
    energy_proxy:     float
    sla_violation:    bool
    warm_containers:  int
    cold_start_proxy: float

    # Optional feedback
    feedback_note:    str = ""


# ── DecisionEngine ─────────────────────────────────────────────────────────────

class DecisionEngine:
    """
    Integrates simulator, planner, PPO, and shield for one-cycle or full-trace decisions.

    Parameters
    ----------
    simulator     : ServerlessSimulator instance (reset externally before traces)
    planner_prov  : BasePlannerProvider  (LocalRuleProvider or OllamaProvider)
    ppo_model     : stable_baselines3 PPO model, or None to skip PPO
    shield        : SafetyShield instance
    run_id        : unique identifier for this run (used in log filename)
    log_dir       : directory for JSONL decision logs
    use_ppo       : if False, planner action is used directly (shield still applied)
    """

    SYSTEM_PROMPT = (
        "You are a serverless container management planner. "
        "Return ONLY valid JSON with keys: recommended_action, rationale, confidence, "
        "container_delta, expected_latency_ms, sla_risk."
    )

    PPO_ACTION_MAP = {
        0: "NO_OP", 1: "PREWARM", 2: "SCALE_OUT",
        3: "SCALE_IN", 4: "REUSE", 5: "TERMINATE_IDLE",
    }

    def __init__(
        self,
        simulator,
        planner_prov,
        shield,
        ppo_model=None,
        run_id: Optional[str] = None,
        log_dir: str = "outputs/decision_logs",
        use_ppo: bool = True,
    ):
        self.sim      = simulator
        self.planner  = planner_prov
        self.shield   = shield
        self.ppo      = ppo_model
        self.run_id   = run_id or str(uuid.uuid4())[:8]
        self.log_dir  = log_dir
        self.use_ppo  = use_ppo and (ppo_model is not None)
        self._records: List[DecisionRecord] = []

        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, f"decision_{self.run_id}.jsonl")
        self._log_fh   = open(self._log_path, "w")

    # ── Public API ────────────────────────────────────────────────────────────

    def run_one_cycle(
        self,
        state_dict: dict,
        actual_load: float,
        forecast_load: float,
        step: int = 0,
    ) -> DecisionRecord:
        """
        Execute one full decision cycle.

        state_dict keys used:
          warm_containers, latency_ms, sla_target_ms, invocation_count, mean_duration_seconds
        """
        cid = str(uuid.uuid4())

        # ── 1. Build planner prompt ───────────────────────────────────────────
        warm = state_dict.get("warm_containers", 4)
        latency = state_dict.get("latency_ms", 0.0)
        sla_ms  = state_dict.get("sla_target_ms", 300.0)

        user_prompt = "\n".join([
            f"forecast_invocations={forecast_load * 100:.2f}",
            f"current_capacity={warm * 10:.1f}",
            f"current_latency_ms={latency:.1f}",
            f"sla_ms={sla_ms:.1f}",
            f"warm_containers={warm}",
            f"cold_start_rate={state_dict.get('cold_start_proxy', 0.1):.3f}",
            f"capacity_limit=50",
        ])
        planner_result = self.planner.call(self.SYSTEM_PROMPT, user_prompt)

        if planner_result.plan_dict:
            planner_action    = planner_result.plan_dict["recommended_action"]
            planner_conf      = planner_result.plan_dict["confidence"]
            planner_risk      = planner_result.plan_dict["sla_risk"]
            planner_delta     = planner_result.plan_dict["container_delta"]
            planner_rationale = planner_result.plan_dict["rationale"]
        else:
            planner_action    = "NO_OP"
            planner_conf      = 0.5
            planner_risk      = "medium"
            planner_delta     = 0
            planner_rationale = "Planner unavailable; fallback to NO_OP"

        # ── 2. PPO action ─────────────────────────────────────────────────────
        ppo_action_int = -1
        ppo_action_str = "N/A"
        ppo_obs_vec: List[float] = []

        if self.use_ppo:
            try:
                obs = self._build_ppo_obs(state_dict, actual_load, forecast_load, step)
                ppo_obs_vec = obs.tolist()
                action_int, _ = self.ppo.predict(obs, deterministic=True)
                ppo_action_int = int(action_int)
                ppo_action_str = self.PPO_ACTION_MAP.get(ppo_action_int, "NO_OP")
            except Exception as e:
                log.warning("PPO predict failed: %s", e)
                ppo_action_str = planner_action

        # ── 3. Choose action to shield ────────────────────────────────────────
        # Priority: PPO (if enabled) > planner
        candidate = ppo_action_str if self.use_ppo else planner_action

        # ── 4. Safety shield ──────────────────────────────────────────────────
        shield_state = {
            "warm_containers": warm,
            "latency_ms":      latency,
            "sla_target_ms":   sla_ms,
            "step":            step,
        }
        shield_res = self.shield.validate(candidate, shield_state)
        final_action = shield_res.approved_action

        # ── 5. Apply action to simulator ──────────────────────────────────────
        sim_metrics = self._apply_action(final_action, actual_load, state_dict)

        # ── 6. Build record ───────────────────────────────────────────────────
        record = DecisionRecord(
            correlation_id=cid,
            timestamp_step=step,
            actual_load=float(actual_load),
            forecast_load=float(forecast_load),
            planner_provider=self.planner.name,
            planner_action=planner_action,
            planner_confidence=planner_conf,
            planner_sla_risk=planner_risk,
            planner_delta=planner_delta,
            planner_rationale=planner_rationale,
            planner_fallback=planner_result.fallback_used,
            ppo_obs_vec=ppo_obs_vec,
            ppo_action_int=ppo_action_int,
            ppo_action_str=ppo_action_str,
            shield_input_action=candidate,
            shield_approved=final_action,
            shield_reason_code=shield_res.reason_code,
            shield_overridden=shield_res.was_overridden,
            shield_details=shield_res.details,
            final_action=final_action,
            latency_ms=sim_metrics["latency_ms"],
            energy_proxy=sim_metrics["energy_proxy"],
            sla_violation=sim_metrics["sla_violation"],
            warm_containers=sim_metrics["warm_containers"],
            cold_start_proxy=sim_metrics["cold_start_proxy"],
        )

        self._records.append(record)
        self._write_record(record)
        return record

    def run_trace(
        self,
        actual_arr: List[float],
        forecast_arr: List[float],
        sla_target_ms: float = 300.0,
        initial_containers: int = 4,
    ) -> List[DecisionRecord]:
        """Run a full trace of N steps through the decision engine."""
        self.shield.reset()
        self.sim.reset()

        records = []
        warm = initial_containers
        for step, (actual, forecast) in enumerate(zip(actual_arr, forecast_arr)):
            # Approximate latency from simulator physics (without stepping it yet)
            if warm > 0:
                capacity = warm * (300.0 / self.sim.mean_duration_seconds
                                   if hasattr(self.sim, 'mean_duration_seconds') else 60.0)
                overload = max(0.0, actual - capacity)
                latency = 70.0 + overload * 2.0 + (1500.0 if overload > 0 else 0.0)
            else:
                latency = 2000.0

            state_dict = {
                "warm_containers": warm,
                "latency_ms":      min(latency, 5000.0),
                "sla_target_ms":   sla_target_ms,
                "invocation_count": actual,
                "mean_duration_seconds": getattr(self.sim, 'mean_duration_seconds', 4.97),
                "cold_start_proxy": 0.1,
            }
            record = self.run_one_cycle(state_dict, actual, forecast, step=step)
            warm = record.warm_containers
            records.append(record)

        return records

    def close(self):
        """Flush and close the JSONL log."""
        if self._log_fh and not self._log_fh.closed:
            self._log_fh.flush()
            self._log_fh.close()
        log.info("Decision log written: %s (%d records)", self._log_path, len(self._records))

    @property
    def records(self) -> List[DecisionRecord]:
        return list(self._records)

    @property
    def log_path(self) -> str:
        return self._log_path

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_ppo_obs(self, state_dict, actual, forecast, step) -> np.ndarray:
        """Build 11-dim observation vector matching ppo_calibrated_env.py."""
        warm     = state_dict.get("warm_containers", 4)
        latency  = state_dict.get("latency_ms", 100.0)
        sla_ms   = state_dict.get("sla_target_ms", 300.0)
        inv_cnt  = actual
        mean_dur = state_dict.get("mean_duration_seconds", 4.97)

        concurrency = inv_cnt * mean_dur / 300.0
        cap_norm    = min(warm / 50.0, 1.0)
        load_norm   = min(concurrency / 100.0, 1.0)
        lat_norm    = min(latency / 2000.0, 1.0)
        sla_norm    = min(sla_ms / 1000.0, 1.0)
        fcst_norm   = min(forecast * mean_dur / 300.0 / 100.0, 1.0)
        sla_ratio   = min(latency / max(sla_ms, 1.0), 2.0) / 2.0
        cold_proxy  = float(state_dict.get("cold_start_proxy", 0.1))
        # Fixed approximations (no episode state available outside env)
        time_norm   = 0.5
        cooldown_n  = 0.5

        return np.array([
            cap_norm, load_norm, lat_norm, sla_norm, fcst_norm,
            sla_ratio, cold_proxy, 0.0, 0.0, time_norm, cooldown_n,
        ], dtype=np.float32)

    def _apply_action(self, action: str, actual_load: float, state_dict: dict) -> dict:
        """Apply action via simulator; return step metrics dict."""
        warm    = state_dict.get("warm_containers", 4)
        sla_ms  = state_dict.get("sla_target_ms", 300.0)
        mean_dur = state_dict.get("mean_duration_seconds", 4.97)

        # Adjust container count
        delta_map = {"SCALE_OUT": 1, "PREWARM": 1, "SCALE_IN": -1, "TERMINATE_IDLE": -1,
                     "REUSE": 0, "NO_OP": 0}
        delta    = delta_map.get(action, 0)
        new_warm = max(1, min(50, warm + delta))

        # Compute metrics using simulator physics
        capacity  = new_warm * (300.0 / max(mean_dur, 0.01))
        concurr   = actual_load * mean_dur / 300.0
        overload  = max(0.0, concurr - capacity)
        cold_rate = min(overload / max(concurr, 1.0), 1.0) if concurr > 0 else 0.0
        latency   = 70.0 + overload * 2.0 + (1500.0 if cold_rate > 0.1 else 0.0)
        latency   = min(latency, 5000.0)
        energy    = new_warm * 0.8 + max(0.0, concurr - capacity * 0.5) * 0.1

        return {
            "latency_ms":      round(latency, 2),
            "energy_proxy":    round(energy, 4),
            "sla_violation":   latency > sla_ms,
            "warm_containers": new_warm,
            "cold_start_proxy": round(cold_rate, 4),
        }

    def _write_record(self, record: DecisionRecord) -> None:
        d = asdict(record)
        self._log_fh.write(json.dumps(d) + "\n")
        self._log_fh.flush()


# ── Minimal simulator stub for standalone use ──────────────────────────────────

class _SimulatorStub:
    """Stand-in when real ServerlessSimulator is unavailable."""
    mean_duration_seconds = 4.97
    def reset(self): pass
