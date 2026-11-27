"""
src/ecommerce_app/orchestrator.py
=================================
Live Serverless Worker Pool & Predictive Container Orchestrator.

Manages real HTTP worker subprocesses (src/ecommerce_app/worker.py),
routes live e-commerce requests, measures real wall-clock latency and cold starts,
and runs the Predictive Control Loop:
  Traffic Telemetry -> Forecast -> Planner (Qwen3/LocalRule) -> PPO -> SafetyShield -> Worker Pool
"""

from __future__ import annotations

import atexit
import collections
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from planner_providers import build_provider_from_env, LocalRuleProvider
from safety_shield import SafetyShield

WORKER_SCRIPT = str(Path(__file__).resolve().parent / "worker.py")
PPO_MODEL_PATH = ROOT / "outputs" / "ppo_container_controller_calibrated.zip"


@dataclass
class WorkerInstance:
    worker_id: str
    port: int
    proc: subprocess.Popen
    state: str  # "warm" | "busy" | "terminated"
    created_at: float
    last_used_at: float
    request_count: int = 0
    boot_ms: float = 0.0

    def to_public_dict(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "port": self.port,
            "state": self.state,
            "uptime_s": round(time.time() - self.created_at, 1),
            "idle_s": round(time.time() - self.last_used_at, 1),
            "request_count": self.request_count,
            "boot_ms": round(self.boot_ms, 1),
        }


class LiveServerlessOrchestrator:
    """
    Coordinates real worker subprocesses and the predictive scaling loop.
    """

    SLA_MS = 300.0
    MIN_WORKERS = 1
    MAX_WORKERS = 16
    COLD_START_INIT_MS = 900  # Realistic runtime boot delay per new container

    def __init__(self, initial_workers: int = 2):
        self._lock = threading.RLock()
        self._workers: Dict[str, WorkerInstance] = {}
        self._mode = "predictive"  # "predictive" | "reactive"
        self._running = True

        # Forecast / burst signal (set when flash sale is scheduled/active)
        self._scheduled_spike_until: float = 0.0
        self._predicted_rps: float = 0.5

        # Telemetry history
        self._req_timestamps: Deque[float] = collections.deque(maxlen=500)
        self._latencies: Deque[float] = collections.deque(maxlen=200)
        self._recent_requests: Deque[dict] = collections.deque(maxlen=50)
        self._decisions: Deque[dict] = collections.deque(maxlen=40)
        self._timeline: Deque[dict] = collections.deque(maxlen=60)

        self._total_requests = 0
        self._total_cold_starts = 0
        self._total_sla_violations = 0
        self._shield_overrides = 0
        self._step_counter = 0

        # Planner + Shield + optional PPO
        self.planner = build_provider_from_env()
        self._fallback_planner = LocalRuleProvider()
        self.shield = SafetyShield(
            min_containers=self.MIN_WORKERS,
            max_containers=self.MAX_WORKERS,
            cooldown_steps=1,
        )
        self.ppo_model = self._load_ppo()

        # Decision log
        log_dir = ROOT / "outputs" / "decision_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._decision_log_path = log_dir / "live_ecommerce_decisions.jsonl"

        atexit.register(self.shutdown)

        # Boot initial warm pool
        for _ in range(initial_workers):
            try:
                self._spawn_worker_sync()
            except Exception as e:
                print(f"[WARN] Initial worker spawn failed: {e}")

        # Background control loop
        self._ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._ctrl_thread.start()

    def _load_ppo(self):
        if PPO_MODEL_PATH.exists():
            try:
                from stable_baselines3 import PPO
                return PPO.load(str(PPO_MODEL_PATH))
            except Exception:
                return None
        return None

    # ── Worker lifecycle ──────────────────────────────────────────────────────

    def _spawn_worker_sync(self) -> WorkerInstance:
        t0 = time.perf_counter()
        wid = f"ctr-{uuid.uuid4().hex[:5]}"
        env = os.environ.copy()
        env["WORKER_ID"] = wid
        env["COLD_START_INIT_MS"] = str(self.COLD_START_INIT_MS)

        proc = subprocess.Popen(
            [sys.executable, WORKER_SCRIPT, "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )

        # Wait for READY:<port>
        port = None
        deadline = time.time() + 10.0
        while time.time() < deadline:
            line = proc.stdout.readline()
            if line.startswith("READY:"):
                port = int(line.split(":", 1)[1].strip())
                break
            if proc.poll() is not None:
                raise RuntimeError(f"Worker {wid} exited prematurely")

        if port is None:
            proc.kill()
            raise TimeoutError(f"Worker {wid} timed out waiting for READY")

        boot_ms = (time.perf_counter() - t0) * 1000.0
        now = time.time()
        inst = WorkerInstance(
            worker_id=wid,
            port=port,
            proc=proc,
            state="warm",
            created_at=now,
            last_used_at=now,
            boot_ms=boot_ms,
        )
        with self._lock:
            self._workers[wid] = inst
        return inst

    def _terminate_one_idle(self) -> Optional[str]:
        with self._lock:
            warm_list = [w for w in self._workers.values() if w.state == "warm"]
            if len(self._workers) <= self.MIN_WORKERS or not warm_list:
                return None
            # Terminate longest-idle warm worker
            target = min(warm_list, key=lambda w: w.last_used_at)
            target.state = "terminated"
            del self._workers[target.worker_id]
        try:
            target.proc.terminate()
            target.proc.wait(timeout=2)
        except Exception:
            try:
                target.proc.kill()
            except Exception:
                pass
        return target.worker_id

    def shutdown(self):
        self._running = False
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for w in workers:
            try:
                w.proc.terminate()
            except Exception:
                pass

    # ── Request Invocation ────────────────────────────────────────────────────

    def invoke(self, fn_name: str, payload: dict) -> dict:
        """
        Route an e-commerce HTTP request to a warm worker, or cold-start a new one.
        Measures real wall-clock end-to-end latency.
        """
        t_start = time.perf_counter()
        cold_start = False
        chosen: Optional[WorkerInstance] = None

        with self._lock:
            warm_candidates = [w for w in self._workers.values() if w.state == "warm"]
            if warm_candidates:
                chosen = min(warm_candidates, key=lambda w: w.request_count)
                chosen.state = "busy"

        if chosen is None:
            # Cold start required!
            cold_start = True
            chosen = self._spawn_worker_sync()
            with self._lock:
                if chosen.worker_id in self._workers:
                    chosen.state = "busy"

        try:
            req_data = json.dumps({"fn": fn_name, "payload": payload}).encode("utf-8")
            url = f"http://127.0.0.1:{chosen.port}/invoke"
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                worker_resp = json.loads(resp.read().decode("utf-8"))
        finally:
            with self._lock:
                if chosen.worker_id in self._workers:
                    chosen.state = "warm"
                    chosen.last_used_at = time.time()
                    chosen.request_count += 1

        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        sla_viol = total_ms > self.SLA_MS

        with self._lock:
            now = time.time()
            self._total_requests += 1
            if cold_start:
                self._total_cold_starts += 1
            if sla_viol:
                self._total_sla_violations += 1
            self._req_timestamps.append(now)
            self._latencies.append(total_ms)
            self._recent_requests.appendleft({
                "id": uuid.uuid4().hex[:6],
                "fn": fn_name,
                "worker_id": chosen.worker_id,
                "latency_ms": total_ms,
                "cold_start": cold_start,
                "sla_violated": sla_viol,
                "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            })

        return {
            "worker_id": chosen.worker_id,
            "cold_start": cold_start,
            "latency_ms": total_ms,
            "sla_violated": sla_viol,
            "data": worker_resp.get("result", {}),
        }

    # ── Predictive Control Loop ───────────────────────────────────────────────

    def set_mode(self, mode: str) -> str:
        if mode not in ("predictive", "reactive"):
            raise ValueError("mode must be 'predictive' or 'reactive'")
        with self._lock:
            self._mode = mode
            self._scheduled_spike_until = 0.0
        # Trim excess idle workers back to 2 for fair Predictive vs Reactive comparison
        while len(self._workers) > 2:
            if not self._terminate_one_idle():
                break
        return mode

    def notify_upcoming_spike(self, duration_s: float = 12.0, expected_rps: float = 12.0):
        """
        Simulates the TFT / workload forecaster detecting an upcoming traffic burst
        (e.g. flash-sale announcement / scheduled spike) 2-3 seconds ahead.
        """
        with self._lock:
            self._scheduled_spike_until = time.time() + duration_s
            self._predicted_rps = expected_rps
        # Immediately run one control step in predictive mode so prewarming reacts fast
        if self._mode == "predictive":
            self._run_control_step()

    def _current_rps(self) -> float:
        now = time.time()
        with self._lock:
            recent = [t for t in self._req_timestamps if now - t <= 5.0]
        return len(recent) / 5.0

    def _run_control_step(self):
        with self._lock:
            self._step_counter += 1
            step = self._step_counter
            mode = self._mode
            warm_count = len(self._workers)
            lat_list = list(self._latencies)[-20:]
            avg_lat = float(np.mean(lat_list)) if lat_list else 25.0
            rps = self._current_rps()
            spike_active = time.time() < self._scheduled_spike_until
            forecast_rps = max(rps * 1.15, self._predicted_rps if spike_active else rps)

        # In reactive mode, no forecast is used (forecast == current rps, no prewarm)
        if mode == "reactive":
            forecast_rps = rps

        # Map RPS to simulated invocation scale for planner/PPO compatibility
        capacity = warm_count * 3.0  # ~3 req/s comfortable capacity per warm worker
        forecast_load = forecast_rps * 3.5

        user_prompt = "\n".join([
            f"forecast_invocations={forecast_load:.2f}",
            f"current_capacity={capacity * 3.5:.1f}",
            f"current_latency_ms={avg_lat:.1f}",
            f"sla_ms={self.SLA_MS:.1f}",
            f"warm_containers={warm_count}",
            f"cold_start_rate={self._total_cold_starts / max(self._total_requests, 1):.3f}",
            f"capacity_limit={self.MAX_WORKERS}",
        ])
        sys_prompt = "Return ONLY JSON with recommended_action, rationale, confidence, container_delta, expected_latency_ms, sla_risk."

        prov_res = self.planner.call(sys_prompt, user_prompt)
        used_fallback = prov_res.fallback_used
        if prov_res.plan_dict is None:
            prov_res = self._fallback_planner.call(sys_prompt, user_prompt)
            used_fallback = True

        plan = prov_res.plan_dict or {"recommended_action": "NO_OP", "confidence": 0.5, "sla_risk": "low", "rationale": "fallback"}
        planner_action = plan["recommended_action"]

        # Choose candidate action depending on mode
        if mode == "reactive":
            # Reactive mode: only scales out AFTER latency violates SLA, scales in if idle
            if avg_lat > self.SLA_MS:
                candidate = "SCALE_OUT"
            elif rps < 0.2 and warm_count > 1:
                candidate = "TERMINATE_IDLE"
            else:
                candidate = "NO_OP"
            ppo_action = "DISABLED(reactive)"
        else:
            # Predictive mode: combine TFT forecast signal + Planner + PPO
            ppo_action = "N/A"
            if self.ppo_model is not None:
                obs = np.array([
                    min(warm_count / 50.0, 1.0),
                    min(rps / 20.0, 1.0),
                    min(avg_lat / 2000.0, 1.0),
                    min(self.SLA_MS / 1000.0, 1.0),
                    min(forecast_rps / 20.0, 1.0),
                    min(avg_lat / self.SLA_MS, 2.0) / 2.0,
                    min(self._total_cold_starts / max(self._total_requests, 1), 1.0),
                    0.0, 0.0, 0.5, 0.5,
                ], dtype=np.float32)
                a_int, _ = self.ppo_model.predict(obs, deterministic=True)
                amap = {0: "NO_OP", 1: "PREWARM", 2: "SCALE_OUT", 3: "SCALE_IN", 4: "REUSE", 5: "TERMINATE_IDLE"}
                ppo_action = amap.get(int(a_int), "NO_OP")

            # Proactive prewarm rule when upcoming spike is forecasted
            if spike_active and warm_count < 6:
                candidate = "PREWARM"
            elif planner_action in ("PREWARM", "SCALE_OUT") and warm_count < 8:
                candidate = planner_action
            elif not spike_active and rps < 0.3 and warm_count > 2:
                candidate = "TERMINATE_IDLE"
            else:
                candidate = planner_action if planner_action != "NO_OP" else ppo_action

        # Safety Shield check
        shield_res = self.shield.validate(candidate, {
            "warm_containers": warm_count,
            "latency_ms": avg_lat,
            "sla_target_ms": self.SLA_MS,
            "step": step,
        })
        final_action = shield_res.approved_action

        # Execute on real worker pool
        exec_note = "no-op"
        if final_action == "PREWARM" and spike_active:
            target = min(10, self.MAX_WORKERS, max(warm_count + 1, int(np.ceil(forecast_rps / 2.0))))
            to_spawn = max(1, target - warm_count)
            try:
                import concurrent.futures as _cf
                with _cf.ThreadPoolExecutor(max_workers=to_spawn) as _pool:
                    spawned = list(_pool.map(lambda _: self._spawn_worker_sync(), range(to_spawn)))
                exec_note = f"prewarmed +{len(spawned)} containers (target={target})"
            except Exception as e:
                exec_note = f"prewarm failed: {e}"
        elif final_action in ("PREWARM", "SCALE_OUT"):
            try:
                new_w = self._spawn_worker_sync()
                exec_note = f"spawned {new_w.worker_id} (boot {new_w.boot_ms:.0f}ms)"
            except Exception as e:
                exec_note = f"spawn failed: {e}"
        elif final_action in ("SCALE_IN", "TERMINATE_IDLE"):
            removed = self._terminate_one_idle()
            exec_note = f"terminated {removed}" if removed else "none idle"
        else:
            exec_note = f"{final_action} (pool={warm_count})"

        with self._lock:
            if shield_res.was_overridden:
                self._shield_overrides += 1
            new_warm_count = len(self._workers)
            energy_proxy = round(new_warm_count * 0.8 + rps * 0.15, 3)

            dec = {
                "correlation_id": uuid.uuid4().hex[:8],
                "step": step,
                "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "mode": mode,
                "rps": round(rps, 2),
                "forecast_rps": round(forecast_rps, 2),
                "planner_action": planner_action,
                "ppo_action": ppo_action,
                "shield_approved": final_action,
                "shield_reason": shield_res.reason_code,
                "shield_overridden": shield_res.was_overridden,
                "exec_note": exec_note,
                "warm_containers": new_warm_count,
                "avg_latency_ms": round(avg_lat, 1),
                "energy_proxy": energy_proxy,
                "planner_fallback": used_fallback,
            }
            self._decisions.appendleft(dec)
            self._timeline.append({
                "ts": dec["ts"],
                "rps": dec["rps"],
                "forecast_rps": dec["forecast_rps"],
                "avg_latency_ms": dec["avg_latency_ms"],
                "warm_containers": new_warm_count,
                "energy_proxy": energy_proxy,
            })

        try:
            with open(self._decision_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dec) + "\n")
        except Exception:
            pass

    def _control_loop(self):
        while self._running:
            time.sleep(2.0)
            if not self._running:
                break
            try:
                self._run_control_step()
            except Exception:
                pass

    # ── Telemetry & Prometheus ────────────────────────────────────────────────

    def get_telemetry(self) -> dict:
        with self._lock:
            lats = list(self._latencies)
            mean_lat = round(float(np.mean(lats)), 1) if lats else 0.0
            p95_lat = round(float(np.percentile(lats, 95)), 1) if lats else 0.0
            workers = [w.to_public_dict() for w in self._workers.values()]
            rps = self._current_rps()
            warm_n = len(workers)
            energy_proxy = round(warm_n * 0.8 + rps * 0.15, 3)

            return {
                "mode": self._mode,
                "planner_backend": f"{self.planner.name} ({'Ollama offline fallback' if self.planner.name == 'local_rules' else self.planner.model_id})",
                "ppo_loaded": self.ppo_model is not None,
                "sla_ms": self.SLA_MS,
                "total_requests": self._total_requests,
                "total_cold_starts": self._total_cold_starts,
                "cold_start_rate_pct": round(self._total_cold_starts / max(self._total_requests, 1) * 100, 1),
                "total_sla_violations": self._total_sla_violations,
                "sla_violation_rate_pct": round(self._total_sla_violations / max(self._total_requests, 1) * 100, 1),
                "shield_overrides": self._shield_overrides,
                "mean_latency_ms": mean_lat,
                "p95_latency_ms": p95_lat,
                "current_rps": round(rps, 2),
                "warm_containers": warm_n,
                "energy_proxy": energy_proxy,
                "workers": workers,
                "recent_requests": list(self._recent_requests)[:15],
                "decisions": list(self._decisions)[:15],
                "timeline": list(self._timeline),
            }

    def prometheus_metrics(self) -> str:
        t = self.get_telemetry()
        lines = [
            "# HELP shoppulse_warm_containers Live warm worker container count",
            "# TYPE shoppulse_warm_containers gauge",
            f'shoppulse_warm_containers{{mode="{t["mode"]}"}} {t["warm_containers"]}',
            "# HELP shoppulse_mean_latency_ms Measured mean HTTP latency in ms",
            "# TYPE shoppulse_mean_latency_ms gauge",
            f'shoppulse_mean_latency_ms{{mode="{t["mode"]}"}} {t["mean_latency_ms"]}',
            "# HELP shoppulse_p95_latency_ms Measured P95 HTTP latency in ms",
            "# TYPE shoppulse_p95_latency_ms gauge",
            f'shoppulse_p95_latency_ms{{mode="{t["mode"]}"}} {t["p95_latency_ms"]}',
            "# HELP shoppulse_requests_total Total e-commerce requests served",
            "# TYPE shoppulse_requests_total counter",
            f'shoppulse_requests_total {t["total_requests"]}',
            "# HELP shoppulse_cold_starts_total Total real cold starts observed",
            "# TYPE shoppulse_cold_starts_total counter",
            f'shoppulse_cold_starts_total {t["total_cold_starts"]}',
            "# HELP shoppulse_sla_violations_total Total requests exceeding 300ms SLA",
            "# TYPE shoppulse_sla_violations_total counter",
            f'shoppulse_sla_violations_total {t["total_sla_violations"]}',
            "# HELP shoppulse_energy_proxy Estimated energy proxy (warm*0.8 + rps*0.15)",
            "# TYPE shoppulse_energy_proxy gauge",
            f'shoppulse_energy_proxy {t["energy_proxy"]}',
        ]
        return "\n".join(lines) + "\n"
