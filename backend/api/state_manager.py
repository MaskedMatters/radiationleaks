"""
backend/api/state_manager.py
Thread-safe simulation state coordinator (`TelemetryBackend`).
Enforces strict per-sector environment isolation for AI models, validation CDF curves,
and training datasets, and integrates concrete wall shielding into sensor telemetry.
"""

from datetime import datetime, timezone
import json
import math
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.ai.network import RadLocNeuralNet
from backend.api.snapshot_service import save_snapshot_bundle_to_disk
from backend.config import (
    AI_CHECKPOINT_PATH,
    DOMAIN_SIZE,
    FACILITIES,
    LOG_FILE_PATH,
    SESSION_STATE_PATH,
    TRAIN_LOG_FILE_PATH,
)
from backend.physics.geometry import compute_ray_wall_shielding, expected_sensor_rates_np
from backend.physics.openmc_engine import HAS_OPENMC, OpenMCEngine


class TelemetryBackend:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.mc_engine = OpenMCEngine(rng_seed=2026)
        self.ai_net = RadLocNeuralNet(
            checkpoint_path=AI_CHECKPOINT_PATH,
            train_log_path=TRAIN_LOG_FILE_PATH,
            seed=42,
        )

        # Always start in STANDBY (not running automatically), while preserving all saved data
        self.running: bool = False
        self.batch_running: bool = False
        self.training_running: bool = False

        self.facility_key: str = "sector-a"
        self.true_source: Dict[str, float] = {"x": 12.40, "y": 8.15}
        self.source_activity_kbq: float = 850.0
        self.ambient_background_cps: float = 18.0
        self.mc_histories: int = 25000

        self.run_counter: int = 0
        self.run_logs: List[Dict[str, Any]] = []
        self.validation_errors_by_sector: Dict[str, List[float]] = {
            k: [] for k in FACILITIES
        }
        self.captures_history: List[Dict[str, Any]] = []
        self.sse_clients: List[queue.Queue[str]] = []

        self._load_persisted_logs_and_session()
        self.latest_frame: Dict[str, Any] = self._build_sector_frame(self.facility_key)

        self.worker_thread = threading.Thread(target=self._continuous_loop, daemon=True)
        self.worker_thread.start()

    @property
    def validation_errors_ai(self) -> List[float]:
        """Returns validation CDF errors strictly for the currently active sector."""
        with self.lock:
            return list(self.validation_errors_by_sector.get(self.facility_key, []))

    def _load_persisted_logs_and_session(self) -> None:
        """Loads existing run logs and session settings from disk without wiping anything."""
        if os.path.exists(LOG_FILE_PATH) and os.path.getsize(LOG_FILE_PATH) > 0:
            try:
                loaded_runs: List[Dict[str, Any]] = []
                with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        line_str = line.strip()
                        if line_str:
                            entry = json.loads(line_str)
                            loaded_runs.append(entry)
                self.run_logs = loaded_runs[-2000:]
                if self.run_logs:
                    last_num = int(self.run_logs[-1].get("run_num", len(self.run_logs)))
                    self.run_counter = max(last_num, len(self.run_logs))
            except (OSError, ValueError):
                pass

        if os.path.exists(SESSION_STATE_PATH) and os.path.getsize(SESSION_STATE_PATH) > 0:
            try:
                with open(SESSION_STATE_PATH, "r", encoding="utf-8") as f:
                    sdata = json.load(f)
                fkey = str(sdata.get("facility_key", "sector-a"))
                if fkey in FACILITIES:
                    self.facility_key = fkey
                ts = sdata.get("true_source")
                if isinstance(ts, dict):
                    self.true_source = {
                        "x": float(ts.get("x", 12.40)),
                        "y": float(ts.get("y", 8.15)),
                    }
                self.source_activity_kbq = float(sdata.get("source_activity_kbq", 850.0))
                self.ambient_background_cps = float(sdata.get("ambient_background_cps", 18.0))

                val_by_sec = sdata.get("validation_errors_by_sector")
                if isinstance(val_by_sec, dict):
                    for k, vlist in val_by_sec.items():
                        if k in FACILITIES and isinstance(vlist, list):
                            self.validation_errors_by_sector[k] = [float(x) for x in vlist]
                else:
                    # Migrate legacy single-sector list if present
                    val_errs = sdata.get("validation_errors_ai", [])
                    if isinstance(val_errs, list):
                        self.validation_errors_by_sector[self.facility_key] = [
                            float(x) for x in val_errs
                        ]

                caps = sdata.get("captures_history", [])
                if isinstance(caps, list):
                    self.captures_history = caps[-50:]
            except (OSError, ValueError):
                pass

    def _save_session_state(self) -> None:
        payload = {
            "facility_key": self.facility_key,
            "true_source": self.true_source,
            "source_activity_kbq": self.source_activity_kbq,
            "ambient_background_cps": self.ambient_background_cps,
            "validation_errors_by_sector": self.validation_errors_by_sector,
            "captures_history": self.captures_history[-50:],
        }
        try:
            tmp_path = SESSION_STATE_PATH + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp_path, SESSION_STATE_PATH)
        except OSError:
            pass

    def _utc_now_str(self) -> str:
        now_dt = datetime.now(timezone.utc)
        base_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        tenths = now_dt.microsecond // 100000
        return f"{base_str}.{tenths} UTC"

    def _get_zone_label(self, facility_key: str, x: float, y: float) -> str:
        zones: List[Dict[str, Any]] = FACILITIES[facility_key]["zones"]
        for z in zones:
            zx = float(z["x"])
            zy = float(z["y"])
            zw = float(z["w"])
            zh = float(z["h"])
            if zx <= x <= zx + zw and zy <= y <= zy + zh:
                label_str = str(z["label"])
                return label_str.split(" ")[0]
        return "SECTOR-CORE"

    def _build_sector_frame(self, facility_key: str) -> Dict[str, Any]:
        """
        Builds a frame strictly for `facility_key` using the most recent run logged
        in `facility_key`, or an idle standby frame if no runs exist for `facility_key`.
        """
        sector_logs = [r for r in self.run_logs if r.get("facility") == facility_key]
        if sector_logs:
            last = sector_logs[-1]
            facility = FACILITIES[facility_key]
            sensors: List[Dict[str, Any]] = facility["sensors"]
            walls: List[Dict[str, float]] = facility["walls"]
            raw_counts: List[int] = list(last.get("raw_counts", [0] * 16))
            bkg_cps = float(last.get("background_cps", self.ambient_background_cps))
            tx = float(last.get("true_x", self.true_source["x"]))
            ty = float(last.get("true_y", self.true_source["y"]))
            act_kbq = float(last.get("true_activity_kbq", self.source_activity_kbq))

            # Evaluate using strictly this sector's AI model
            ai_sol = self.ai_net.predict(facility_key, raw_counts, bkg_cps)
            ai_err = round(math.hypot(float(ai_sol["x"]) - tx, float(ai_sol["y"]) - ty), 3)
            ai_sol["error_meters"] = ai_err

            sx_arr = np.array([float(s["x"]) for s in sensors], dtype=np.float64)
            sy_arr = np.array([float(s["y"]) for s in sensors], dtype=np.float64)
            model_fits = expected_sensor_rates_np(
                sx_arr,
                sy_arr,
                float(ai_sol["x"]),
                float(ai_sol["y"]),
                float(ai_sol["activity"]),
                bkg_cps,
                walls=walls,
            )
            readings: List[Dict[str, Any]] = []
            for idx, s in enumerate(sensors):
                raw = int(raw_counts[idx]) if idx < len(raw_counts) else 0
                net = max(0.0, float(raw) - bkg_cps)
                m_fit = round(float(model_fits[idx]), 1)
                dist_est = round(
                    math.hypot(float(s["x"]) - float(ai_sol["x"]), float(s["y"]) - float(ai_sol["y"])),
                    2,
                )
                n_walls, w_trans = compute_ray_wall_shielding(
                    float(s["x"]), float(s["y"]), tx, ty, walls
                )
                status = "NOMINAL"
                if raw > bkg_cps + 45.0 and raw > bkg_cps * 2.2:
                    status = "ALARM"
                elif raw > bkg_cps + 15.0 and raw > bkg_cps * 1.45:
                    status = "ELEVATED"
                readings.append({
                    "id": str(s["id"]),
                    "x": float(s["x"]),
                    "y": float(s["y"]),
                    "raw": raw,
                    "net": round(net, 1),
                    "modelFit": m_fit,
                    "distEst": dist_est,
                    "wallCrossings": n_walls,
                    "wallTransmission": round(w_trans, 3),
                    "status": status,
                })

            return {
                "has_run": True,
                "run_id": str(last.get("run_id", f"RUN-{self.run_counter:04d}")),
                "frame_count": int(last.get("run_num", self.run_counter)),
                "timestamp_utc": str(last.get("timestamp_utc", self._utc_now_str())),
                "running": self.running,
                "batch_running": self.batch_running,
                "training_running": self.training_running,
                "mode": str(last.get("mode", "RESTORED")),
                "facility_key": facility_key,
                "true_source": {"x": tx, "y": ty},
                "true_zone": self._get_zone_label(facility_key, tx, ty),
                "source_activity_kbq": act_kbq,
                "ambient_background_cps": bkg_cps,
                "mc_histories": int(last.get("histories", self.mc_histories)),
                "mc_engine": str(last.get("mc_engine", "openmc-photon-mc-kernel")),
                "mc_elapsed_ms": float(last.get("mc_elapsed_ms", 0.0)),
                "solution": ai_sol,
                "ai_solution": ai_sol,
                "sensor_readings": readings,
                "log_entry": last,
                "ai_model_status": self.get_ai_status(facility_key),
                "pathway_telemetry": self.ai_net.get_pathway_telemetry(
                    facility_key, raw_counts, bkg_cps
                ),
            }

        return self._build_idle_standby_frame(facility_key)

    def _build_idle_standby_frame(self, facility_key: Optional[str] = None) -> Dict[str, Any]:
        fkey = facility_key or self.facility_key
        facility = FACILITIES[fkey]
        sensors: List[Dict[str, Any]] = facility["sensors"]
        walls: List[Dict[str, float]] = facility["walls"]
        tx = self.true_source["x"]
        ty = self.true_source["y"]

        idle_readings: List[Dict[str, Any]] = []
        for s in sensors:
            n_walls, w_trans = compute_ray_wall_shielding(
                float(s["x"]), float(s["y"]), tx, ty, walls
            )
            idle_readings.append({
                "id": str(s["id"]),
                "x": float(s["x"]),
                "y": float(s["y"]),
                "raw": 0,
                "net": 0.0,
                "modelFit": 0.0,
                "distEst": 0.0,
                "wallCrossings": n_walls,
                "wallTransmission": round(w_trans, 3),
                "status": "STANDBY",
            })

        diag = self.get_ai_status(fkey)
        empty_sol: Dict[str, Any] = {
            "solver": "STANDBY",
            "facility_key": fkey,
            "is_trained": bool(diag.get("is_trained", False)),
            "trained_epochs": int(diag.get("trained_epochs", 0)),
            "x": tx,
            "y": ty,
            "activity": 0.0,
            "error_meters": 0.0,
            "ci95_meters": 0.0,
            "latency_ms": 0.0,
            "iterations": 0,
            "converged": False,
        }
        return {
            "has_run": False,
            "run_id": "STANDBY-0000",
            "frame_count": 0,
            "timestamp_utc": self._utc_now_str(),
            "running": False,
            "batch_running": False,
            "training_running": False,
            "mode": "STANDBY",
            "facility_key": fkey,
            "true_source": {"x": tx, "y": ty},
            "true_zone": self._get_zone_label(fkey, tx, ty),
            "source_activity_kbq": self.source_activity_kbq,
            "ambient_background_cps": self.ambient_background_cps,
            "mc_histories": self.mc_histories,
            "mc_engine": "openmc-native" if HAS_OPENMC else "openmc-photon-mc-kernel",
            "mc_elapsed_ms": 0.0,
            "solution": dict(empty_sol),
            "ai_solution": dict(empty_sol),
            "sensor_readings": idle_readings,
            "log_entry": None,
            "ai_model_status": diag,
            "pathway_telemetry": self.ai_net.get_pathway_telemetry(
                fkey, None, self.ambient_background_cps
            ),
        }

    def switch_facility(self, new_fkey: str) -> None:
        """
        Switches active facility sector and loads that sector's isolated AI model,
        training dataset, validation errors, and latest frame.
        """
        if new_fkey not in FACILITIES:
            return
        with self.lock:
            self.facility_key = new_fkey
            self.latest_frame = self._build_sector_frame(new_fkey)
            self._save_session_state()
            full_state = self.get_full_state()

        self.broadcast_event("sector_changed", full_state)

    def scrub_ai_only(self) -> None:
        """Scrubs the AI model and training data for the currently active sector."""
        with self.lock:
            fkey = self.facility_key
            self.ai_net.scrub_sector(fkey)
            self.validation_errors_by_sector[fkey] = []
            self._save_session_state()
            self.latest_frame = self._build_sector_frame(fkey)

        self.broadcast_event(
            "ai_scrubbed",
            {
                "facility_key": fkey,
                "ai_model_status": self.get_ai_status(fkey),
                "training_samples": [],
                "validation_errors_ai": [],
                "pathway_telemetry": self.get_pathway_telemetry(),
            },
        )

    def scrub_all_data_and_ai(self) -> None:
        with self.lock:
            self.running = False
            self.ai_net.scrub_all()
            self.run_counter = 0
            self.run_logs.clear()
            for k in FACILITIES:
                self.validation_errors_by_sector[k] = []
            try:
                with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
                    f.truncate(0)
            except OSError:
                pass
            self._save_session_state()
            self.latest_frame = self._build_idle_standby_frame(self.facility_key)

        self.broadcast_event("all_scrubbed", self.get_full_state())

    def broadcast_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        msg = f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
        with self.lock:
            dead_clients: List[queue.Queue[str]] = []
            for q in self.sse_clients:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    dead_clients.append(q)
            for dq in dead_clients:
                if dq in self.sse_clients:
                    self.sse_clients.remove(dq)

    def execute_single_run(
        self,
        mode: str = "SINGLE",
        override_pos: Optional[Tuple[float, float]] = None,
        broadcast: bool = True,
    ) -> Dict[str, Any]:
        with self.lock:
            facility_key = self.facility_key
            facility = FACILITIES[facility_key]
            sensors: List[Dict[str, Any]] = facility["sensors"]
            walls: List[Dict[str, float]] = facility["walls"]
            if override_pos is not None:
                tx = float(override_pos[0])
                ty = float(override_pos[1])
            else:
                tx = float(self.true_source["x"])
                ty = float(self.true_source["y"])
            act_kbq = self.source_activity_kbq
            bkg_cps = self.ambient_background_cps
            histories = self.mc_histories

        # 1. Run OpenMC / Monte Carlo photon transport simulation (with concrete wall shielding)
        mc_res = self.mc_engine.run_monte_carlo_frame(
            facility_key=facility_key,
            source_x=tx,
            source_y=ty,
            activity_kbq=act_kbq,
            background_cps=bkg_cps,
            n_histories=histories,
        )
        raw_counts: List[int] = mc_res["raw_counts"]
        wall_crossings: List[int] = mc_res["wall_crossings"]
        wall_transmissions: List[float] = mc_res["wall_transmissions"]

        # 2. Run Sector-Specific AI Neural Network Localization
        ai_sol = self.ai_net.predict(facility_key, raw_counts, bkg_cps)
        ai_err = round(math.hypot(float(ai_sol["x"]) - tx, float(ai_sol["y"]) - ty), 3)
        ai_sol["error_meters"] = ai_err

        # 3. Compute expected sensor flux at the AI's predicted location (including concrete walls)
        sx_arr = np.array([float(s["x"]) for s in sensors], dtype=np.float64)
        sy_arr = np.array([float(s["y"]) for s in sensors], dtype=np.float64)
        model_fits = expected_sensor_rates_np(
            sx_arr,
            sy_arr,
            float(ai_sol["x"]),
            float(ai_sol["y"]),
            float(ai_sol["activity"]),
            bkg_cps,
            walls=walls,
        )

        sensor_readings: List[Dict[str, Any]] = []
        for idx, s in enumerate(sensors):
            raw = int(raw_counts[idx])
            net = max(0.0, float(raw) - bkg_cps)
            m_fit = round(float(model_fits[idx]), 1)
            dist_est = round(
                math.hypot(float(s["x"]) - float(ai_sol["x"]), float(s["y"]) - float(ai_sol["y"])),
                2,
            )
            status = "NOMINAL"
            if raw > bkg_cps + 45.0 and raw > bkg_cps * 2.2:
                status = "ALARM"
            elif raw > bkg_cps + 15.0 and raw > bkg_cps * 1.45:
                status = "ELEVATED"

            sensor_readings.append({
                "id": str(s["id"]),
                "x": float(s["x"]),
                "y": float(s["y"]),
                "raw": raw,
                "net": round(net, 1),
                "modelFit": m_fit,
                "distEst": dist_est,
                "wallCrossings": int(wall_crossings[idx]),
                "wallTransmission": round(float(wall_transmissions[idx]), 3),
                "status": status,
            })

        with self.lock:
            self.run_counter += 1
            run_id = f"RUN-{self.run_counter:04d}"
            ts_utc = self._utc_now_str()

            log_entry: Dict[str, Any] = {
                "run_id": run_id,
                "run_num": self.run_counter,
                "timestamp_utc": ts_utc,
                "mode": mode,
                "facility": facility_key,
                "histories": histories,
                "mc_engine": mc_res["engine"],
                "mc_elapsed_ms": mc_res["mc_elapsed_ms"],
                "true_x": round(tx, 2),
                "true_y": round(ty, 2),
                "true_activity_kbq": round(act_kbq, 1),
                "background_cps": round(bkg_cps, 1),
                "ai_is_trained": bool(ai_sol["is_trained"]),
                "ai_epochs": int(ai_sol["trained_epochs"]),
                "ai_x": float(ai_sol["x"]),
                "ai_y": float(ai_sol["y"]),
                "ai_activity_kbq": float(ai_sol["activity"]),
                "ai_error_m": ai_err,
                "ai_latency_ms": float(ai_sol["latency_ms"]),
                "raw_counts": raw_counts,
                "wall_crossings": wall_crossings,
            }

            self.run_logs.append(log_entry)
            if len(self.run_logs) > 2000:
                self.run_logs.pop(0)

            try:
                with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except OSError:
                pass

            self._save_session_state()

            frame: Dict[str, Any] = {
                "has_run": True,
                "run_id": run_id,
                "frame_count": self.run_counter,
                "timestamp_utc": ts_utc,
                "running": self.running,
                "batch_running": self.batch_running,
                "training_running": self.training_running,
                "mode": mode,
                "facility_key": facility_key,
                "true_source": {"x": round(tx, 2), "y": round(ty, 2)},
                "true_zone": self._get_zone_label(facility_key, tx, ty),
                "source_activity_kbq": act_kbq,
                "ambient_background_cps": bkg_cps,
                "mc_histories": histories,
                "mc_engine": mc_res["engine"],
                "mc_elapsed_ms": mc_res["mc_elapsed_ms"],
                "solution": ai_sol,
                "ai_solution": ai_sol,
                "sensor_readings": sensor_readings,
                "log_entry": log_entry,
                "ai_model_status": self.get_ai_status(facility_key),
                "pathway_telemetry": self.ai_net.get_pathway_telemetry(
                    facility_key, raw_counts, bkg_cps
                ),
            }
            self.latest_frame = frame

        if broadcast:
            self.broadcast_event("frame", frame)

        return frame

    def run_validation_batch_streamed(self, n_runs: int = 100) -> None:
        with self.lock:
            if self.batch_running:
                return
            self.batch_running = True
            target_fkey = self.facility_key

        def _worker() -> None:
            errs_ai: List[float] = []
            try:
                for i in range(1, n_runs + 1):
                    tx = round(float(self.mc_engine.rng.uniform(1.6, DOMAIN_SIZE - 1.6)), 2)
                    ty = round(float(self.mc_engine.rng.uniform(1.6, DOMAIN_SIZE - 1.6)), 2)
                    with self.lock:
                        self.true_source = {"x": tx, "y": ty}

                    frame = self.execute_single_run(
                        mode=f"BATCH ({i}/{n_runs})",
                        override_pos=(tx, ty),
                        broadcast=True,
                    )
                    errs_ai.append(float(frame["ai_solution"]["error_meters"]))

                    sorted_ai = sorted(errs_ai)
                    with self.lock:
                        self.validation_errors_by_sector[target_fkey] = sorted_ai

                    if i % 4 == 0 or i == n_runs:
                        self.broadcast_event(
                            "batch_progress",
                            {
                                "facility_key": target_fkey,
                                "completed": i,
                                "total": n_runs,
                                "validation_errors_ai": sorted_ai,
                            },
                        )
                    time.sleep(0.025)
            finally:
                with self.lock:
                    self.batch_running = False
                    self._save_session_state()
                    final_errs = list(self.validation_errors_by_sector.get(target_fkey, []))
                self.broadcast_event(
                    "batch_complete",
                    {
                        "facility_key": target_fkey,
                        "validation_errors_ai": final_errs,
                        "batch_running": False,
                    },
                )

        threading.Thread(target=_worker, daemon=True).start()

    def run_ai_training_streamed(
        self, train_source: str = "logged_runs", n_samples: int = 64, epochs: int = 50
    ) -> Dict[str, Any]:
        with self.lock:
            if self.training_running:
                return {"ok": False, "error": "Training already in progress"}
            fkey = self.facility_key
            logged_copy = list(self.run_logs)
            if train_source == "logged_runs":
                matching = [r for r in logged_copy if r.get("facility") == fkey]
                if not matching:
                    return {
                        "ok": False,
                        "error": f"No logged runs exist for {fkey.upper()} yet! Run simulations in {fkey.upper()} first or click 'Generate & Train (64 MC Samples)'.",
                    }
            self.training_running = True

        def _on_progress(prog: Dict[str, Any]) -> None:
            payload = dict(prog)
            payload["facility_key"] = fkey
            payload["ai_model_status"] = self.get_ai_status(fkey)
            self.broadcast_event("ai_training_progress", payload)

        def _worker() -> None:
            try:
                if train_source == "logged_runs":
                    self.ai_net.train_on_logged_runs(
                        facility_key=fkey,
                        logged_runs=logged_copy,
                        epochs=epochs,
                        progress_cb=_on_progress,
                    )
                else:
                    self.ai_net.train_on_new_openmc_samples(
                        facility_key=fkey,
                        mc_engine=self.mc_engine,
                        n_samples=n_samples,
                        epochs=epochs,
                        progress_cb=_on_progress,
                    )
            finally:
                with self.lock:
                    self.training_running = False
                    self.latest_frame = self._build_sector_frame(fkey)
                    self._save_session_state()
                    sector_train = self.ai_net.get_sector_training_samples(fkey, limit=150)
                self.broadcast_event(
                    "ai_training_complete",
                    {
                        "facility_key": fkey,
                        "ai_model_status": self.get_ai_status(fkey),
                        "training_samples": sector_train,
                        "pathway_telemetry": self.get_pathway_telemetry(),
                        "latest_frame": self.latest_frame,
                    },
                )

        threading.Thread(target=_worker, daemon=True).start()
        return {"ok": True, "started": True, "facility_key": fkey}

    def save_snapshot_bundle(self, body: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            cap_record = save_snapshot_bundle_to_disk(
                body=body,
                run_counter=self.run_counter,
                captured_at_utc=self._utc_now_str(),
                facility_key=self.facility_key,
                true_source=self.true_source,
                source_activity_kbq=self.source_activity_kbq,
                ambient_background_cps=self.ambient_background_cps,
                ai_model_status=self.get_ai_status(),
                latest_frame=self.latest_frame,
                validation_errors_ai=self.validation_errors_ai,
            )
            self.captures_history.append(cap_record)
            self._save_session_state()

        self.broadcast_event("snapshot_saved", cap_record)
        return {"ok": True, "capture": cap_record}

    def _continuous_loop(self) -> None:
        while True:
            time.sleep(1.0)
            with self.lock:
                is_running = self.running and not self.batch_running
            if is_running:
                self.execute_single_run(mode="STREAM", broadcast=True)

    def get_ai_status(self, facility_key: Optional[str] = None) -> Dict[str, Any]:
        with self.lock:
            fkey = facility_key or self.facility_key
            diag = self.ai_net.get_model_diagnostics(fkey)
            diag["training_running"] = self.training_running
            return diag

    def get_pathway_telemetry(self) -> Dict[str, Any]:
        with self.lock:
            counts: Optional[List[int]] = None
            sector_logs = [r for r in self.run_logs if r.get("facility") == self.facility_key]
            if sector_logs:
                last_counts = sector_logs[-1].get("raw_counts")
                if isinstance(last_counts, list):
                    counts = [int(v) for v in last_counts]
            return self.ai_net.get_pathway_telemetry(
                facility_key=self.facility_key,
                counts=counts,
                background_cps=self.ambient_background_cps,
            )

    def get_full_state(self) -> Dict[str, Any]:
        with self.lock:
            fkey = self.facility_key
            sector_logs = [r for r in self.run_logs if r.get("facility") == fkey]
            rev_logs = list(reversed(sector_logs))
            recent_logs = rev_logs[:150]
            recent_train = self.ai_net.get_sector_training_samples(fkey, limit=150)

            return {
                "running": self.running,
                "batch_running": self.batch_running,
                "training_running": self.training_running,
                "facility_key": fkey,
                "facilities": FACILITIES,
                "true_source": self.true_source,
                "source_activity_kbq": self.source_activity_kbq,
                "ambient_background_cps": self.ambient_background_cps,
                "mc_histories": self.mc_histories,
                "has_openmc_pkg": HAS_OPENMC,
                "ai_model_status": self.get_ai_status(fkey),
                "pathway_telemetry": self.get_pathway_telemetry(),
                "latest_frame": self.latest_frame,
                "validation_errors_ai": list(self.validation_errors_by_sector.get(fkey, [])),
                "recent_logs": recent_logs,
                "total_logged_runs": len(sector_logs),
                "training_samples": recent_train,
                "total_training_samples": len(recent_train),
                "captures_history": list(reversed(self.captures_history[-20:])),
            }


BACKEND = TelemetryBackend()
