#!/usr/bin/env python3
"""
server.py
Real-Time Backend Control Server for RAD-MON-04 Radiation Leak Localization System.

Features:
- Persistent State: AI neural network weights (`data/ai_checkpoint.json`), training
  samples (`data/ai_training_samples.jsonl`), simulation logs (`data/run_logs.jsonl`),
  and session state (`data/session_state.json`) are automatically loaded on startup
  and saved after every run/training update so nothing is lost on server restart.
- Pure AI Localization: All triangulation / L-BFGS code removed; only the AI neural
  network performs localization and activity estimation.
- Snapshot Capture (`POST /api/capture-snapshot`): Saves PNG screenshots of the 2D
  facility map, CDF chart, Flux chart, a combined composite image, and JSON telemetry
  into a timestamped folder under `captures/`.
"""

import base64
import csv
import io
import json
import math
import os
import queue
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np

from ai_model import RadLocNeuralNet, expected_sensor_rates_np
from openmc_engine import (
    DOMAIN_SIZE,
    FACILITIES,
    HAS_OPENMC,
    OpenMCEngine,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CAPTURES_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(DATA_DIR, "run_logs.jsonl")
TRAIN_LOG_FILE_PATH = os.path.join(DATA_DIR, "ai_training_samples.jsonl")
AI_CHECKPOINT_PATH = os.path.join(DATA_DIR, "ai_checkpoint.json")
SESSION_STATE_PATH = os.path.join(DATA_DIR, "session_state.json")


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
        self.validation_errors_ai: List[float] = []
        self.captures_history: List[Dict[str, Any]] = []
        self.sse_clients: List[queue.Queue[str]] = []

        self._load_persisted_logs_and_session()
        self.latest_frame: Dict[str, Any] = self._build_initial_frame()

        self.worker_thread = threading.Thread(target=self._continuous_loop, daemon=True)
        self.worker_thread.start()

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
                val_errs = sdata.get("validation_errors_ai", [])
                if isinstance(val_errs, list):
                    self.validation_errors_ai = [float(x) for x in val_errs]
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
            "validation_errors_ai": self.validation_errors_ai,
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

    def _build_initial_frame(self) -> Dict[str, Any]:
        if self.run_logs:
            last = self.run_logs[-1]
            fkey = str(last.get("facility", self.facility_key))
            if fkey not in FACILITIES:
                fkey = self.facility_key
            sensors: List[Dict[str, Any]] = FACILITIES[fkey]["sensors"]
            raw_counts: List[int] = list(last.get("raw_counts", [0] * 16))
            bkg_cps = float(last.get("background_cps", self.ambient_background_cps))
            tx = float(last.get("true_x", self.true_source["x"]))
            ty = float(last.get("true_y", self.true_source["y"]))
            act_kbq = float(last.get("true_activity_kbq", self.source_activity_kbq))

            ai_sol = {
                "solver": "AI-MLP" if last.get("ai_is_trained") else "AI-MLP (UNTRAINED)",
                "is_trained": bool(last.get("ai_is_trained", False)),
                "trained_epochs": int(last.get("ai_epochs", 0)),
                "x": float(last.get("ai_x", tx)),
                "y": float(last.get("ai_y", ty)),
                "activity": float(last.get("ai_activity_kbq", act_kbq)),
                "error_meters": float(last.get("ai_error_m", 0.0)),
                "ci95_meters": 0.35 if last.get("ai_is_trained") else 2.50,
                "latency_ms": float(last.get("ai_latency_ms", 0.1)),
                "iterations": 1,
                "converged": bool(last.get("ai_is_trained", False)),
            }

            sx_arr = np.array([float(s["x"]) for s in sensors], dtype=np.float64)
            sy_arr = np.array([float(s["y"]) for s in sensors], dtype=np.float64)
            model_fits = expected_sensor_rates_np(
                sx_arr, sy_arr, float(ai_sol["x"]), float(ai_sol["y"]), float(ai_sol["activity"]), bkg_cps
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
                    "status": status,
                })

            return {
                "has_run": True,
                "run_id": str(last.get("run_id", f"RUN-{self.run_counter:04d}")),
                "frame_count": self.run_counter,
                "timestamp_utc": str(last.get("timestamp_utc", self._utc_now_str())),
                "running": False,
                "batch_running": False,
                "training_running": False,
                "mode": str(last.get("mode", "RESTORED")),
                "facility_key": fkey,
                "true_source": {"x": tx, "y": ty},
                "true_zone": self._get_zone_label(fkey, tx, ty),
                "source_activity_kbq": act_kbq,
                "ambient_background_cps": bkg_cps,
                "mc_histories": int(last.get("histories", self.mc_histories)),
                "mc_engine": str(last.get("mc_engine", "openmc-photon-mc-kernel")),
                "mc_elapsed_ms": float(last.get("mc_elapsed_ms", 0.0)),
                "solution": ai_sol,
                "ai_solution": ai_sol,
                "sensor_readings": readings,
                "log_entry": last,
                "ai_model_status": self.get_ai_status(),
            }

        return self._build_idle_standby_frame()

    def _build_idle_standby_frame(self) -> Dict[str, Any]:
        facility = FACILITIES[self.facility_key]
        sensors: List[Dict[str, Any]] = facility["sensors"]
        idle_readings: List[Dict[str, Any]] = [
            {
                "id": str(s["id"]),
                "x": float(s["x"]),
                "y": float(s["y"]),
                "raw": 0,
                "net": 0.0,
                "modelFit": 0.0,
                "distEst": 0.0,
                "status": "STANDBY",
            }
            for s in sensors
        ]
        empty_sol: Dict[str, Any] = {
            "solver": "STANDBY",
            "is_trained": False,
            "trained_epochs": 0,
            "x": self.true_source["x"],
            "y": self.true_source["y"],
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
            "facility_key": self.facility_key,
            "true_source": {"x": self.true_source["x"], "y": self.true_source["y"]},
            "true_zone": self._get_zone_label(self.facility_key, self.true_source["x"], self.true_source["y"]),
            "source_activity_kbq": self.source_activity_kbq,
            "ambient_background_cps": self.ambient_background_cps,
            "mc_histories": self.mc_histories,
            "mc_engine": "openmc-native" if HAS_OPENMC else "openmc-photon-mc-kernel",
            "mc_elapsed_ms": 0.0,
            "solution": dict(empty_sol),
            "ai_solution": dict(empty_sol),
            "sensor_readings": idle_readings,
            "log_entry": None,
            "ai_model_status": self.get_ai_status(),
        }

    def scrub_ai_only(self) -> None:
        with self.lock:
            self.ai_net.scrub_all()
            self.validation_errors_ai.clear()
            self._save_session_state()
            self.latest_frame["ai_model_status"] = self.get_ai_status()

        self.broadcast_event(
            "ai_scrubbed",
            {
                "ai_model_status": self.get_ai_status(),
                "training_samples": [],
                "validation_errors_ai": [],
            },
        )

    def scrub_all_data_and_ai(self) -> None:
        with self.lock:
            self.running = False
            self.ai_net.scrub_all()
            self.run_counter = 0
            self.run_logs.clear()
            self.validation_errors_ai.clear()
            try:
                with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
                    f.truncate(0)
            except OSError:
                pass
            self._save_session_state()
            self.latest_frame = self._build_idle_standby_frame()

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
            if override_pos is not None:
                tx = float(override_pos[0])
                ty = float(override_pos[1])
            else:
                tx = float(self.true_source["x"])
                ty = float(self.true_source["y"])
            act_kbq = self.source_activity_kbq
            bkg_cps = self.ambient_background_cps
            histories = self.mc_histories

        # 1. Run OpenMC / Monte Carlo photon transport simulation
        mc_res = self.mc_engine.run_monte_carlo_frame(
            facility_key=facility_key,
            source_x=tx,
            source_y=ty,
            activity_kbq=act_kbq,
            background_cps=bkg_cps,
            n_histories=histories,
        )
        raw_counts: List[int] = mc_res["raw_counts"]

        # 2. Run Pure AI Neural Network Localization
        ai_sol = self.ai_net.predict(facility_key, raw_counts, bkg_cps)
        ai_err = round(math.hypot(float(ai_sol["x"]) - tx, float(ai_sol["y"]) - ty), 3)
        ai_sol["error_meters"] = ai_err

        # 3. Compute expected sensor flux at the AI's predicted location
        sx_arr = np.array([float(s["x"]) for s in sensors], dtype=np.float64)
        sy_arr = np.array([float(s["y"]) for s in sensors], dtype=np.float64)
        model_fits = expected_sensor_rates_np(
            sx_arr,
            sy_arr,
            float(ai_sol["x"]),
            float(ai_sol["y"]),
            float(ai_sol["activity"]),
            bkg_cps,
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
                "ai_model_status": self.get_ai_status(),
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
                        self.validation_errors_ai = sorted_ai

                    if i % 4 == 0 or i == n_runs:
                        self.broadcast_event(
                            "batch_progress",
                            {
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
                self.broadcast_event(
                    "batch_complete",
                    {
                        "validation_errors_ai": self.validation_errors_ai,
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
                        "error": f"No logged runs exist for {fkey.upper()} yet! Run simulations first or click 'Generate & Train (64 MC Samples)'.",
                    }
            self.training_running = True

        def _on_progress(prog: Dict[str, Any]) -> None:
            payload = dict(prog)
            payload["facility_key"] = fkey
            payload["ai_model_status"] = self.get_ai_status()
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
                    self._save_session_state()
                    rev_samples = list(reversed(self.ai_net.training_samples_log))
                    recent_train = [
                        rev_samples[idx]
                        for idx in range(min(150, len(rev_samples)))
                    ]
                self.broadcast_event(
                    "ai_training_complete",
                    {
                        "ai_model_status": self.get_ai_status(),
                        "training_samples": recent_train,
                    },
                )

        threading.Thread(target=_worker, daemon=True).start()
        return {"ok": True, "started": True}

    def save_snapshot_bundle(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Saves PNG screenshots of the 2D Facility Map, CDF Chart, Sensor Flux Chart,
        and a combined composite image + JSON telemetry metadata into a timestamped
        folder under `captures/`.
        """
        now_dt = datetime.now(timezone.utc)
        ts_slug = now_dt.strftime("%Y-%m-%d_%H-%M-%S_UTC")
        with self.lock:
            run_tag = f"RUN-{self.run_counter:04d}"
            folder_name = f"snapshot_{ts_slug}_{run_tag}"
            folder_path = os.path.join(CAPTURES_DIR, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            image_keys = [
                ("map_png", "facility_map.png"),
                ("cdf_png", "cdf_error_chart.png"),
                ("flux_png", "sensor_flux_chart.png"),
                ("composite_png", "dashboard_composite.png"),
            ]
            saved_files: List[str] = []

            for key, filename in image_keys:
                data_url = str(body.get(key, ""))
                if data_url.startswith("data:image/png;base64,"):
                    b64_part = data_url.split(",", 1)[1]
                    raw_bytes = base64.b64decode(b64_part)
                    file_dest = os.path.join(folder_path, filename)
                    with open(file_dest, "wb") as img_f:
                        img_f.write(raw_bytes)
                    saved_files.append(filename)

            metadata = {
                "captured_at_utc": self._utc_now_str(),
                "folder_name": folder_name,
                "folder_path": folder_path,
                "run_id": run_tag,
                "facility_key": self.facility_key,
                "true_source": self.true_source,
                "source_activity_kbq": self.source_activity_kbq,
                "ambient_background_cps": self.ambient_background_cps,
                "ai_model_status": self.get_ai_status(),
                "latest_frame": self.latest_frame,
                "validation_errors_ai": self.validation_errors_ai,
                "saved_files": saved_files + ["telemetry_metadata.json"],
            }
            meta_dest = os.path.join(folder_path, "telemetry_metadata.json")
            with open(meta_dest, "w", encoding="utf-8") as mf:
                json.dump(metadata, mf, indent=2)

            cap_record = {
                "folder_name": folder_name,
                "folder_path": f"captures/{folder_name}",
                "captured_at_utc": metadata["captured_at_utc"],
                "run_id": run_tag,
                "files": metadata["saved_files"],
            }
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

    def get_ai_status(self) -> Dict[str, Any]:
        with self.lock:
            diag = self.ai_net.get_model_diagnostics(self.facility_key)
            diag["training_running"] = self.training_running
            return diag

    def get_full_state(self) -> Dict[str, Any]:
        with self.lock:
            rev_logs = list(reversed(self.run_logs))
            recent_logs = [rev_logs[i] for i in range(min(150, len(rev_logs)))]
            rev_train = list(reversed(self.ai_net.training_samples_log))
            recent_train = [rev_train[i] for i in range(min(150, len(rev_train)))]

            return {
                "running": self.running,
                "batch_running": self.batch_running,
                "training_running": self.training_running,
                "facility_key": self.facility_key,
                "facilities": FACILITIES,
                "true_source": self.true_source,
                "source_activity_kbq": self.source_activity_kbq,
                "ambient_background_cps": self.ambient_background_cps,
                "mc_histories": self.mc_histories,
                "has_openmc_pkg": HAS_OPENMC,
                "ai_model_status": self.get_ai_status(),
                "latest_frame": self.latest_frame,
                "validation_errors_ai": self.validation_errors_ai,
                "recent_logs": recent_logs,
                "total_logged_runs": len(self.run_logs),
                "training_samples": recent_train,
                "total_training_samples": len(self.ai_net.training_samples_log),
                "captures_history": list(reversed(self.captures_history[-20:])),
            }


BACKEND = TelemetryBackend()


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        _ = (format, args)

    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        parsed = json.loads(raw.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/state":
            self._send_json(BACKEND.get_full_state())
            return

        if path == "/api/stream":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            client_q: queue.Queue[str] = queue.Queue(maxsize=150)
            with BACKEND.lock:
                BACKEND.sse_clients.append(client_q)

            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while True:
                    try:
                        msg = client_q.get(timeout=10.0)
                        self.wfile.write(msg.encode("utf-8"))
                        self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with BACKEND.lock:
                    if client_q in BACKEND.sse_clients:
                        BACKEND.sse_clients.remove(client_q)
            return

        if path == "/api/export/logs.csv":
            self._handle_export_csv()
            return

        if path == "/api/export/logs.json":
            self._handle_export_json()
            return

        if path == "/api/export/training_data.csv":
            self._handle_export_training_csv()
            return

        if path == "/api/export/training_data.json":
            self._handle_export_training_json()
            return

        if path == "/api/export/model.json":
            self._handle_export_model()
            return

        if path == "/api/export/openmc.py":
            self._handle_export_openmc_script()
            return

        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._read_json_body()

        if path == "/api/control":
            run_on_change = bool(body.get("trigger_run", False))
            mode_label = str(body.get("mode", "PARAM-UPDATE"))

            with BACKEND.lock:
                if "running" in body:
                    BACKEND.running = bool(body["running"])
                if "facility_key" in body and str(body["facility_key"]) in FACILITIES:
                    new_fkey = str(body["facility_key"])
                    if new_fkey != BACKEND.facility_key:
                        BACKEND.facility_key = new_fkey
                        if len(BACKEND.run_logs) == 0:
                            BACKEND.latest_frame = BACKEND._build_idle_standby_frame()
                if "source_activity_kbq" in body:
                    BACKEND.source_activity_kbq = max(5.0, min(15000.0, float(body["source_activity_kbq"])))
                if "ambient_background_cps" in body:
                    BACKEND.ambient_background_cps = max(1.0, min(500.0, float(body["ambient_background_cps"])))
                if "mc_histories" in body:
                    BACKEND.mc_histories = max(1000, min(100000, int(body["mc_histories"])))
                if "true_source" in body and isinstance(body["true_source"], dict):
                    ts = body["true_source"]
                    BACKEND.true_source = {
                        "x": max(0.2, min(DOMAIN_SIZE - 0.2, float(ts.get("x", 10.0)))),
                        "y": max(0.2, min(DOMAIN_SIZE - 0.2, float(ts.get("y", 10.0)))),
                    }
                BACKEND._save_session_state()

            if run_on_change:
                BACKEND.execute_single_run(mode=mode_label, broadcast=True)
            else:
                BACKEND.broadcast_event(
                    "status",
                    {
                        "running": BACKEND.running,
                        "facility_key": BACKEND.facility_key,
                        "true_source": BACKEND.true_source,
                        "source_activity_kbq": BACKEND.source_activity_kbq,
                        "ambient_background_cps": BACKEND.ambient_background_cps,
                        "ai_model_status": BACKEND.get_ai_status(),
                    },
                )

            self._send_json({
                "ok": True,
                "running": BACKEND.running,
                "ai_model_status": BACKEND.get_ai_status(),
            })
            return

        if path == "/api/simulate-once":
            randomize = bool(body.get("randomize_position", False))
            if randomize:
                with BACKEND.lock:
                    BACKEND.true_source = {
                        "x": round(float(BACKEND.mc_engine.rng.uniform(1.6, DOMAIN_SIZE - 1.6)), 2),
                        "y": round(float(BACKEND.mc_engine.rng.uniform(1.6, DOMAIN_SIZE - 1.6)), 2),
                    }
            frame = BACKEND.execute_single_run(
                mode="NEW-EVENT" if randomize else "SINGLE-STEP",
                broadcast=True,
            )
            self._send_json({"ok": True, "frame": frame})
            return

        if path == "/api/run-batch":
            n_runs = max(10, min(500, int(body.get("n_runs", 100))))
            BACKEND.run_validation_batch_streamed(n_runs=n_runs)
            self._send_json({"ok": True, "started": True, "n_runs": n_runs})
            return

        if path == "/api/train-ai":
            train_source = str(body.get("train_source", "logged_runs"))
            n_samples = max(8, min(512, int(body.get("n_samples", 64))))
            epochs = max(5, min(300, int(body.get("epochs", 50))))
            res = BACKEND.run_ai_training_streamed(
                train_source=train_source, n_samples=n_samples, epochs=epochs
            )
            self._send_json(res, status=200 if res.get("ok") else 400)
            return

        if path == "/api/capture-snapshot":
            res = BACKEND.save_snapshot_bundle(body)
            self._send_json(res)
            return

        if path == "/api/ai/scrub":
            BACKEND.scrub_ai_only()
            self._send_json({"ok": True, "ai_model_status": BACKEND.get_ai_status()})
            return

        if path == "/api/scrub-all":
            BACKEND.scrub_all_data_and_ai()
            self._send_json({"ok": True, "state": BACKEND.get_full_state()})
            return

        if path == "/api/logs/clear":
            with BACKEND.lock:
                BACKEND.run_logs.clear()
                BACKEND.run_counter = 0
                try:
                    with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
                        f.truncate(0)
                except OSError:
                    pass
                BACKEND._save_session_state()
            BACKEND.broadcast_event("logs_cleared", {})
            self._send_json({"ok": True})
            return

        self._send_json({"error": "Not found"}, status=404)

    def _handle_export_csv(self) -> None:
        with BACKEND.lock:
            logs = list(BACKEND.run_logs)

        buf = io.StringIO()
        writer = csv.writer(buf)
        header_row = [
            "run_id",
            "timestamp_utc",
            "mode",
            "facility",
            "mc_histories",
            "mc_elapsed_ms",
            "true_x_m",
            "true_y_m",
            "true_activity_kbq",
            "background_cps",
            "ai_is_trained",
            "ai_epochs",
            "ai_x_m",
            "ai_y_m",
            "ai_activity_kbq",
            "ai_error_m",
            "ai_latency_ms",
        ] + [f"S-{i:02d}_cps" for i in range(1, 17)]
        writer.writerow(header_row)

        for row in logs:
            counts: List[Any] = row.get("raw_counts", [0] * 16)
            row_values = [
                row["run_id"],
                row["timestamp_utc"],
                row["mode"],
                row["facility"],
                row["histories"],
                row["mc_elapsed_ms"],
                row["true_x"],
                row["true_y"],
                row["true_activity_kbq"],
                row["background_cps"],
                row.get("ai_is_trained", False),
                row.get("ai_epochs", 0),
                row["ai_x"],
                row["ai_y"],
                row["ai_activity_kbq"],
                row["ai_error_m"],
                row["ai_latency_ms"],
            ] + list(counts)
            writer.writerow(row_values)

        data = buf.getvalue().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="rad_mon_04_run_logs.csv"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_export_json(self) -> None:
        with BACKEND.lock:
            payload = {
                "system_id": "RAD-MON-04",
                "exported_at_utc": BACKEND._utc_now_str(),
                "total_runs": len(BACKEND.run_logs),
                "ai_model_status": BACKEND.get_ai_status(),
                "runs": BACKEND.run_logs,
            }
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="rad_mon_04_run_logs.json"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_export_training_csv(self) -> None:
        with BACKEND.lock:
            samples = list(BACKEND.ai_net.training_samples_log)

        buf = io.StringIO()
        writer = csv.writer(buf)
        header_row = [
            "sample_id",
            "timestamp_utc",
            "source_tag",
            "facility",
            "epochs_applied",
            "true_x_m",
            "true_y_m",
            "true_activity_kbq",
            "background_cps",
            "pre_train_err_m",
            "post_pred_x_m",
            "post_pred_y_m",
            "post_pred_act_kbq",
            "post_train_err_m",
        ] + [f"S-{i:02d}_cps" for i in range(1, 17)]
        writer.writerow(header_row)

        for s in samples:
            counts: List[Any] = s.get("raw_counts", [0] * 16)
            row_values = [
                s["sample_id"],
                s["timestamp_utc"],
                s["source_tag"],
                s["facility"],
                s["epochs_applied"],
                s["true_x"],
                s["true_y"],
                s["true_activity_kbq"],
                s["background_cps"],
                s["pre_err_m"],
                s["post_pred_x"],
                s["post_pred_y"],
                s["post_pred_act"],
                s["post_err_m"],
            ] + list(counts)
            writer.writerow(row_values)

        data = buf.getvalue().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="rad_mon_04_ai_training_data.csv"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_export_training_json(self) -> None:
        with BACKEND.lock:
            payload = {
                "system_id": "RAD-MON-04",
                "exported_at_utc": BACKEND._utc_now_str(),
                "ai_model_status": BACKEND.get_ai_status(),
                "total_training_samples": len(BACKEND.ai_net.training_samples_log),
                "training_samples": BACKEND.ai_net.training_samples_log,
            }
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="rad_mon_04_ai_training_data.json"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_export_model(self) -> None:
        with BACKEND.lock:
            bundle = BACKEND.ai_net.export_model_bundle(BACKEND.facility_key)
            fkey = BACKEND.facility_key
        data = json.dumps(bundle, indent=2).encode("utf-8")
        fname = f"radnet_ai_model_{fkey}.json"
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_export_openmc_script(self) -> None:
        with BACKEND.lock:
            script = OpenMCEngine.generate_openmc_python_deck(
                facility_key=BACKEND.facility_key,
                source_x=BACKEND.true_source["x"],
                source_y=BACKEND.true_source["y"],
                activity_kbq=BACKEND.source_activity_kbq,
                histories=BACKEND.mc_histories,
            )
        data = script.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/x-python; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="openmc_facility_model.py"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    os.chdir(BASE_DIR)
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), DashboardRequestHandler)
    st = BACKEND.get_ai_status()
    print(f"[RAD-MON-04] OpenMC & Pure AI Telemetry Backend listening on http://localhost:{port}")
    print(
        f"[RAD-MON-04] Restored AI State: {st['knowledge_state']} "
        f"({st['trained_epochs']} epochs, {st['samples_seen']} samples) | "
        f"Restored Runs: {len(BACKEND.run_logs)}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[RAD-MON-04] Shutting down server (all AI weights & logs persisted to data/).")
        server.server_close()


if __name__ == "__main__":
    main()
