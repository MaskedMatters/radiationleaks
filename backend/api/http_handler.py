"""
backend/api/http_handler.py
HTTP & Server-Sent Events (SSE) Request Handler for the RAD-MON-04 Control Center.
Serves modular frontend assets from `frontend/` and routes `/api/*` endpoints.
"""

from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import posixpath
import queue
from typing import Any, Dict
from urllib.parse import unquote, urlparse

from backend.api.exporter import (
    build_run_logs_csv,
    build_run_logs_json,
    build_training_samples_csv,
    build_training_samples_json,
)
from backend.api.state_manager import BACKEND
from backend.config import BASE_DIR, CAPTURES_DIR, DOMAIN_SIZE, FACILITIES, FRONTEND_DIR, LOG_FILE_PATH
from backend.physics.openmc_engine import OpenMCEngine


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        _ = (format, args)

    def translate_path(self, path: str) -> str:
        """Maps static file requests to `frontend/` (or `captures/` for saved snapshots)."""
        clean_path = urlparse(path).path
        clean_path = posixpath.normpath(unquote(clean_path))
        words = [w for w in clean_path.split("/") if w and w not in (".", "..")]

        if words and words[0] == "captures":
            return os.path.join(CAPTURES_DIR, *words[1:])

        if not words:
            return os.path.join(FRONTEND_DIR, "index.html")

        candidate = os.path.join(FRONTEND_DIR, *words)
        if os.path.exists(candidate):
            return candidate
        return os.path.join(BASE_DIR, *words)

    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_bytes(
        self, data: bytes, content_type: str, filename: str, status: int = 200
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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
            with BACKEND.lock:
                logs = list(BACKEND.run_logs)
            self._send_bytes(
                build_run_logs_csv(logs),
                "text/csv; charset=utf-8",
                "rad_mon_04_run_logs.csv",
            )
            return

        if path == "/api/export/logs.json":
            with BACKEND.lock:
                data = build_run_logs_json(
                    BACKEND._utc_now_str(),
                    BACKEND.get_ai_status(),
                    list(BACKEND.run_logs),
                )
            self._send_bytes(
                data,
                "application/json; charset=utf-8",
                "rad_mon_04_run_logs.json",
            )
            return

        if path == "/api/export/training_data.csv":
            with BACKEND.lock:
                samples = list(BACKEND.ai_net.training_samples_log)
            self._send_bytes(
                build_training_samples_csv(samples),
                "text/csv; charset=utf-8",
                "rad_mon_04_ai_training_data.csv",
            )
            return

        if path == "/api/export/training_data.json":
            with BACKEND.lock:
                data = build_training_samples_json(
                    BACKEND._utc_now_str(),
                    BACKEND.get_ai_status(),
                    list(BACKEND.ai_net.training_samples_log),
                )
            self._send_bytes(
                data,
                "application/json; charset=utf-8",
                "rad_mon_04_ai_training_data.json",
            )
            return

        if path == "/api/export/model.json":
            with BACKEND.lock:
                bundle = BACKEND.ai_net.export_model_bundle(BACKEND.facility_key)
                fkey = BACKEND.facility_key
            self._send_bytes(
                json.dumps(bundle, indent=2).encode("utf-8"),
                "application/json; charset=utf-8",
                f"radnet_ai_model_{fkey}.json",
            )
            return

        if path == "/api/export/openmc.py":
            with BACKEND.lock:
                script = OpenMCEngine.generate_openmc_python_deck(
                    facility_key=BACKEND.facility_key,
                    source_x=BACKEND.true_source["x"],
                    source_y=BACKEND.true_source["y"],
                    activity_kbq=BACKEND.source_activity_kbq,
                    histories=BACKEND.mc_histories,
                )
            self._send_bytes(
                script.encode("utf-8"),
                "text/x-python; charset=utf-8",
                "openmc_facility_model.py",
            )
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
                    BACKEND.source_activity_kbq = max(
                        5.0, min(15000.0, float(body["source_activity_kbq"]))
                    )
                if "ambient_background_cps" in body:
                    BACKEND.ambient_background_cps = max(
                        1.0, min(500.0, float(body["ambient_background_cps"]))
                    )
                if "mc_histories" in body:
                    BACKEND.mc_histories = max(
                        1000, min(100000, int(body["mc_histories"]))
                    )
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
                        "pathway_telemetry": BACKEND.get_pathway_telemetry(),
                    },
                )

            self._send_json({
                "ok": True,
                "running": BACKEND.running,
                "ai_model_status": BACKEND.get_ai_status(),
                "pathway_telemetry": BACKEND.get_pathway_telemetry(),
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


def run_server() -> None:
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

