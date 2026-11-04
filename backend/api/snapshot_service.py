"""
backend/api/snapshot_service.py
Decodes base64 PNG data URLs from the frontend and saves timestamped screenshot
bundles (`facility_map.png`, `cdf_error_chart.png`, `sensor_flux_chart.png`,
`neural_pathways.png`, `dashboard_composite.png`, and `telemetry_metadata.json`)
under `captures/`.
"""

import base64
from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List

from backend.config import CAPTURES_DIR


def save_snapshot_bundle_to_disk(
    body: Dict[str, Any],
    run_counter: int,
    captured_at_utc: str,
    facility_key: str,
    true_source: Dict[str, float],
    source_activity_kbq: float,
    ambient_background_cps: float,
    ai_model_status: Dict[str, Any],
    latest_frame: Dict[str, Any],
    validation_errors_ai: List[float],
) -> Dict[str, Any]:
    """Saves PNG screenshots and JSON telemetry metadata into a timestamped folder."""
    now_dt = datetime.now(timezone.utc)
    ts_slug = now_dt.strftime("%Y-%m-%d_%H-%M-%S_UTC")
    run_tag = f"RUN-{run_counter:04d}"
    folder_name = f"snapshot_{ts_slug}_{run_tag}"
    folder_path = os.path.join(CAPTURES_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    image_keys = [
        ("map_png", "facility_map.png"),
        ("cdf_png", "cdf_error_chart.png"),
        ("flux_png", "sensor_flux_chart.png"),
        ("pathway_png", "neural_pathways.png"),
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
        "captured_at_utc": captured_at_utc,
        "folder_name": folder_name,
        "folder_path": folder_path,
        "run_id": run_tag,
        "facility_key": facility_key,
        "true_source": true_source,
        "source_activity_kbq": source_activity_kbq,
        "ambient_background_cps": ambient_background_cps,
        "ai_model_status": ai_model_status,
        "latest_frame": latest_frame,
        "validation_errors_ai": validation_errors_ai,
        "saved_files": saved_files + ["telemetry_metadata.json"],
    }
    meta_dest = os.path.join(folder_path, "telemetry_metadata.json")
    with open(meta_dest, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)

    return {
        "folder_name": folder_name,
        "folder_path": f"captures/{folder_name}",
        "captured_at_utc": captured_at_utc,
        "run_id": run_tag,
        "files": metadata["saved_files"],
    }

