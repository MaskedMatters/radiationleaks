"""
backend/api/exporter.py
Builds CSV and JSON export payloads for OpenMC simulation run logs and AI training datasets.
"""

import csv
import io
import json
from typing import Any, Dict, List


def build_run_logs_csv(logs: List[Dict[str, Any]]) -> bytes:
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

    return buf.getvalue().encode("utf-8")


def build_run_logs_json(
    exported_at_utc: str,
    ai_model_status: Dict[str, Any],
    logs: List[Dict[str, Any]],
) -> bytes:
    payload = {
        "system_id": "RAD-MON-04",
        "exported_at_utc": exported_at_utc,
        "total_runs": len(logs),
        "ai_model_status": ai_model_status,
        "runs": logs,
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def build_training_samples_csv(samples: List[Dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    header_row = [
        "sample_id",
        "timestamp_utc",
        "source_tag",
        "facility",
        "epochs_applied",
        "outcome",
        "true_x_m",
        "true_y_m",
        "true_activity_kbq",
        "background_cps",
        "pre_pred_x_m",
        "pre_pred_y_m",
        "pre_train_err_m",
        "post_pred_x_m",
        "post_pred_y_m",
        "post_pred_act_kbq",
        "post_train_err_m",
        "reward_delta_m",
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
            s.get("outcome", "REWARDED"),
            s["true_x"],
            s["true_y"],
            s["true_activity_kbq"],
            s["background_cps"],
            s.get("pre_pred_x", ""),
            s.get("pre_pred_y", ""),
            s["pre_err_m"],
            s["post_pred_x"],
            s["post_pred_y"],
            s["post_pred_act"],
            s["post_err_m"],
            s.get("reward_delta_m", 0.0),
        ] + list(counts)
        writer.writerow(row_values)

    return buf.getvalue().encode("utf-8")


def build_training_samples_json(
    exported_at_utc: str,
    ai_model_status: Dict[str, Any],
    samples: List[Dict[str, Any]],
) -> bytes:
    payload = {
        "system_id": "RAD-MON-04",
        "exported_at_utc": exported_at_utc,
        "ai_model_status": ai_model_status,
        "total_training_samples": len(samples),
        "training_samples": samples,
    }
    return json.dumps(payload, indent=2).encode("utf-8")

