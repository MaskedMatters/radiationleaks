"""
backend/ai/pathway_analyzer.py
Computes neural pathway reinforcement, synapse attribution (+Δ|W| rewarded vs. -Δ|W| penalized),
and spatial training trajectories for the dashboard visualizer.
"""

from typing import Any, Dict, List
import numpy as np

from backend.ai.state import SectorModelState


def compute_pathway_telemetry(
    facility_key: str,
    w: SectorModelState,
    a1: np.ndarray,
    a2: np.ndarray,
    out_sig: np.ndarray,
    training_samples_log: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Computes a structured visualization snapshot of the AI's internal neural pathways:
    - Which sensor channels [S-01..S-16] have the strongest learned synaptic weights
      and how much they were reinforced (REWARDED) vs suppressed (PENALIZED) relative
      to initialization.
    - Top 8 hidden layer H1 nodes and Top 6 hidden layer H2 nodes, with their connecting
      synapse edges (Inputs -> H1 -> H2 -> Outputs [X, Y, Activity]), labeled as
      REWARDED (magnitude increased along negative loss gradient) or PENALIZED
      (magnitude decreased / inhibited).
    - Recent spatial training trajectories (pre-train penalized guess -> post-train
      rewarded shift -> true target) for plotting on the 2D floorplan and pathway panel.
    """
    is_trained = w.trained_epochs > 0

    # Combine relative-count weight [0..15] and log-count weight [16..31] per sensor channel
    W1_curr = w.W1[:16, :] + w.W1[16:32, :]  # (16, 128)
    W1_base = w.W1_init[:16, :] + w.W1_init[16:32, :]  # (16, 128)

    # Effective end-to-end linear sensitivity matrix from 16 sensors to 3 outputs: (16, 3)
    eff_io = W1_curr @ w.W2 @ w.W3  # (16, 3)

    sensor_attribution: List[Dict[str, Any]] = []
    for i in range(16):
        strength_now = float(np.linalg.norm(W1_curr[i, :]))
        strength_init = float(np.linalg.norm(W1_base[i, :]))
        delta_s = strength_now - strength_init
        sensor_attribution.append({
            "id": f"S-{i + 1:02d}",
            "synapse_strength": round(strength_now, 3),
            "delta_strength": round(delta_s, 3),
            "status": "REWARDED" if delta_s >= 0.0 else "PENALIZED",
            "influence_x": round(float(eff_io[i, 0]), 3),
            "influence_y": round(float(eff_io[i, 1]), 3),
            "influence_act": round(float(eff_io[i, 2]), 3),
        })

    # Select top 8 H1 neurons and top 6 H2 neurons by learned weight change + activation
    h1_scores = np.linalg.norm(w.W1 - w.W1_init, axis=0) + 0.2 * np.linalg.norm(w.W1, axis=0)
    h2_scores = np.linalg.norm(w.W2 - w.W2_init, axis=0) + 0.2 * np.linalg.norm(w.W2, axis=0)
    top_h1 = [int(idx) for idx in np.argsort(h1_scores)[-8:][::-1].tolist()]
    top_h2 = [int(idx) for idx in np.argsort(h2_scores)[-6:][::-1].tolist()]

    edges_in_h1: List[Dict[str, Any]] = []
    for s_idx in range(16):
        for h1_rank, h1_idx in enumerate(top_h1):
            w_now = float(W1_curr[s_idx, h1_idx])
            w_init = float(W1_base[s_idx, h1_idx])
            delta_mag = abs(w_now) - abs(w_init)
            edges_in_h1.append({
                "from": s_idx,
                "to": h1_rank,
                "weight": round(w_now, 4),
                "delta": round(delta_mag, 4),
                "rewarded": bool(delta_mag >= 0.0),
            })

    edges_h1_h2: List[Dict[str, Any]] = []
    for h1_rank, h1_idx in enumerate(top_h1):
        for h2_rank, h2_idx in enumerate(top_h2):
            w_now = float(w.W2[h1_idx, h2_idx])
            w_init = float(w.W2_init[h1_idx, h2_idx])
            delta_mag = abs(w_now) - abs(w_init)
            edges_h1_h2.append({
                "from": h1_rank,
                "to": h2_rank,
                "weight": round(w_now, 4),
                "delta": round(delta_mag, 4),
                "rewarded": bool(delta_mag >= 0.0),
            })

    edges_h2_out: List[Dict[str, Any]] = []
    for h2_rank, h2_idx in enumerate(top_h2):
        for out_idx in range(3):
            w_now = float(w.W3[h2_idx, out_idx])
            w_init = float(w.W3_init[h2_idx, out_idx])
            delta_mag = abs(w_now) - abs(w_init)
            edges_h2_out.append({
                "from": h2_rank,
                "to": out_idx,
                "weight": round(w_now, 4),
                "delta": round(delta_mag, 4),
                "rewarded": bool(delta_mag >= 0.0),
            })

    # Count total rewarded vs penalized synapses across the entire network
    d1 = np.abs(w.W1) - np.abs(w.W1_init)
    d2 = np.abs(w.W2) - np.abs(w.W2_init)
    d3 = np.abs(w.W3) - np.abs(w.W3_init)
    if is_trained:
        rewarded_synapses = int(np.sum(d1 > 1e-4) + np.sum(d2 > 1e-4) + np.sum(d3 > 1e-4))
        penalized_synapses = int(np.sum(d1 < -1e-4) + np.sum(d2 < -1e-4) + np.sum(d3 < -1e-4))
    else:
        rewarded_synapses = 0
        penalized_synapses = 0

    sector_samples = [
        s for s in training_samples_log
        if s.get("facility") == facility_key and "pre_pred_x" in s
    ][-40:]

    return {
        "facility_key": facility_key,
        "is_trained": is_trained,
        "trained_epochs": w.trained_epochs,
        "samples_seen": w.samples_seen,
        "rewarded_synapses": rewarded_synapses,
        "penalized_synapses": penalized_synapses,
        "total_synapses": int(w.W1.size + w.W2.size + w.W3.size),
        "h1_nodes": [
            {"rank": r, "neuron_id": f"H1-{idx:03d}", "activation": round(float(a1[0, idx]), 3)}
            for r, idx in enumerate(top_h1)
        ],
        "h2_nodes": [
            {"rank": r, "neuron_id": f"H2-{idx:02d}", "activation": round(float(a2[0, idx]), 3)}
            for r, idx in enumerate(top_h2)
        ],
        "out_nodes": [
            {"idx": 0, "label": "EST X [m]", "sig": round(float(out_sig[0, 0]), 3)},
            {"idx": 1, "label": "EST Y [m]", "sig": round(float(out_sig[0, 1]), 3)},
            {"idx": 2, "label": "ACT [kBq]", "sig": round(float(out_sig[0, 2]), 3)},
        ],
        "edges_in_h1": edges_in_h1,
        "edges_h1_h2": edges_h1_h2,
        "edges_h2_out": edges_h2_out,
        "sensor_attribution": sensor_attribution,
        "epoch_curve": w.epoch_curve[-120:],
        "spatial_trajectories": sector_samples,
    }

