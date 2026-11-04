"""
backend/ai/persistence.py
Atomic disk checkpoint saver and loader for `RadLocNeuralNet`.
"""

import json
import os
import time
from typing import Any, Dict, List, Tuple
import numpy as np

from backend.ai.state import SectorModelState


def save_checkpoint_to_disk(
    checkpoint_path: str,
    model_version: str,
    total_train_samples: int,
    epochs_completed: int,
    last_val_mae_meters: float,
    weights_by_facility: Dict[str, SectorModelState],
) -> None:
    """Saves all sector neural network weights, baseline weights, and learning curves to disk."""
    sectors_payload: Dict[str, Any] = {}
    for fkey, st in weights_by_facility.items():
        sectors_payload[fkey] = {
            "W1": st.W1.tolist(),
            "b1": st.b1.tolist(),
            "W2": st.W2.tolist(),
            "b2": st.b2.tolist(),
            "W3": st.W3.tolist(),
            "b3": st.b3.tolist(),
            "W1_init": st.W1_init.tolist(),
            "W2_init": st.W2_init.tolist(),
            "W3_init": st.W3_init.tolist(),
            "last_dW1": st.last_dW1.tolist(),
            "last_dW2": st.last_dW2.tolist(),
            "last_dW3": st.last_dW3.tolist(),
            "m_adam": [arr.tolist() for arr in st.m_adam],
            "v_adam": [arr.tolist() for arr in st.v_adam],
            "adam_step": st.adam_step,
            "trained_epochs": st.trained_epochs,
            "samples_seen": st.samples_seen,
            "last_loss": st.last_loss,
            "val_mae_m": st.val_mae_m,
            "epoch_curve": st.epoch_curve[-300:],
        }

    payload = {
        "model_version": model_version,
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_train_samples": total_train_samples,
        "epochs_completed": epochs_completed,
        "last_val_mae_meters": last_val_mae_meters,
        "sectors": sectors_payload,
    }
    try:
        tmp_path = checkpoint_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, checkpoint_path)
    except OSError:
        pass


def load_checkpoint_from_disk(
    checkpoint_path: str,
    train_log_path: str,
    weights_by_facility: Dict[str, SectorModelState],
) -> Tuple[int, int, float, List[Dict[str, Any]]]:
    """
    Loads persisted AI checkpoint and training sample logs from disk if available.
    Returns (total_train_samples, epochs_completed, last_val_mae_meters, training_samples_log).
    """
    total_train_samples = 0
    epochs_completed = 0
    last_val_mae_meters = 0.0
    training_samples_log: List[Dict[str, Any]] = []

    if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            total_train_samples = int(data.get("total_train_samples", 0))
            epochs_completed = int(data.get("epochs_completed", 0))
            last_val_mae_meters = float(data.get("last_val_mae_meters", 0.0))
            sectors_data = data.get("sectors", {})
            if isinstance(sectors_data, dict):
                for fkey, sdict in sectors_data.items():
                    W1 = np.array(sdict["W1"], dtype=np.float64)
                    b1 = np.array(sdict["b1"], dtype=np.float64)
                    W2 = np.array(sdict["W2"], dtype=np.float64)
                    b2 = np.array(sdict["b2"], dtype=np.float64)
                    W3 = np.array(sdict["W3"], dtype=np.float64)
                    b3 = np.array(sdict["b3"], dtype=np.float64)
                    W1_init = np.array(sdict.get("W1_init", W1.tolist()), dtype=np.float64)
                    W2_init = np.array(sdict.get("W2_init", W2.tolist()), dtype=np.float64)
                    W3_init = np.array(sdict.get("W3_init", W3.tolist()), dtype=np.float64)
                    last_dW1 = np.array(sdict.get("last_dW1", np.zeros_like(W1).tolist()), dtype=np.float64)
                    last_dW2 = np.array(sdict.get("last_dW2", np.zeros_like(W2).tolist()), dtype=np.float64)
                    last_dW3 = np.array(sdict.get("last_dW3", np.zeros_like(W3).tolist()), dtype=np.float64)

                    m_adam = [np.array(a, dtype=np.float64) for a in sdict.get("m_adam", [])]
                    v_adam = [np.array(a, dtype=np.float64) for a in sdict.get("v_adam", [])]
                    if len(m_adam) != 6:
                        m_adam = [np.zeros_like(p) for p in (W1, b1, W2, b2, W3, b3)]
                    if len(v_adam) != 6:
                        v_adam = [np.zeros_like(p) for p in (W1, b1, W2, b2, W3, b3)]

                    raw_curve = sdict.get("epoch_curve", [])
                    epoch_curve = list(raw_curve) if isinstance(raw_curve, list) else []

                    weights_by_facility[fkey] = SectorModelState(
                        W1=W1,
                        b1=b1,
                        W2=W2,
                        b2=b2,
                        W3=W3,
                        b3=b3,
                        W1_init=W1_init,
                        W2_init=W2_init,
                        W3_init=W3_init,
                        last_dW1=last_dW1,
                        last_dW2=last_dW2,
                        last_dW3=last_dW3,
                        m_adam=m_adam,
                        v_adam=v_adam,
                        adam_step=int(sdict.get("adam_step", 0)),
                        trained_epochs=int(sdict.get("trained_epochs", 0)),
                        samples_seen=int(sdict.get("samples_seen", 0)),
                        last_loss=float(sdict.get("last_loss", 0.0)),
                        val_mae_m=float(sdict.get("val_mae_m", 0.0)),
                        epoch_curve=epoch_curve,
                    )
        except (OSError, ValueError, KeyError):
            pass

    if os.path.exists(train_log_path) and os.path.getsize(train_log_path) > 0:
        try:
            loaded_samples: List[Dict[str, Any]] = []
            with open(train_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        loaded_samples.append(json.loads(line_str))
            training_samples_log = loaded_samples[-1000:]
        except (OSError, ValueError):
            pass

    return total_train_samples, epochs_completed, last_val_mae_meters, training_samples_log

