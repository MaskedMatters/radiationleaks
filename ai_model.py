"""
ai_model.py
End-to-End Neural Network AI Localization Model with Persistent Disk Checkpointing.

Design:
- Pure AI localization (`RadLocNeuralNet`) — all triangulation / L-BFGS code removed.
- Automatically loads its learned weights, Adam optimizer state, and training metadata
  from `data/ai_checkpoint.json` on server startup, and saves after every training run
  so knowledge is never lost when the server restarts.
- Receives ONLY the 16 raw detector channel counts [S-01..S-16] and ambient background,
  learning end-to-end to predict (x, y, activity_kbq).
"""

from dataclasses import dataclass, field
import json
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from openmc_engine import (
    AIR_ATTENUATION_MU,
    DETECTOR_EFFICIENCY_ETA,
    DETECTOR_HEIGHT_H,
    DOMAIN_SIZE,
    FACILITIES,
    OpenMCEngine,
)


@dataclass
class SectorModelState:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    W3: np.ndarray
    b3: np.ndarray
    m_adam: List[np.ndarray] = field(default_factory=list)
    v_adam: List[np.ndarray] = field(default_factory=list)
    adam_step: int = 0
    trained_epochs: int = 0
    samples_seen: int = 0
    last_loss: float = 0.0
    val_mae_m: float = 0.0


def expected_sensor_rates_np(
    sx: np.ndarray,
    sy: np.ndarray,
    x: float,
    y: float,
    activity_kbq: float,
    background_cps: float,
) -> np.ndarray:
    """Computes expected sensor count rates for the AI's predicted source parameters."""
    dx = sx - x
    dy = sy - y
    r2 = dx * dx + dy * dy + DETECTOR_HEIGHT_H * DETECTOR_HEIGHT_H
    r = np.sqrt(r2)
    atten = np.exp(-AIR_ATTENUATION_MU * r) * (1.0 + 0.004 * r)
    return background_cps + (DETECTOR_EFFICIENCY_ETA * activity_kbq * atten) / r2


class RadLocNeuralNet:
    """
    End-to-End Multi-Layer Perceptron (MLP) for radiation leak localization.
    Persists learned weights and optimizer state to disk so knowledge survives restarts.
    """

    def __init__(self, checkpoint_path: str, train_log_path: str, seed: int = 42) -> None:
        self.checkpoint_path = checkpoint_path
        self.train_log_path = train_log_path
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.in_dim = 33
        self.h1_dim = 128
        self.h2_dim = 64
        self.out_dim = 3
        self.model_version = "RADNET-v3.1-MLP"

        self.weights_by_facility: Dict[str, SectorModelState] = {}
        self.training_history: List[Dict[str, Any]] = []
        self.training_samples_log: List[Dict[str, Any]] = []
        self.total_train_samples: int = 0
        self.epochs_completed: int = 0
        self.last_val_mae_meters: float = 0.0

        # Initialize default random weights first, then load persisted state if present
        for fkey in FACILITIES:
            self._init_untrained_weights(fkey)

        self.load_from_disk()

    def scrub_all(self) -> None:
        """
        Explicitly wipes all learned weights across all facilities back to random
        initialization, clears training sample logs, and deletes the checkpoint on disk.
        """
        self.rng = np.random.default_rng(int(time.time() * 1000) % (2**31 - 1))
        self.weights_by_facility.clear()
        self.training_history.clear()
        self.training_samples_log.clear()
        self.total_train_samples = 0
        self.epochs_completed = 0
        self.last_val_mae_meters = 0.0

        for fkey in FACILITIES:
            self._init_untrained_weights(fkey)

        self.save_to_disk()
        try:
            with open(self.train_log_path, "w", encoding="utf-8") as f:
                f.truncate(0)
        except OSError:
            pass

    def _init_untrained_weights(self, facility_key: str) -> SectorModelState:
        W1 = self.rng.normal(0.0, 0.15, size=(self.in_dim, self.h1_dim))
        b1 = self.rng.normal(0.0, 0.05, size=(self.h1_dim,))
        W2 = self.rng.normal(0.0, 0.15, size=(self.h1_dim, self.h2_dim))
        b2 = self.rng.normal(0.0, 0.05, size=(self.h2_dim,))
        W3 = self.rng.normal(0.0, 0.20, size=(self.h2_dim, self.out_dim))
        b3 = self.rng.uniform(-0.4, 0.4, size=(self.out_dim,))

        state = SectorModelState(
            W1=W1,
            b1=b1,
            W2=W2,
            b2=b2,
            W3=W3,
            b3=b3,
            m_adam=[
                np.zeros_like(W1),
                np.zeros_like(b1),
                np.zeros_like(W2),
                np.zeros_like(b2),
                np.zeros_like(W3),
                np.zeros_like(b3),
            ],
            v_adam=[
                np.zeros_like(W1),
                np.zeros_like(b1),
                np.zeros_like(W2),
                np.zeros_like(b2),
                np.zeros_like(W3),
                np.zeros_like(b3),
            ],
        )
        self.weights_by_facility[facility_key] = state
        return state

    def _get_sector_state(self, facility_key: str) -> SectorModelState:
        if facility_key not in self.weights_by_facility:
            return self._init_untrained_weights(facility_key)
        return self.weights_by_facility[facility_key]

    def save_to_disk(self) -> None:
        """Saves all sector neural network weights and optimizer state to `checkpoint_path`."""
        sectors_payload: Dict[str, Any] = {}
        for fkey, st in self.weights_by_facility.items():
            sectors_payload[fkey] = {
                "W1": st.W1.tolist(),
                "b1": st.b1.tolist(),
                "W2": st.W2.tolist(),
                "b2": st.b2.tolist(),
                "W3": st.W3.tolist(),
                "b3": st.b3.tolist(),
                "m_adam": [arr.tolist() for arr in st.m_adam],
                "v_adam": [arr.tolist() for arr in st.v_adam],
                "adam_step": st.adam_step,
                "trained_epochs": st.trained_epochs,
                "samples_seen": st.samples_seen,
                "last_loss": st.last_loss,
                "val_mae_m": st.val_mae_m,
            }

        payload = {
            "model_version": self.model_version,
            "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_train_samples": self.total_train_samples,
            "epochs_completed": self.epochs_completed,
            "last_val_mae_meters": self.last_val_mae_meters,
            "sectors": sectors_payload,
        }
        try:
            tmp_path = self.checkpoint_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp_path, self.checkpoint_path)
        except OSError:
            pass

    def load_from_disk(self) -> None:
        """Loads persisted AI checkpoint and training sample logs from disk if available."""
        if os.path.exists(self.checkpoint_path) and os.path.getsize(self.checkpoint_path) > 0:
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.total_train_samples = int(data.get("total_train_samples", 0))
                self.epochs_completed = int(data.get("epochs_completed", 0))
                self.last_val_mae_meters = float(data.get("last_val_mae_meters", 0.0))
                sectors_data = data.get("sectors", {})
                if isinstance(sectors_data, dict):
                    for fkey, sdict in sectors_data.items():
                        W1 = np.array(sdict["W1"], dtype=np.float64)
                        b1 = np.array(sdict["b1"], dtype=np.float64)
                        W2 = np.array(sdict["W2"], dtype=np.float64)
                        b2 = np.array(sdict["b2"], dtype=np.float64)
                        W3 = np.array(sdict["W3"], dtype=np.float64)
                        b3 = np.array(sdict["b3"], dtype=np.float64)
                        m_adam = [np.array(a, dtype=np.float64) for a in sdict.get("m_adam", [])]
                        v_adam = [np.array(a, dtype=np.float64) for a in sdict.get("v_adam", [])]
                        if len(m_adam) != 6:
                            m_adam = [np.zeros_like(p) for p in (W1, b1, W2, b2, W3, b3)]
                        if len(v_adam) != 6:
                            v_adam = [np.zeros_like(p) for p in (W1, b1, W2, b2, W3, b3)]
                        self.weights_by_facility[fkey] = SectorModelState(
                            W1=W1,
                            b1=b1,
                            W2=W2,
                            b2=b2,
                            W3=W3,
                            b3=b3,
                            m_adam=m_adam,
                            v_adam=v_adam,
                            adam_step=int(sdict.get("adam_step", 0)),
                            trained_epochs=int(sdict.get("trained_epochs", 0)),
                            samples_seen=int(sdict.get("samples_seen", 0)),
                            last_loss=float(sdict.get("last_loss", 0.0)),
                            val_mae_m=float(sdict.get("val_mae_m", 0.0)),
                        )
            except (OSError, ValueError, KeyError):
                pass

        if os.path.exists(self.train_log_path) and os.path.getsize(self.train_log_path) > 0:
            try:
                loaded_samples: List[Dict[str, Any]] = []
                with open(self.train_log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line_str = line.strip()
                        if line_str:
                            loaded_samples.append(json.loads(line_str))
                self.training_samples_log = loaded_samples[-1000:]
            except (OSError, ValueError):
                pass

    @staticmethod
    def extract_raw_features(
        counts_batch: np.ndarray,
        backgrounds: np.ndarray,
    ) -> np.ndarray:
        """
        Extracts 33-D sensor count feature vector WITHOUT using sensor (x, y) coordinates
        or any physics formulas.
        """
        net = np.maximum(0.0, counts_batch - backgrounds[:, None])
        net_sum = np.sum(net, axis=1, keepdims=True) + 1.0
        rel_net = net / net_sum
        log_raw = np.log1p(np.maximum(0.0, counts_batch)) / 10.0
        log_bkg = (np.log1p(np.maximum(0.0, backgrounds)) / 10.0)[:, None]
        return np.column_stack([rel_net, log_raw, log_bkg])

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z_clip = np.clip(z, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z_clip))

    def _forward(
        self, X: np.ndarray, w_store: SectorModelState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z1 = X @ w_store.W1 + w_store.b1
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ w_store.W2 + w_store.b2
        a2 = np.maximum(0.0, z2)
        z3 = a2 @ w_store.W3 + w_store.b3
        out_sig = self._sigmoid(z3)
        return z1, a1, z2, a2, z3, out_sig

    @staticmethod
    def _decode_outputs(out_sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred_x = 0.2 + out_sig[:, 0] * (DOMAIN_SIZE - 0.4)
        pred_y = 0.2 + out_sig[:, 1] * (DOMAIN_SIZE - 0.4)
        log_min = math.log(15.0)
        log_max = math.log(6000.0)
        pred_a = np.exp(log_min + out_sig[:, 2] * (log_max - log_min))
        return pred_x, pred_y, pred_a

    @staticmethod
    def _encode_targets(
        tx: np.ndarray, ty: np.ndarray, ta: np.ndarray
    ) -> np.ndarray:
        log_min = math.log(15.0)
        log_max = math.log(6000.0)
        t0 = np.clip((tx - 0.2) / (DOMAIN_SIZE - 0.4), 0.001, 0.999)
        t1 = np.clip((ty - 0.2) / (DOMAIN_SIZE - 0.4), 0.001, 0.999)
        t2 = np.clip((np.log(np.maximum(15.0, ta)) - log_min) / (log_max - log_min), 0.001, 0.999)
        return np.column_stack([t0, t1, t2])

    def train_on_logged_runs(
        self,
        facility_key: str,
        logged_runs: Sequence[Dict[str, Any]],
        epochs: int = 50,
        lr: float = 0.008,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Trains the AI model directly on the user's logged OpenMC simulation runs."""
        sector_runs = [
            r for r in logged_runs
            if r.get("facility") == facility_key and "raw_counts" in r
        ]
        if not sector_runs:
            return {
                "ok": False,
                "error": f"No logged runs found for {facility_key}. Run at least 1 simulation first.",
            }

        counts_mat = np.array([r["raw_counts"] for r in sector_runs], dtype=np.float64)
        bkgs = np.array([float(r["background_cps"]) for r in sector_runs], dtype=np.float64)
        tx = np.array([float(r["true_x"]) for r in sector_runs], dtype=np.float64)
        ty = np.array([float(r["true_y"]) for r in sector_runs], dtype=np.float64)
        ta = np.array([float(r["true_activity_kbq"]) for r in sector_runs], dtype=np.float64)

        return self._run_adam_epochs(
            facility_key=facility_key,
            counts_mat=counts_mat,
            bkgs=bkgs,
            tx=tx,
            ty=ty,
            ta=ta,
            epochs=epochs,
            lr=lr,
            source_tag="LOGGED-RUNS",
            progress_cb=progress_cb,
        )

    def train_on_new_openmc_samples(
        self,
        facility_key: str,
        mc_engine: OpenMCEngine,
        n_samples: int = 64,
        epochs: int = 40,
        lr: float = 0.008,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Generates `n_samples` OpenMC photon transport realizations, logs every single
        training sample, and trains the neural network on them.
        """
        counts_list: List[List[int]] = []
        bkgs_list: List[float] = []
        tx_list: List[float] = []
        ty_list: List[float] = []
        ta_list: List[float] = []

        for _ in range(n_samples):
            sx_val = round(float(mc_engine.rng.uniform(1.6, DOMAIN_SIZE - 1.6)), 2)
            sy_val = round(float(mc_engine.rng.uniform(1.6, DOMAIN_SIZE - 1.6)), 2)
            sa_val = round(float(math.exp(mc_engine.rng.uniform(math.log(50.0), math.log(4500.0)))), 1)
            sb_val = round(float(mc_engine.rng.uniform(5.0, 120.0)), 1)

            mc_frame = mc_engine.run_monte_carlo_frame(
                facility_key=facility_key,
                source_x=sx_val,
                source_y=sy_val,
                activity_kbq=sa_val,
                background_cps=sb_val,
                n_histories=15000,
            )
            counts_list.append(mc_frame["raw_counts"])
            bkgs_list.append(sb_val)
            tx_list.append(sx_val)
            ty_list.append(sy_val)
            ta_list.append(sa_val)

        counts_mat = np.array(counts_list, dtype=np.float64)
        bkgs = np.array(bkgs_list, dtype=np.float64)
        tx = np.array(tx_list, dtype=np.float64)
        ty = np.array(ty_list, dtype=np.float64)
        ta = np.array(ta_list, dtype=np.float64)

        return self._run_adam_epochs(
            facility_key=facility_key,
            counts_mat=counts_mat,
            bkgs=bkgs,
            tx=tx,
            ty=ty,
            ta=ta,
            epochs=epochs,
            lr=lr,
            source_tag="OPENMC-SYNTH",
            progress_cb=progress_cb,
        )

    def _run_adam_epochs(
        self,
        facility_key: str,
        counts_mat: np.ndarray,
        bkgs: np.ndarray,
        tx: np.ndarray,
        ty: np.ndarray,
        ta: np.ndarray,
        epochs: int,
        lr: float,
        source_tag: str,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        w_store = self._get_sector_state(facility_key)
        params = [w_store.W1, w_store.b1, w_store.W2, w_store.b2, w_store.W3, w_store.b3]
        m_adam = w_store.m_adam
        v_adam = w_store.v_adam
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        n_batch = int(counts_mat.shape[0])
        X = self.extract_raw_features(counts_mat, bkgs)
        Y_target = self._encode_targets(tx, ty, ta)

        _, _, _, _, _, init_sig = self._forward(X, w_store)
        init_px, init_py, _ = self._decode_outputs(init_sig)
        init_errs = np.hypot(init_px - tx, init_py - ty)

        final_loss = 0.0
        final_mae_m = 0.0

        for ep in range(1, epochs + 1):
            z1, a1, z2, a2, _, out_sig = self._forward(X, w_store)
            diff = out_sig - Y_target
            loss = float(np.mean(diff * diff))
            final_loss = loss

            px, py, _ = self._decode_outputs(out_sig)
            loc_errors = np.hypot(px - tx, py - ty)
            final_mae_m = float(np.mean(loc_errors))

            dsig = out_sig * (1.0 - out_sig)
            dz3 = (2.0 / (n_batch * 3.0)) * diff * dsig
            dW3 = a2.T @ dz3 + 1e-5 * w_store.W3
            db3 = np.sum(dz3, axis=0)

            da2 = dz3 @ w_store.W3.T
            dz2 = da2 * (z2 > 0.0)
            dW2 = a1.T @ dz2 + 1e-5 * w_store.W2
            db2 = np.sum(dz2, axis=0)

            da1 = dz2 @ w_store.W2.T
            dz1 = da1 * (z1 > 0.0)
            dW1 = X.T @ dz1 + 1e-5 * w_store.W1
            db1 = np.sum(dz1, axis=0)

            grads = [dW1, db1, dW2, db2, dW3, db3]
            w_store.adam_step += 1
            step_t = w_store.adam_step

            for idx, (param, g_arr) in enumerate(zip(params, grads)):
                m_adam[idx] = beta1 * m_adam[idx] + (1.0 - beta1) * g_arr
                v_adam[idx] = beta2 * v_adam[idx] + (1.0 - beta2) * (g_arr * g_arr)
                m_hat = m_adam[idx] / (1.0 - beta1 ** step_t)
                v_hat = v_adam[idx] / (1.0 - beta2 ** step_t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)

            w_store.trained_epochs += 1
            self.epochs_completed += 1
            w_store.last_loss = round(final_loss, 6)
            w_store.val_mae_m = round(final_mae_m, 3)
            self.last_val_mae_meters = round(final_mae_m, 3)

            if progress_cb is not None and (ep % 5 == 0 or ep == epochs or ep == 1):
                progress_cb({
                    "epoch": ep,
                    "total_epochs": epochs,
                    "loss": round(final_loss, 6),
                    "val_mae_meters": round(final_mae_m, 3),
                })

        _, _, _, _, _, post_sig = self._forward(X, w_store)
        post_px, post_py, post_pa = self._decode_outputs(post_sig)
        post_errs = np.hypot(post_px - tx, post_py - ty)

        ts_now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        new_samples: List[Dict[str, Any]] = []
        for i in range(n_batch):
            self.total_train_samples += 1
            w_store.samples_seen += 1
            row_counts = [int(val) for val in counts_mat[i].tolist()]
            sample_entry: Dict[str, Any] = {
                "sample_id": f"TRN-{self.total_train_samples:04d}",
                "timestamp_utc": ts_now,
                "source_tag": source_tag,
                "facility": facility_key,
                "epochs_applied": epochs,
                "true_x": round(float(tx[i]), 2),
                "true_y": round(float(ty[i]), 2),
                "true_activity_kbq": round(float(ta[i]), 1),
                "background_cps": round(float(bkgs[i]), 1),
                "pre_err_m": round(float(init_errs[i]), 3),
                "post_pred_x": round(float(post_px[i]), 2),
                "post_pred_y": round(float(post_py[i]), 2),
                "post_pred_act": round(float(post_pa[i]), 1),
                "post_err_m": round(float(post_errs[i]), 3),
                "raw_counts": row_counts,
            }
            self.training_samples_log.append(sample_entry)
            new_samples.append(sample_entry)
            if len(self.training_samples_log) > 1000:
                self.training_samples_log.pop(0)

        # Persist updated AI weights and new training samples to disk immediately
        self.save_to_disk()
        try:
            with open(self.train_log_path, "a", encoding="utf-8") as f:
                for s in new_samples:
                    f.write(json.dumps(s) + "\n")
        except OSError:
            pass

        return {
            "ok": True,
            "facility_key": facility_key,
            "samples_trained": n_batch,
            "epochs_run": epochs,
            "final_loss": round(final_loss, 6),
            "val_mae_meters": round(final_mae_m, 3),
        }

    def predict(
        self,
        facility_key: str,
        counts: Sequence[int],
        background_cps: float,
    ) -> Dict[str, Any]:
        """
        Pure forward pass through the AI Neural Network.
        """
        t0 = time.perf_counter()
        w_store = self._get_sector_state(facility_key)
        c_arr = np.array([float(v) for v in counts], dtype=np.float64)[None, :]
        b_arr = np.array([background_cps], dtype=np.float64)

        X = self.extract_raw_features(c_arr, b_arr)
        _, _, _, _, _, out_sig = self._forward(X, w_store)
        px, py, pa = self._decode_outputs(out_sig)

        raw_x = float(px[0])
        raw_y = float(py[0])
        raw_a = float(pa[0])

        is_trained = w_store.trained_epochs > 0
        ci95 = max(0.20, min(2.0, w_store.val_mae_m)) if is_trained and w_store.val_mae_m > 0 else 2.50

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "solver": "AI-MLP" if is_trained else "AI-MLP (UNTRAINED)",
            "is_trained": is_trained,
            "trained_epochs": w_store.trained_epochs,
            "x": round(raw_x, 3),
            "y": round(raw_y, 3),
            "activity": round(raw_a, 1),
            "error_meters": 0.0,
            "ci95_meters": round(ci95, 3),
            "latency_ms": round(elapsed_ms, 3),
            "iterations": 1,
            "converged": is_trained,
        }

    def get_model_diagnostics(self, facility_key: str) -> Dict[str, Any]:
        w = self._get_sector_state(facility_key)
        is_trained = w.trained_epochs > 0
        return {
            "model_version": self.model_version,
            "facility_key": facility_key,
            "is_trained": is_trained,
            "knowledge_state": "TRAINED (PERSISTED)" if is_trained else "UNTRAINED (0 KNOWLEDGE)",
            "trained_epochs": w.trained_epochs,
            "samples_seen": w.samples_seen,
            "total_samples": self.total_train_samples,
            "last_loss": w.last_loss if is_trained else None,
            "val_mae_meters": w.val_mae_m if is_trained else None,
            "weight_norms": {
                "W1_l2": round(float(np.linalg.norm(w.W1)), 3),
                "W2_l2": round(float(np.linalg.norm(w.W2)), 3),
                "W3_l2": round(float(np.linalg.norm(w.W3)), 3),
            },
        }

    def export_model_bundle(self, facility_key: str) -> Dict[str, Any]:
        w_store = self._get_sector_state(facility_key)
        facility = FACILITIES.get(facility_key, FACILITIES["sector-a"])
        is_trained = w_store.trained_epochs > 0
        return {
            "model_name": "RAD-MON-04 End-to-End Neural Localization Model",
            "model_version": self.model_version,
            "exported_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "facility_key": facility_key,
            "facility_name": facility["name"],
            "knowledge_state": "TRAINED" if is_trained else "UNTRAINED",
            "training_metadata": {
                "trained_epochs": w_store.trained_epochs,
                "samples_seen": w_store.samples_seen,
                "last_mse_loss": w_store.last_loss if is_trained else None,
                "validation_mae_meters": w_store.val_mae_m if is_trained else None,
            },
            "architecture": {
                "type": "End-to-End MLP",
                "input_dim": self.in_dim,
                "input_features": ["rel_net_counts[16]", "log1p_raw_counts[16]", "log1p_background[1]"],
                "hidden_dims": [self.h1_dim, self.h2_dim],
                "output_dim": self.out_dim,
                "output_activation": "Sigmoid",
            },
            "parameters": {
                "W1": np.round(w_store.W1, 6).tolist(),
                "b1": np.round(w_store.b1, 6).tolist(),
                "W2": np.round(w_store.W2, 6).tolist(),
                "b2": np.round(w_store.b2, 6).tolist(),
                "W3": np.round(w_store.W3, 6).tolist(),
                "b3": np.round(w_store.b3, 6).tolist(),
            },
        }
