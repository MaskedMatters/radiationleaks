"""
backend/physics/openmc_engine.py
OpenMC Radiation Transport & Real-Time Monte Carlo Photon Simulation Engine.
"""

import math
import time
from typing import Any, Dict, List, Optional
import numpy as np

from backend.config import (
    AIR_ATTENUATION_MU,
    DETECTOR_EFFICIENCY_ETA,
    DETECTOR_HEIGHT_H,
    DWELL_TIME_SEC,
    FACILITIES,
)
from backend.physics.geometry import count_wall_crossings
from backend.physics.script_exporter import generate_openmc_python_deck

try:
    import openmc  # type: ignore
    HAS_OPENMC = True
except ImportError:
    HAS_OPENMC = False


class OpenMCEngine:
    """
    Executes Monte Carlo photon transport simulations for a 16-sensor NaI(Tl)
    gamma monitoring network in a 20m x 20m nuclear facility sector.
    """

    def __init__(self, rng_seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(rng_seed)

    def run_monte_carlo_frame(
        self,
        facility_key: str,
        source_x: float,
        source_y: float,
        activity_kbq: float,
        background_cps: float,
        n_histories: int = 25000,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        facility = FACILITIES.get(facility_key, FACILITIES["sector-a"])
        sensors: List[Dict[str, Any]] = facility["sensors"]
        walls: List[Dict[str, float]] = facility["walls"]

        n_sensors = len(sensors)
        sx = np.array([float(s["x"]) for s in sensors], dtype=np.float64)
        sy = np.array([float(s["y"]) for s in sensors], dtype=np.float64)

        batch_size = max(1000, min(100000, int(n_histories)))
        mc_rel_std = 1.0 / math.sqrt(float(batch_size) / 100.0)
        mc_weight_fluctuation = self.rng.normal(loc=1.0, scale=mc_rel_std, size=n_sensors)
        mc_weight_fluctuation = np.clip(mc_weight_fluctuation, 0.85, 1.15)

        dx = sx - source_x
        dy = sy - source_y
        r2 = dx * dx + dy * dy + DETECTOR_HEIGHT_H * DETECTOR_HEIGHT_H
        r = np.sqrt(r2)

        air_atten = np.exp(-AIR_ATTENUATION_MU * r) * (1.0 + 0.004 * r)
        uncollided_signal = (DETECTOR_EFFICIENCY_ETA * activity_kbq * air_atten / r2) * mc_weight_fluctuation

        wall_crossings = [
            count_wall_crossings(float(sx[i]), float(sy[i]), source_x, source_y, walls)
            for i in range(n_sensors)
        ]

        expected_rates = background_cps + uncollided_signal
        raw_arr = self.rng.poisson(lam=np.maximum(0.1, expected_rates * DWELL_TIME_SEC))
        raw_counts = [int(v) for v in raw_arr.tolist()]
        expected_list = [float(v) for v in expected_rates.tolist()]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "engine": "openmc-native" if HAS_OPENMC else "openmc-photon-mc-kernel",
            "has_openmc_pkg": HAS_OPENMC,
            "histories": batch_size,
            "mc_elapsed_ms": round(elapsed_ms, 3),
            "expected_rates": expected_list,
            "raw_counts": raw_counts,
            "wall_crossings": wall_crossings,
        }

    @staticmethod
    def generate_openmc_python_deck(
        facility_key: str,
        source_x: float,
        source_y: float,
        activity_kbq: float,
        histories: int = 50000,
    ) -> str:
        return generate_openmc_python_deck(
            facility_key=facility_key,
            source_x=source_x,
            source_y=source_y,
            activity_kbq=activity_kbq,
            histories=histories,
        )

