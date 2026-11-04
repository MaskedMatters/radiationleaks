"""
backend/physics/geometry.py
Ray-wall intersection math, slant-path concrete shielding attenuation, and expected
detector count rate physics.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from backend.config import (
    AIR_ATTENUATION_MU,
    DETECTOR_EFFICIENCY_ETA,
    DETECTOR_HEIGHT_H,
    WALL_LINEAR_ATTEN_MU,
    WALL_SCATTER_FLOOR,
    WALL_THICKNESS_M,
)


def segments_intersect(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float
) -> bool:
    """Checks if 2D line segment AB intersects segment CD."""
    def ccw(p1x: float, p1y: float, p2x: float, p2y: float, p3x: float, p3y: float) -> bool:
        return (p3y - p1y) * (p2x - p1x) > (p2y - p1y) * (p3x - p1x)

    return (
        ccw(ax, ay, cx, cy, dx, dy) != ccw(bx, by, cx, cy, dx, dy)
        and ccw(ax, ay, bx, by, cx, cy) != ccw(ax, ay, bx, by, dx, dy)
    )


def compute_ray_wall_shielding(
    sx: float,
    sy: float,
    tx: float,
    ty: float,
    walls: List[Dict[str, float]],
) -> Tuple[int, float]:
    """
    Computes the number of concrete wall crossings and the net photon transmission factor
    (0.0 .. 1.0) along the ray from source (tx, ty) to sensor (sx, sy).

    Accounts for oblique incidence angle (slant path length through concrete wall)
    and 3D elevation slant factor.
    """
    ux = sx - tx
    uy = sy - ty
    ray_len_2d = math.hypot(ux, uy)
    if ray_len_2d < 1e-6 or not walls:
        return 0, 1.0

    slant_3d = math.hypot(ray_len_2d, DETECTOR_HEIGHT_H) / ray_len_2d
    crossings = 0
    transmission = 1.0

    for w in walls:
        x1, y1, x2, y2 = float(w["x1"]), float(w["y1"]), float(w["x2"]), float(w["y2"])
        if segments_intersect(tx, ty, sx, sy, x1, y1, x2, y2):
            crossings += 1
            vx = x2 - x1
            vy = y2 - y1
            wall_len = math.hypot(vx, vy)
            if wall_len > 1e-6:
                sin_alpha = abs(ux * vy - uy * vx) / (ray_len_2d * wall_len)
            else:
                sin_alpha = 1.0
            sin_clamped = max(0.35, min(1.0, sin_alpha))
            d_concrete = (WALL_THICKNESS_M / sin_clamped) * slant_3d
            wall_trans = (
                (1.0 - WALL_SCATTER_FLOOR) * math.exp(-WALL_LINEAR_ATTEN_MU * d_concrete)
                + WALL_SCATTER_FLOOR
            )
            transmission *= wall_trans

    return crossings, transmission


def count_wall_crossings(
    sx: float, sy: float, tx: float, ty: float, walls: List[Dict[str, float]]
) -> int:
    """Counts how many concrete wall segments intersect the ray from source to detector."""
    crossings, _ = compute_ray_wall_shielding(sx, sy, tx, ty, walls)
    return crossings


def expected_sensor_rates_np(
    sx: np.ndarray,
    sy: np.ndarray,
    x: float,
    y: float,
    activity_kbq: float,
    background_cps: float,
    walls: Optional[List[Dict[str, float]]] = None,
) -> np.ndarray:
    """
    Computes expected sensor count rates for a given source position and activity,
    including concrete wall attenuation when `walls` are provided.
    """
    dx = sx - x
    dy = sy - y
    r2 = dx * dx + dy * dy + DETECTOR_HEIGHT_H * DETECTOR_HEIGHT_H
    r = np.sqrt(r2)
    atten = np.exp(-AIR_ATTENUATION_MU * r) * (1.0 + 0.004 * r)
    unshielded_signal = (DETECTOR_EFFICIENCY_ETA * activity_kbq * atten) / r2

    if walls:
        n_sensors = int(sx.shape[0])
        trans_arr = np.ones(n_sensors, dtype=np.float64)
        for i in range(n_sensors):
            _, tr = compute_ray_wall_shielding(float(sx[i]), float(sy[i]), x, y, walls)
            trans_arr[i] = tr
        return background_cps + unshielded_signal * trans_arr

    return background_cps + unshielded_signal
