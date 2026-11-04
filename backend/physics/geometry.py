"""
backend/physics/geometry.py
Ray-wall intersection math and expected detector count rate physics.
"""

from typing import Dict, List
import numpy as np

from backend.config import (
    AIR_ATTENUATION_MU,
    DETECTOR_EFFICIENCY_ETA,
    DETECTOR_HEIGHT_H,
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


def count_wall_crossings(
    sx: float, sy: float, tx: float, ty: float, walls: List[Dict[str, float]]
) -> int:
    """Counts how many concrete wall segments intersect the ray from source to detector."""
    crossings = 0
    for w in walls:
        if segments_intersect(sx, sy, tx, ty, w["x1"], w["y1"], w["x2"], w["y2"]):
            crossings += 1
    return crossings


def expected_sensor_rates_np(
    sx: np.ndarray,
    sy: np.ndarray,
    x: float,
    y: float,
    activity_kbq: float,
    background_cps: float,
) -> np.ndarray:
    """Computes expected sensor count rates for a given source position and activity."""
    dx = sx - x
    dy = sy - y
    r2 = dx * dx + dy * dy + DETECTOR_HEIGHT_H * DETECTOR_HEIGHT_H
    r = np.sqrt(r2)
    atten = np.exp(-AIR_ATTENUATION_MU * r) * (1.0 + 0.004 * r)
    return background_cps + (DETECTOR_EFFICIENCY_ETA * activity_kbq * atten) / r2

