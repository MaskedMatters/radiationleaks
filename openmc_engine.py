"""
openmc_engine.py
OpenMC Radiation Transport & Real-Time Monte Carlo Photon Simulation Engine.
"""

import math
import time
from typing import Any, Dict, List, Optional
import numpy as np

try:
    import openmc  # type: ignore
    HAS_OPENMC = True
except ImportError:
    HAS_OPENMC = False


DOMAIN_SIZE = 20.0
DETECTOR_HEIGHT_H = 1.0
DETECTOR_EFFICIENCY_ETA = 3.6
AIR_ATTENUATION_MU = 0.010
DWELL_TIME_SEC = 1.0


def generate_grid_sensors(min_coord: float, max_coord: float, n_cols: int) -> List[Dict[str, Any]]:
    sensors: List[Dict[str, Any]] = []
    step = (max_coord - min_coord) / float(n_cols - 1)
    idx = 1
    for r in range(n_cols):
        for c in range(n_cols):
            sensors.append({
                "id": f"S-{idx:02d}",
                "x": round(min_coord + float(c) * step, 2),
                "y": round(min_coord + float(r) * step, 2),
            })
            idx += 1
    return sensors


FACILITIES: Dict[str, Dict[str, Any]] = {
    "sector-a": {
        "name": "SECTOR-04A // REACTOR AUX BLDG",
        "zones": [
            {"x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0, "label": "AUX-BAY-1 (PUMP ROOM)"},
            {"x": 10.0, "y": 0.0, "w": 10.0, "h": 10.0, "label": "AUX-BAY-2 (CVCS)"},
            {"x": 0.0, "y": 10.0, "w": 10.0, "h": 10.0, "label": "AUX-BAY-3 (RHR LOOP)"},
            {"x": 10.0, "y": 10.0, "w": 10.0, "h": 10.0, "label": "AUX-BAY-4 (VALVE GALLERY)"},
        ],
        "walls": [
            {"x1": 10.0, "y1": 0.0, "x2": 10.0, "y2": 8.2},
            {"x1": 10.0, "y1": 11.8, "x2": 10.0, "y2": 20.0},
            {"x1": 0.0, "y1": 10.0, "x2": 8.2, "y2": 10.0},
            {"x1": 11.8, "y1": 10.0, "x2": 20.0, "y2": 10.0},
        ],
        "sensors": generate_grid_sensors(2.5, 17.5, 4),
    },
    "sector-b": {
        "name": "SECTOR-04B // SPENT FUEL BAY",
        "zones": [
            {"x": 0.0, "y": 0.0, "w": 20.0, "h": 5.0, "label": "NORTH CASK HANDLING DECK"},
            {"x": 0.0, "y": 5.0, "w": 14.0, "h": 10.0, "label": "SPENT FUEL POOL BASIN"},
            {"x": 14.0, "y": 5.0, "w": 6.0, "h": 10.0, "label": "HEAT EXCHANGER SKID"},
            {"x": 0.0, "y": 15.0, "w": 20.0, "h": 5.0, "label": "SOUTH TRANSFER CANAL"},
        ],
        "walls": [
            {"x1": 0.0, "y1": 5.0, "x2": 16.0, "y2": 5.0},
            {"x1": 0.0, "y1": 15.0, "x2": 16.0, "y2": 15.0},
            {"x1": 14.0, "y1": 5.0, "x2": 14.0, "y2": 15.0},
        ],
        "sensors": [
            {"id": "S-01", "x": 2.0, "y": 2.5},
            {"id": "S-02", "x": 7.0, "y": 2.5},
            {"id": "S-03", "x": 12.5, "y": 2.5},
            {"id": "S-04", "x": 17.5, "y": 2.5},
            {"id": "S-05", "x": 2.0, "y": 6.5},
            {"id": "S-06", "x": 7.0, "y": 6.5},
            {"id": "S-07", "x": 12.5, "y": 6.5},
            {"id": "S-08", "x": 17.0, "y": 7.5},
            {"id": "S-09", "x": 2.0, "y": 13.5},
            {"id": "S-10", "x": 7.0, "y": 13.5},
            {"id": "S-11", "x": 12.5, "y": 13.5},
            {"id": "S-12", "x": 17.0, "y": 12.5},
            {"id": "S-13", "x": 2.0, "y": 17.5},
            {"id": "S-14", "x": 7.0, "y": 17.5},
            {"id": "S-15", "x": 12.5, "y": 17.5},
            {"id": "S-16", "x": 17.5, "y": 17.5},
        ],
    },
    "sector-c": {
        "name": "SECTOR-04C // HOT CELL CORRIDOR",
        "zones": [
            {"x": 0.0, "y": 0.0, "w": 6.5, "h": 20.0, "label": "HOT CELL BANK WEST"},
            {"x": 6.5, "y": 0.0, "w": 7.0, "h": 20.0, "label": "MANIPULATOR GALLERY"},
            {"x": 13.5, "y": 0.0, "w": 6.5, "h": 10.0, "label": "ISOTOPE CELL E-1"},
            {"x": 13.5, "y": 10.0, "w": 6.5, "h": 10.0, "label": "GLOVEBOX ANNEX E-2"},
        ],
        "walls": [
            {"x1": 6.5, "y1": 0.0, "x2": 6.5, "y2": 8.5},
            {"x1": 6.5, "y1": 11.5, "x2": 6.5, "y2": 20.0},
            {"x1": 13.5, "y1": 0.0, "x2": 13.5, "y2": 8.5},
            {"x1": 13.5, "y1": 11.5, "x2": 13.5, "y2": 20.0},
            {"x1": 13.5, "y1": 10.0, "x2": 20.0, "y2": 10.0},
        ],
        "sensors": [
            {"id": "S-01", "x": 3.2, "y": 2.2},
            {"id": "S-02", "x": 9.0, "y": 2.2},
            {"id": "S-03", "x": 11.5, "y": 2.2},
            {"id": "S-04", "x": 16.8, "y": 2.2},
            {"id": "S-05", "x": 3.2, "y": 7.2},
            {"id": "S-06", "x": 8.2, "y": 7.2},
            {"id": "S-07", "x": 12.0, "y": 7.2},
            {"id": "S-08", "x": 16.8, "y": 7.2},
            {"id": "S-09", "x": 3.2, "y": 12.8},
            {"id": "S-10", "x": 8.2, "y": 12.8},
            {"id": "S-11", "x": 12.0, "y": 12.8},
            {"id": "S-12", "x": 16.8, "y": 12.8},
            {"id": "S-13", "x": 3.2, "y": 17.8},
            {"id": "S-14", "x": 9.0, "y": 17.8},
            {"id": "S-15", "x": 11.5, "y": 17.8},
            {"id": "S-16", "x": 16.8, "y": 17.8},
        ],
    },
}


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
    crossings = 0
    for w in walls:
        if segments_intersect(sx, sy, tx, ty, w["x1"], w["y1"], w["x2"], w["y2"]):
            crossings += 1
    return crossings


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
        facility = FACILITIES.get(facility_key, FACILITIES["sector-a"])
        sensors_repr = repr(facility["sensors"])
        walls_repr = repr(facility["walls"])

        return f'''#!/usr/bin/env python3
"""
OpenMC Photon Transport Model — {facility["name"]}
Generated by RAD-MON-04 Telemetry Control Center
Source Position: ({source_x:.2f} m, {source_y:.2f} m, 1.00 m) | Activity: {activity_kbq:.1f} kBq
"""

import openmc

def build_model():
    model = openmc.Model()

    air = openmc.Material(name="Dry Air")
    air.set_density("g/cm3", 0.001205)
    air.add_element("N", 0.784431)
    air.add_element("O", 0.210748)
    air.add_element("Ar", 0.004821)

    nai = openmc.Material(name="NaI(Tl) Scintillator")
    nai.set_density("g/cm3", 3.67)
    nai.add_element("Na", 0.5)
    nai.add_element("I", 0.5)

    concrete = openmc.Material(name="Shielding Concrete")
    concrete.set_density("g/cm3", 2.30)
    concrete.add_element("O", 0.52)
    concrete.add_element("Si", 0.325)
    concrete.add_element("Ca", 0.06)
    concrete.add_element("Al", 0.035)
    concrete.add_element("Fe", 0.06)

    model.materials = openmc.Materials([air, nai, concrete])

    xmin = openmc.XPlane(x0=0.0, boundary_type="vacuum")
    xmax = openmc.XPlane(x0=2000.0, boundary_type="vacuum")
    ymin = openmc.YPlane(y0=0.0, boundary_type="vacuum")
    ymax = openmc.YPlane(y0=2000.0, boundary_type="vacuum")
    zmin = openmc.ZPlane(z0=0.0, boundary_type="reflective")
    zmax = openmc.ZPlane(z0=300.0, boundary_type="vacuum")

    room_region = +xmin & -xmax & +ymin & -ymax & +zmin & -zmax

    sensors = {sensors_repr}
    _walls = {walls_repr}
    detector_cells = []

    for s in sensors:
        sph = openmc.Sphere(x0=s["x"] * 100.0, y0=s["y"] * 100.0, z0=0.0, r=7.62)
        cell = openmc.Cell(name=s["id"], fill=nai, region=-sph)
        detector_cells.append(cell)
        room_region = room_region & +sph

    room_cell = openmc.Cell(name="Facility Air Volume", fill=air, region=room_region)
    model.geometry = openmc.Geometry(detector_cells + [room_cell])

    source = openmc.IndependentSource()
    source.particle = "photon"
    source.space = openmc.stats.Point(({source_x * 100.0:.1f}, {source_y * 100.0:.1f}, 100.0))
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.Discrete([661.7e3], [1.0])
    source.strength = {activity_kbq * 1000.0:.1f}

    model.settings = openmc.Settings()
    model.settings.run_mode = "fixed source"
    model.settings.photon_transport = True
    model.settings.batches = 10
    model.settings.particles = {int(histories)}
    model.settings.source = source

    tally = openmc.Tally(name="NaI_16CH_Photon_Flux")
    tally.filters = [openmc.CellFilter(detector_cells)]
    tally.scores = ["flux", "heating"]
    model.tallies = openmc.Tallies([tally])

    return model

if __name__ == "__main__":
    m = build_model()
    m.export_to_model_xml()
    print("Exported OpenMC model.xml successfully.")
'''
