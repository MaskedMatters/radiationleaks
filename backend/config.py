"""
backend/config.py
Central configuration, physical constants, facility sector geometries, and file paths
for the RAD-MON-04 Radiation Leak Localization System.
"""

import os
from typing import Any, Dict, List


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DATA_DIR = os.path.join(BASE_DIR, "data")
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CAPTURES_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(DATA_DIR, "run_logs.jsonl")
TRAIN_LOG_FILE_PATH = os.path.join(DATA_DIR, "ai_training_samples.jsonl")
AI_CHECKPOINT_PATH = os.path.join(DATA_DIR, "ai_checkpoint.json")
SESSION_STATE_PATH = os.path.join(DATA_DIR, "session_state.json")

# Physical & Domain Constants (Cs-137 662 keV Gamma Transport)
DOMAIN_SIZE = 20.0
DETECTOR_HEIGHT_H = 1.0
DETECTOR_EFFICIENCY_ETA = 3.6
AIR_ATTENUATION_MU = 0.010
DWELL_TIME_SEC = 1.0

# Concrete Shielding Wall Constants (22 cm Reinforced Concrete Partition Walls)
WALL_THICKNESS_M = 0.22
WALL_LINEAR_ATTEN_MU = 6.80  # Effective attenuation coeff (m^-1) for 662 keV in concrete with buildup
WALL_SCATTER_FLOOR = 0.025   # Residual room-scatter / skyshine fraction around wall baffles


def generate_grid_sensors(min_coord: float, max_coord: float, n_cols: int) -> List[Dict[str, Any]]:
    """Generates an evenly spaced N x N sensor array across the 20m x 20m domain."""
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
