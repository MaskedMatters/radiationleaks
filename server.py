#!/usr/bin/env python3
"""
server.py
Main entrypoint for the RAD-MON-04 Radiation Leak Localization System.

Architecture:
- `backend/config.py`: Facility geometries, physical constants, and paths (`data/`, `captures/`)
- `backend/physics/`: OpenMC photon transport kernel, ray-wall shielding, and script exporter
- `backend/ai/`: `RadLocNeuralNet` (MLP), synapse/pathway attribution analyzer, and persistence
- `backend/api/`: State coordinator, snapshot capture service, CSV/JSON exporters, and HTTP/SSE handler
- `frontend/`: Tabbed SCADA telemetry interface (`index.html`, `css/styles.css`, `js/*.js`)
"""

from backend.api.http_handler import DashboardRequestHandler, run_server
from backend.api.state_manager import BACKEND, TelemetryBackend

__all__ = ["BACKEND", "TelemetryBackend", "DashboardRequestHandler", "run_server"]


if __name__ == "__main__":
    run_server()
