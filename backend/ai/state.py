"""
backend/ai/state.py
Typed dataclass holding per-sector neural network weights, baseline reference weights,
Adam optimizer momentum/variance arrays, and epoch learning curves.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np


@dataclass
class SectorModelState:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    W3: np.ndarray
    b3: np.ndarray
    W1_init: np.ndarray = field(default_factory=lambda: np.zeros((33, 128)))
    W2_init: np.ndarray = field(default_factory=lambda: np.zeros((128, 64)))
    W3_init: np.ndarray = field(default_factory=lambda: np.zeros((64, 3)))
    last_dW1: np.ndarray = field(default_factory=lambda: np.zeros((33, 128)))
    last_dW2: np.ndarray = field(default_factory=lambda: np.zeros((128, 64)))
    last_dW3: np.ndarray = field(default_factory=lambda: np.zeros((64, 3)))
    m_adam: List[np.ndarray] = field(default_factory=list)
    v_adam: List[np.ndarray] = field(default_factory=list)
    adam_step: int = 0
    trained_epochs: int = 0
    samples_seen: int = 0
    last_loss: float = 0.0
    val_mae_m: float = 0.0
    epoch_curve: List[Dict[str, Any]] = field(default_factory=list)

