"""
Baseline methods for comparison.

Implements:
- ANN baseline (Paper [2])
- ePCDNN baseline (Paper [1]) 
- DRL baseline (Paper [3])
- Mechanistic solver baseline
"""

from .ann_pemfc import PEMFC_ANN, PEMFC_ANN_Trainer
from .epcdnn_vrfb import VRFB_ePCDNN, ePCDNN_VRFB_Trainer, PhysicsConstrainedLoss

__all__ = [
    'PEMFC_ANN',
    'PEMFC_ANN_Trainer',
    'VRFB_ePCDNN',
    'ePCDNN_VRFB_Trainer',
    'PhysicsConstrainedLoss',
]
