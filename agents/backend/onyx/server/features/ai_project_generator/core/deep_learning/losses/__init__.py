"""
Loss Functions Module - Custom Loss Functions
==============================================

Provides custom loss functions for different tasks:
- Classification losses
- Regression losses
- Custom losses
- Loss combinations
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from .custom_losses import (
    FocalLoss,
    LabelSmoothingLoss,
    DiceLoss,
    CombinedLoss,
    create_loss
)

__all__ = [
    "FocalLoss",
    "LabelSmoothingLoss",
    "DiceLoss",
    "CombinedLoss",
    "create_loss",
]

