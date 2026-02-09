"""
Core Module - Core Abstractions and Base Classes
================================================

Provides core abstractions and base classes for the entire framework.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn

__all__ = [
    "BaseComponent",
    "ComponentRegistry",
    "Factory",
]
