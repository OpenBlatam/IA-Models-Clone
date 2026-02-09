"""
Evaluation Module - Metrics and Validation
===========================================

This module provides evaluation functionality:
- Classification metrics
- Regression metrics
- Custom metrics
- Model evaluation utilities
"""

from typing import Dict, Any, List, Optional
import torch
import numpy as np

__all__ = [
    "Metrics",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "evaluate_model",
]



