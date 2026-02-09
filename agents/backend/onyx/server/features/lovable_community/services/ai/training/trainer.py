"""
Trainer Module

Complete training pipeline and utilities.
"""

import sys
from pathlib import Path

# Import from parent directory
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from training_utils import (
    Trainer,
    TrainingMetrics
)

__all__ = [
    "Trainer",
    "TrainingMetrics",
]

