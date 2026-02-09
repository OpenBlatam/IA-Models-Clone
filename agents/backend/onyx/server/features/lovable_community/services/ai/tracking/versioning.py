"""
Model Versioning Module

Model versioning and registry.
"""

import sys
from pathlib import Path

# Import from parent directory
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from model_versioning import (
    ModelVersion,
    ModelRegistry,
    compare_model_versions
)

__all__ = [
    "ModelVersion",
    "ModelRegistry",
    "compare_model_versions",
]

