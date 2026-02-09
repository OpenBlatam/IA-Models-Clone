"""
Compression Module

Model compression: pruning and other techniques.
"""

import sys
from pathlib import Path

# Import from parent directory
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from model_optimization import ModelPruner

__all__ = [
    "ModelPruner",
]

