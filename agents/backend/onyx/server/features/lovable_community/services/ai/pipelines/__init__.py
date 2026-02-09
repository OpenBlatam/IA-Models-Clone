"""
Pipeline Module

Provides high-level pipelines for common workflows:
- Training pipelines
- Inference pipelines
- Evaluation pipelines
- End-to-end workflows
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from training_pipeline import TrainingPipeline

__all__ = [
    "TrainingPipeline",
]

