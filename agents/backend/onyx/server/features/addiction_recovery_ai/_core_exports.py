"""
Core component exports for Addiction Recovery AI
"""

# Core components
from .core.addiction_analyzer import AddictionAnalyzer
from .core.recovery_planner import RecoveryPlanner
from .core.progress_tracker import ProgressTracker
from .core.relapse_prevention import RelapsePrevention

# Modular Architecture - Base Classes
from .core.base.base_model import (
    BaseModel,
    BasePredictor,
    BaseGenerator,
    BaseAnalyzer
)
from .core.base.base_trainer import (
    BaseTrainer,
    BaseEvaluator
)

__all__ = [
    "AddictionAnalyzer",
    "RecoveryPlanner",
    "ProgressTracker",
    "RelapsePrevention",
    "BaseModel",
    "BasePredictor",
    "BaseGenerator",
    "BaseAnalyzer",
    "BaseTrainer",
    "BaseEvaluator",
]

