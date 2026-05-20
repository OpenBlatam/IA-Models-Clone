"""
TruthGPT Core - Unified Optimization Framework
A clean, modular architecture for neural network optimization
"""

from .optimization import OptimizationEngine, OptimizationConfig
from .models import ModelManager, ModelConfig
from .training import TrainingManager, TrainingConfig
from .inference import InferenceEngine, InferenceConfig
from .monitoring import MonitoringSystem, MetricsCollector

__version__ = "2.0.0"
__all__ = [
    "OptimizationEngine",
    "OptimizationConfig", 
    "ModelManager",
    "ModelConfig",
    "TrainingManager", 
    "TrainingConfig",
    "InferenceEngine",
    "InferenceConfig",
    "MonitoringSystem",
    "MetricsCollector"
]

