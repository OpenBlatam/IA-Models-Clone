"""
TruthGPT Core - Unified Optimization Framework
A clean, modular architecture for neural network optimization
"""

from .optimization import OptimizationEngine, OptimizationConfig, OptimizationLevel
from .models import ModelManager, ModelConfig, ModelType
from .training import TrainingManager, TrainingConfig
from .inference import InferenceEngine, InferenceConfig
from .monitoring import MonitoringSystem, MetricsCollector, SystemMetrics, ModelMetrics, TrainingMetrics

__version__ = "2.0.0"
__all__ = [
    "OptimizationEngine",
    "OptimizationConfig",
    "OptimizationLevel",
    "ModelManager",
    "ModelConfig",
    "ModelType",
    "TrainingManager", 
    "TrainingConfig",
    "InferenceEngine",
    "InferenceConfig",
    "MonitoringSystem",
    "MetricsCollector",
    "SystemMetrics",
    "ModelMetrics",
    "TrainingMetrics"
]

