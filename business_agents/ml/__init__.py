"""
Machine Learning Package
========================

Machine learning pipeline integration for the Business Agents System.
"""

from .pipeline import MLPipeline, PipelineManager
from .models import MLModel, ModelRegistry, ModelVersion
from .training import TrainingManager, TrainingJob, TrainingMetrics
from .inference import InferenceEngine, PredictionRequest, PredictionResponse
from .features import FeatureStore, FeatureExtractor, FeatureValidator
from .types import (
    ModelType, TrainingStatus, InferenceStatus, FeatureType,
    ModelMetadata, TrainingConfig, InferenceConfig
)

__all__ = [
    "MLPipeline",
    "PipelineManager", 
    "MLModel",
    "ModelRegistry",
    "ModelVersion",
    "TrainingManager",
    "TrainingJob",
    "TrainingMetrics",
    "InferenceEngine",
    "PredictionRequest",
    "PredictionResponse",
    "FeatureStore",
    "FeatureExtractor",
    "FeatureValidator",
    "ModelType",
    "TrainingStatus",
    "InferenceStatus",
    "FeatureType",
    "ModelMetadata",
    "TrainingConfig",
    "InferenceConfig"
]
