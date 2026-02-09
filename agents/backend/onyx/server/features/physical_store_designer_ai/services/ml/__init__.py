"""
Machine Learning Services Module

Services for ML operations, training, and model management.
"""

# Training services
from ..advanced_validation_service import AdvancedValidationService
from ..advanced_augmentation_service import AdvancedAugmentationService
from ..custom_loss_service import CustomLossService
from ..advanced_optimizers_service import AdvancedOptimizersService
from ..lr_finder_service import LRFinderService
from ..model_debugging_service import ModelDebuggingService

# Model services
from ..transfer_learning_service import TransferLearningService
from ..multitask_learning_service import MultiTaskLearningService
from ..continual_learning_service import ContinualLearningService
from ..nas_service import NASService
from ..automl_service import AutoMLService
from ..ensembling_service import EnsemblingService

# Advanced ML services
from ..advanced_transformers_service import AdvancedTransformersService
from ..advanced_diffusion_service import AdvancedDiffusionService
from ..prompt_engineering_service import PromptEngineeringService
from ..model_optimization_service import ModelOptimizationService
from ..advanced_quantization_service import AdvancedQuantizationService
from ..distributed_training_service import DistributedTrainingService

# Production ML services
from ..api_serving_service import APIServingService
from ..ab_testing_ml_service import ABTestingMLService
from ..model_rollback_service import ModelRollbackService
from ..benchmarking_service import BenchmarkingService
from ..auto_scaling_service import AutoScalingService
from ..health_check_ml_service import HealthCheckMLService

# Utility ML services
from ..visualization_service import VisualizationService
from ..model_comparison_service import ModelComparisonService
from ..batch_processing_service import BatchProcessingService
from ..memory_optimization_service import MemoryOptimizationService
from ..model_conversion_service import ModelConversionService
from ..advanced_metrics_service import AdvancedMetricsService

__all__ = [
    # Training
    "AdvancedValidationService",
    "AdvancedAugmentationService",
    "CustomLossService",
    "AdvancedOptimizersService",
    "LRFinderService",
    "ModelDebuggingService",
    # Models
    "TransferLearningService",
    "MultiTaskLearningService",
    "ContinualLearningService",
    "NASService",
    "AutoMLService",
    "EnsemblingService",
    # Advanced ML
    "AdvancedTransformersService",
    "AdvancedDiffusionService",
    "PromptEngineeringService",
    "ModelOptimizationService",
    "AdvancedQuantizationService",
    "DistributedTrainingService",
    # Production
    "APIServingService",
    "ABTestingMLService",
    "ModelRollbackService",
    "BenchmarkingService",
    "AutoScalingService",
    "HealthCheckMLService",
    # Utilities
    "VisualizationService",
    "ModelComparisonService",
    "BatchProcessingService",
    "MemoryOptimizationService",
    "ModelConversionService",
    "AdvancedMetricsService",
]

