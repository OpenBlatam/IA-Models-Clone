"""
ML Module - Core Machine Learning Components
Reorganized for better structure

This module provides a clean interface to all ML components.
Use: from ml import ViTSkinAnalyzer, Trainer, SkinDataset
"""

# Models - Import from parent directory
import sys
from pathlib import Path

# Add parent to path for imports
_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

try:
    from models.base import BaseModel, SkinAnalysisModel, ModelFactory, ModelConfig
    from models.pytorch_models import (
        SkinAnalysisCNN,
        SkinQualityRegressor,
        ConditionClassifier,
        EnhancedSkinAnalyzer
    )
    from models.vision_transformers import (
        VisionTransformer,
        ViTSkinAnalyzer,
        LoRAViT
    )
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    import logging
    logging.warning(f"Models not available: {e}")

try:
    # Training
    from training.trainer import Trainer, create_data_loaders
    from training.losses import MultiTaskLoss, ConditionLoss, MetricLoss, FocalLoss, DiceLoss
    from training.optimizers import get_optimizer, get_scheduler, get_optimizer_and_scheduler
    from training.metrics import MetricCalculator, ClassificationMetrics, RegressionMetrics
    from training.distributed import (
        setup_distributed,
        wrap_model_for_distributed,
        get_world_size,
        get_rank,
        is_main_process
    )
    TRAINING_AVAILABLE = True
except ImportError as e:
    TRAINING_AVAILABLE = False
    import logging
    logging.warning(f"Training not available: {e}")

try:
    # Data
    from data.datasets import SkinDataset, SkinVideoDataset, MultiTaskDataset
    from data.transforms import get_train_transforms, get_val_transforms, get_test_transforms
    from data.preprocessing import ImagePreprocessor, VideoPreprocessor
    DATA_AVAILABLE = True
except ImportError as e:
    DATA_AVAILABLE = False
    import logging
    logging.warning(f"Data not available: {e}")

try:
    # Optimization (optional)
    from utils.optimization import (
        compile_model,
        optimize_for_inference,
        quantize_model,
        FastInferenceEngine
    )
    from utils.advanced_optimization import (
        enable_gradient_checkpointing,
        enable_flash_attention,
        MemoryEfficientModel,
        SmartBatchProcessor
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

__all__ = []

# Add models if available
if MODELS_AVAILABLE:
    __all__.extend([
        'BaseModel',
        'SkinAnalysisModel',
        'ModelFactory',
        'ModelConfig',
        'SkinAnalysisCNN',
        'SkinQualityRegressor',
        'ConditionClassifier',
        'EnhancedSkinAnalyzer',
        'VisionTransformer',
        'ViTSkinAnalyzer',
        'LoRAViT',
    ])

# Add training if available
if TRAINING_AVAILABLE:
    __all__.extend([
        'Trainer',
        'RefactoredTrainer',
        'TrainingPipeline',
        'create_data_loaders',
        'MultiTaskLoss',
        'ConditionLoss',
        'MetricLoss',
        'FocalLoss',
        'DiceLoss',
        'get_optimizer',
        'get_scheduler',
        'get_optimizer_and_scheduler',
        'MetricCalculator',
        'ClassificationMetrics',
        'RegressionMetrics',
        'TrainingCallback',
        'EarlyStoppingCallback',
        'ModelCheckpointCallback',
        'LearningRateSchedulerCallback',
        'MetricsLoggingCallback',
        'setup_distributed',
        'wrap_model_for_distributed',
        'get_world_size',
        'get_rank',
        'is_main_process',
    ])

# Add data if available
if DATA_AVAILABLE:
    __all__.extend([
        'SkinDataset',
        'SkinVideoDataset',
        'MultiTaskDataset',
        'get_train_transforms',
        'get_val_transforms',
        'get_test_transforms',
        'ImagePreprocessor',
        'VideoPreprocessor',
    ])

# Add optimization if available
if OPTIMIZATION_AVAILABLE:
    __all__.extend([
        'compile_model',
        'optimize_for_inference',
        'quantize_model',
        'FastInferenceEngine',
        'enable_gradient_checkpointing',
        'enable_flash_attention',
        'MemoryEfficientModel',
        'SmartBatchProcessor'
    ])

