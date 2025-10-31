from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .model_architectures import (
from .training_pipeline import (
from .loss_functions import (
from .data_processing import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Deep Learning Package for HeyGen AI.

Advanced neural network architectures, training pipelines, loss functions,
and data processing for video generation, text processing, and multimodal
learning following PEP 8 style guidelines.
"""

    ModelArchitectureConfig,
    MultiHeadSelfAttention,
    TransformerBlock,
    VideoGenerationTransformer,
    TextProcessingTransformer,
    MultimodalFusionNetwork,
    ConvolutionalVideoEncoder,
    create_model_architecture
)

    TrainingConfig,
    TrainingMetrics,
    AdvancedTrainingPipeline,
    create_training_pipeline
)

    VideoGenerationLoss,
    TextProcessingLoss,
    MultimodalLoss,
    FocalLoss,
    DiceLoss,
    create_loss_function
)

    VideoDataset,
    TextDataset,
    MultimodalDataset,
    VideoAugmentation,
    DataLoaderFactory,
    create_data_loader
)

__all__ = [
    # Model architectures
    "ModelArchitectureConfig",
    "MultiHeadSelfAttention",
    "TransformerBlock",
    "VideoGenerationTransformer",
    "TextProcessingTransformer",
    "MultimodalFusionNetwork",
    "ConvolutionalVideoEncoder",
    "create_model_architecture",
    
    # Training pipeline
    "TrainingConfig",
    "TrainingMetrics",
    "AdvancedTrainingPipeline",
    "create_training_pipeline",
    
    # Loss functions
    "VideoGenerationLoss",
    "TextProcessingLoss",
    "MultimodalLoss",
    "FocalLoss",
    "DiceLoss",
    "create_loss_function",
    
    # Data processing
    "VideoDataset",
    "TextDataset",
    "MultimodalDataset",
    "VideoAugmentation",
    "DataLoaderFactory",
    "create_data_loader"
] 