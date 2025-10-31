from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .models.transformer_models import (
from .models.diffusion_models import (
from .models.llm_models import (
from .models.vision_models import (
from .training.trainer import (
from .data.data_loader import (
from .inference.inference_engine import (
from .utils.model_utils import (
from .gradio_interfaces import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Advanced AI Models Module - Deep Learning, Transformers, Diffusion Models & LLMs
Optimized with latest PyTorch, Transformers, Diffusers, and Gradio libraries.
"""

    AdvancedTransformerModel,
    MultiModalTransformer,
    CustomAttentionMechanism,
    PositionalEncoding
)

    StableDiffusionPipeline,
    CustomDiffusionModel,
    DiffusionScheduler,
    TextToImagePipeline
)

    AdvancedLLMModel,
    LoRAFineTuner,
    CustomTokenizer,
    LLMInferenceEngine
)

    VisionTransformer,
    ImageClassificationModel,
    ObjectDetectionModel,
    SegmentationModel
)

    AdvancedTrainer,
    MixedPrecisionTrainer,
    DistributedTrainer,
    CustomLossFunctions
)

    AdvancedDataLoader,
    MultiModalDataset,
    CustomTransforms,
    DataAugmentation
)

    ModelInferenceEngine,
    BatchInference,
    RealTimeInference,
    ModelOptimization
)

    ModelCheckpointing,
    ExperimentTracking,
    PerformanceProfiling,
    ModelEvaluation
)

    create_model_demo,
    create_inference_interface,
    create_training_interface,
    create_evaluation_dashboard
)

__version__ = "2.0.0"
__author__ = "Advanced AI Models Team"

__all__ = [
    # Models
    "AdvancedTransformerModel",
    "MultiModalTransformer", 
    "CustomAttentionMechanism",
    "PositionalEncoding",
    "StableDiffusionPipeline",
    "CustomDiffusionModel",
    "DiffusionScheduler",
    "TextToImagePipeline",
    "AdvancedLLMModel",
    "LoRAFineTuner",
    "CustomTokenizer",
    "LLMInferenceEngine",
    "VisionTransformer",
    "ImageClassificationModel",
    "ObjectDetectionModel",
    "SegmentationModel",
    
    # Training
    "AdvancedTrainer",
    "MixedPrecisionTrainer",
    "DistributedTrainer", 
    "CustomLossFunctions",
    
    # Data
    "AdvancedDataLoader",
    "MultiModalDataset",
    "CustomTransforms",
    "DataAugmentation",
    
    # Inference
    "ModelInferenceEngine",
    "BatchInference",
    "RealTimeInference",
    "ModelOptimization",
    
    # Utils
    "ModelCheckpointing",
    "ExperimentTracking",
    "PerformanceProfiling",
    "ModelEvaluation",
    
    # Gradio
    "create_model_demo",
    "create_inference_interface",
    "create_training_interface",
    "create_evaluation_dashboard"
] 