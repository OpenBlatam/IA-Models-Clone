"""
Detectors - Detectores de framework y tipo de modelo
=====================================================

Funciones para detectar automáticamente el framework, tipo de modelo,
y técnicas de Deep Learning desde keywords del proyecto.
Optimizado siguiendo mejores prácticas de Python y FastAPI.

Sigue mejores prácticas de Deep Learning y soporta:
- Transformers (BERT, GPT, T5, etc.)
- Diffusion Models (Stable Diffusion, DDPM, DDIM, etc.)
- Fine-tuning techniques (LoRA, P-tuning, etc.)
- Training optimizations (mixed precision, gradient accumulation, etc.)
"""

import logging
from enum import Enum
from typing import Dict, Any, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Tipos de modelos soportados."""
    TRANSFORMER = "transformer"
    DIFFUSION = "diffusion"
    CNN = "cnn"
    RNN = "rnn"
    BASE = "base"


class FrameworkType(str, Enum):
    """Frameworks soportados."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"


class FineTuningTechnique(str, Enum):
    """Técnicas de fine-tuning soportadas."""
    LORA = "lora"
    P_TUNING = "p_tuning"
    FULL_FINETUNING = "full_finetuning"
    ADAPTERS = "adapters"
    NONE = "none"


@dataclass(frozen=True)
class DeepLearningFeatures:
    """
    Características de Deep Learning detectadas.
    
    Agrupa todas las características detectadas del proyecto.
    Inmutable para mejor seguridad.
    """
    framework: FrameworkType
    model_type: ModelType
    fine_tuning_technique: FineTuningTechnique = FineTuningTechnique.NONE
    requires_mixed_precision: bool = False
    requires_gradient_accumulation: bool = False
    requires_multi_gpu: bool = False
    requires_experiment_tracking: bool = False
    requires_gradio: bool = False
    requires_training: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario.
        
        Returns:
            Diccionario con características
        """
        return {
            "framework": self.framework.value,
            "model_type": self.model_type.value,
            "fine_tuning_technique": self.fine_tuning_technique.value,
            "requires_mixed_precision": self.requires_mixed_precision,
            "requires_gradient_accumulation": self.requires_gradient_accumulation,
            "requires_multi_gpu": self.requires_multi_gpu,
            "requires_experiment_tracking": self.requires_experiment_tracking,
            "requires_gradio": self.requires_gradio,
            "requires_training": self.requires_training,
        }


_TENSORFLOW_KEYWORDS: Set[str] = {
    "tensorflow", "tf", "keras", "tf.keras", "tensorflow.keras"
}

_JAX_KEYWORDS: Set[str] = {
    "jax", "flax", "jax.numpy", "flax.linen"
}

_TRANSFORMER_KEYWORDS: Set[str] = {
    "transformer", "bert", "gpt", "llm", "attention", "t5", "roberta",
    "distilbert", "albert", "electra", "xlnet", "bart", "deberta",
    "multi-head attention", "self-attention", "encoder", "decoder"
}

_DIFFUSION_KEYWORDS: Set[str] = {
    "diffusion", "stable diffusion", "latent", "ddpm", "ddim", "scheduler",
    "noise scheduler", "unet", "vae", "stable-diffusion", "sd-xl",
    "latent diffusion", "text-to-image", "image generation"
}

_CNN_KEYWORDS: Set[str] = {
    "cnn", "conv", "convolutional", "image", "vision", "resnet", "vgg",
    "efficientnet", "mobilenet", "densenet", "inception", "alexnet",
    "computer vision", "image classification", "object detection"
}

_RNN_KEYWORDS: Set[str] = {
    "rnn", "lstm", "gru", "sequence", "recurrent", "temporal",
    "time series", "sequential", "bidirectional", "attention lstm"
}

_LORA_KEYWORDS: Set[str] = {
    "lora", "low-rank", "low rank adaptation"
}

_P_TUNING_KEYWORDS: Set[str] = {
    "p-tuning", "p tuning", "prompt tuning"
}

_ADAPTERS_KEYWORDS: Set[str] = {
    "adapter", "adapters", "adapter layers"
}

_MIXED_PRECISION_KEYWORDS: Set[str] = {
    "mixed precision", "fp16", "bf16", "half precision",
    "torch.cuda.amp", "autocast"
}

_GRADIENT_ACCUMULATION_KEYWORDS: Set[str] = {
    "gradient accumulation", "accumulate gradients",
    "gradient_accumulation_steps"
}

_MULTI_GPU_KEYWORDS: Set[str] = {
    "multi-gpu", "multi gpu", "distributed", "dataparallel",
    "distributeddataparallel", "ddp", "dp"
}

_EXPERIMENT_TRACKING_KEYWORDS: Set[str] = {
    "wandb", "tensorboard", "mlflow", "experiment tracking",
    "tracking", "logging metrics"
}

_GRADIO_KEYWORDS: Set[str] = {
    "gradio", "interface", "demo", "ui"
}


def _normalize_keywords(keywords: Dict[str, Any]) -> str:
    """
    Normalizar keywords a string para búsqueda (función pura).
    
    Args:
        keywords: Keywords extraídos
        
    Returns:
        String normalizado con todos los keywords
    """
    keywords_list = keywords.get("keywords", [])
    keywords_str = " ".join(str(kw).lower() for kw in keywords_list)
    
    if "description" in keywords:
        keywords_str += " " + str(keywords["description"]).lower()
    if "title" in keywords:
        keywords_str += " " + str(keywords["title"]).lower()
    
    return keywords_str


def _contains_any_keyword(text: str, keywords: Set[str]) -> bool:
    """
    Verifica si el texto contiene alguna keyword (función pura).
    
    Args:
        text: Texto a buscar
        keywords: Set de keywords
        
    Returns:
        True si contiene alguna keyword, False en caso contrario
    """
    return any(kw in text for kw in keywords)


def detect_framework(keywords: Dict[str, Any]) -> FrameworkType:
    """
    Detectar framework desde keywords.
    
    Args:
        keywords: Keywords extraídos del proyecto
        
    Returns:
        Framework detectado
    """
    keywords_str = _normalize_keywords(keywords)
    
    if _contains_any_keyword(keywords_str, _TENSORFLOW_KEYWORDS):
        logger.debug("Detected TensorFlow framework")
        return FrameworkType.TENSORFLOW
    
    if _contains_any_keyword(keywords_str, _JAX_KEYWORDS):
        logger.debug("Detected JAX framework")
        return FrameworkType.JAX
    
    logger.debug("Defaulting to PyTorch framework")
    return FrameworkType.PYTORCH


def detect_model_type(keywords: Dict[str, Any]) -> ModelType:
    """
    Detectar tipo de modelo desde keywords.
    
    Soporta detección de:
    - Transformers: BERT, GPT, T5, RoBERTa, etc.
    - Diffusion: Stable Diffusion, DDPM, DDIM, etc.
    - CNN: ResNet, VGG, EfficientNet, etc.
    - RNN: LSTM, GRU, etc.
    
    Args:
        keywords: Keywords extraídos del proyecto
        
    Returns:
        Tipo de modelo detectado
    """
    keywords_str = _normalize_keywords(keywords)
    
    if _contains_any_keyword(keywords_str, _TRANSFORMER_KEYWORDS):
        logger.debug("Detected Transformer model type")
        return ModelType.TRANSFORMER
    
    if _contains_any_keyword(keywords_str, _DIFFUSION_KEYWORDS):
        logger.debug("Detected Diffusion model type")
        return ModelType.DIFFUSION
    
    if _contains_any_keyword(keywords_str, _CNN_KEYWORDS):
        logger.debug("Detected CNN model type")
        return ModelType.CNN
    
    if _contains_any_keyword(keywords_str, _RNN_KEYWORDS):
        logger.debug("Detected RNN model type")
        return ModelType.RNN
    
    logger.debug("Defaulting to base model type")
    return ModelType.BASE


def detect_fine_tuning_technique(keywords: Dict[str, Any]) -> FineTuningTechnique:
    """
    Detectar técnica de fine-tuning desde keywords.
    
    Soporta:
    - LoRA (Low-Rank Adaptation)
    - P-tuning
    - Full fine-tuning
    - Adapters
    
    Args:
        keywords: Keywords extraídos del proyecto
        
    Returns:
        Técnica de fine-tuning detectada
    """
    keywords_str = _normalize_keywords(keywords)
    
    if _contains_any_keyword(keywords_str, _LORA_KEYWORDS):
        logger.debug("Detected LoRA fine-tuning technique")
        return FineTuningTechnique.LORA
    
    if _contains_any_keyword(keywords_str, _P_TUNING_KEYWORDS):
        logger.debug("Detected P-tuning fine-tuning technique")
        return FineTuningTechnique.P_TUNING
    
    if _contains_any_keyword(keywords_str, _ADAPTERS_KEYWORDS):
        logger.debug("Detected Adapters fine-tuning technique")
        return FineTuningTechnique.ADAPTERS
    
    logger.debug("Defaulting to full fine-tuning")
    return FineTuningTechnique.FULL_FINETUNING


def detect_deep_learning_features(keywords: Dict[str, Any]) -> DeepLearningFeatures:
    """
    Detectar todas las características de Deep Learning.
    
    Detecta:
    - Framework (PyTorch, TensorFlow, JAX)
    - Tipo de modelo (Transformer, Diffusion, CNN, RNN)
    - Técnica de fine-tuning (LoRA, P-tuning, etc.)
    - Optimizaciones requeridas (mixed precision, gradient accumulation, etc.)
    
    Args:
        keywords: Keywords extraídos del proyecto
        
    Returns:
        Características detectadas
    """
    framework = detect_framework(keywords)
    model_type = detect_model_type(keywords)
    fine_tuning_technique = detect_fine_tuning_technique(keywords)
    
    keywords_str = _normalize_keywords(keywords)
    
    requires_mixed_precision = _contains_any_keyword(
        keywords_str, _MIXED_PRECISION_KEYWORDS
    )
    
    requires_gradient_accumulation = _contains_any_keyword(
        keywords_str, _GRADIENT_ACCUMULATION_KEYWORDS
    )
    
    requires_multi_gpu = _contains_any_keyword(
        keywords_str, _MULTI_GPU_KEYWORDS
    )
    
    requires_experiment_tracking = _contains_any_keyword(
        keywords_str, _EXPERIMENT_TRACKING_KEYWORDS
    )
    
    requires_gradio = (
        keywords.get("requires_gradio", False) or
        _contains_any_keyword(keywords_str, _GRADIO_KEYWORDS)
    )
    
    requires_training = keywords.get("requires_training", True)
    
    features = DeepLearningFeatures(
        framework=framework,
        model_type=model_type,
        fine_tuning_technique=fine_tuning_technique,
        requires_mixed_precision=requires_mixed_precision,
        requires_gradient_accumulation=requires_gradient_accumulation,
        requires_multi_gpu=requires_multi_gpu,
        requires_experiment_tracking=requires_experiment_tracking,
        requires_gradio=requires_gradio,
        requires_training=requires_training
    )
    
    logger.info(f"Detected Deep Learning features: {features.to_dict()}")
    return features
