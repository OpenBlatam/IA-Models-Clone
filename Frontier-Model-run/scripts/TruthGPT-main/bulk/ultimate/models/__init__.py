"""
Ultimate Models - The most advanced model implementations ever created
Provides cutting-edge model architectures, superior performance, and enterprise-grade features
"""

from .ultimate_transformer import UltimateTransformer, TransformerConfig, AttentionConfig
from .ultimate_llm import UltimateLLM, LLMConfig, GenerationConfig
from .ultimate_diffusion import UltimateDiffusion, DiffusionConfig, SchedulerConfig
from .ultimate_vision import UltimateVision, VisionConfig, BackboneConfig
from .ultimate_multimodal import UltimateMultimodal, MultimodalConfig, FusionConfig

__all__ = [
    # Transformer
    'UltimateTransformer', 'TransformerConfig', 'AttentionConfig',
    
    # LLM
    'UltimateLLM', 'LLMConfig', 'GenerationConfig',
    
    # Diffusion
    'UltimateDiffusion', 'DiffusionConfig', 'SchedulerConfig',
    
    # Vision
    'UltimateVision', 'VisionConfig', 'BackboneConfig',
    
    # Multimodal
    'UltimateMultimodal', 'MultimodalConfig', 'FusionConfig'
]
