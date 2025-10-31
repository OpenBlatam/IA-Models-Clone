"""
Ultra Models - The most advanced model implementations ever created
Provides extreme performance, maximum efficiency, and cutting-edge architectures
"""

from .ultra_transformer import UltraTransformer, UltraTransformerConfig, UltraMultiHeadAttention
from .ultra_llm import UltraLLM, UltraLLMConfig, UltraGPT, UltraBERT, UltraT5
from .ultra_diffusion import UltraDiffusion, UltraDiffusionConfig, UltraStableDiffusion
from .ultra_vision import UltraVision, UltraVisionConfig, UltraResNet, UltraViT

__all__ = [
    # Transformer
    'UltraTransformer', 'UltraTransformerConfig', 'UltraMultiHeadAttention',
    
    # LLM
    'UltraLLM', 'UltraLLMConfig', 'UltraGPT', 'UltraBERT', 'UltraT5',
    
    # Diffusion
    'UltraDiffusion', 'UltraDiffusionConfig', 'UltraStableDiffusion',
    
    # Vision
    'UltraVision', 'UltraVisionConfig', 'UltraResNet', 'UltraViT'
]
