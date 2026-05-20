"""
Imagen Video Enhancer AI - Sistema de mejoramiento de imágenes y videos
======================================================================

Sistema autónomo para mejorar imágenes y videos usando OpenRouter y TruthGPT.
Similar a Krea AI, permite subir imágenes/videos y mejorarlos con IA.

Características:
- Mejora de calidad de imágenes
- Mejora de calidad de videos
- Upscaling inteligente
- Reducción de ruido
- Mejora de colores y contraste
- Restauración de imágenes antiguas
- Arquitectura SAM3 para procesamiento paralelo
"""

from .core.enhancer_agent import EnhancerAgent
from .config.enhancer_config import EnhancerConfig

__version__ = "1.0.0"
__all__ = ["EnhancerAgent", "EnhancerConfig"]




