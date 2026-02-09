"""
Color Grading AI TruthGPT
==========================

Sistema de color grading automático con arquitectura SAM3, integrado con OpenRouter y TruthGPT.
Similar a DaVinci Resolve pero completamente automático.
"""

from .core.color_grading_agent import ColorGradingAgent
from .config.color_grading_config import ColorGradingConfig

__all__ = ["ColorGradingAgent", "ColorGradingConfig"]




