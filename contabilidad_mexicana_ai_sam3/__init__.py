"""
Contabilidad Mexicana AI SAM3
=============================

Sistema de contabilidad mexicana con arquitectura SAM3, integrado con OpenRouter y TruthGPT.
Combina la funcionalidad de contabilidad fiscal con la arquitectura avanzada de SAM3.
"""

from .core.contador_sam3_agent import ContadorSAM3Agent
from .config.contador_sam3_config import ContadorSAM3Config

__version__ = "1.0.0"
__all__ = ["ContadorSAM3Agent", "ContadorSAM3Config"]
