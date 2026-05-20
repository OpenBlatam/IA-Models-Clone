"""
Contabilidad Mexicana AI
========================

Sistema de IA para resolver problemas contables y fiscales mexicanos.
Basado en Contarely, proporciona asesoría fiscal, cálculo de impuestos,
guías y soporte para trámites del SAT usando OpenRouter.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"

from .core.contador_ai import ContadorAI
from .config.contador_config import ContadorConfig

__all__ = ["ContadorAI", "ContadorConfig"]
