"""
MCP Contracts - Contratos de entrada/salida estandarizados
===========================================================

Define "frames" de contexto estandarizados:
- Qué campos, token limits, formatos
- Helpers para serializar/deserializar contextos
"""

from .models import ContextFrame, PromptFrame, FrameMetadata
from .serializer import FrameSerializer

__all__ = [
    "ContextFrame",
    "PromptFrame",
    "FrameMetadata",
    "FrameSerializer",
]

