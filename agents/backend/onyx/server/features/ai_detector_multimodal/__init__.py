"""
AI Detector Multimodal - Detector de contenido generado por IA
================================================================

Detecta cualquier input o output creado por IA con análisis forense.
Sistema avanzado de detección multimodal que analiza texto, imágenes y audio.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "Multimodal AI content detector with forensic analysis"

# Try to import components with error handling
try:
    from .core.detector import MultimodalAIDetector
except ImportError:
    MultimodalAIDetector = None

try:
    from .api.router import router
except ImportError:
    router = None

__all__ = [
    "MultimodalAIDetector",
    "router",
]
