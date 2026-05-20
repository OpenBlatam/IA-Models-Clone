"""
Manuales Hogar AI
==================

Sistema de IA para generar manuales paso a paso tipo LEGO para:
- Plomería
- Techos y reparaciones
- Carpintería
- Electricidad
- Albañilería
- Otros oficios populares

Permite procesar imágenes (fotos de problemas) o descripciones de texto
y genera guías visuales y detalladas de cómo resolver el problema.

Versión: 1.0.0
Autor: Blatam Academy
Licencia: Propietaria
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI system for generating step-by-step home repair manuals (plumbing, roofing, carpentry, electrical, masonry)"

# Try to import components with error handling
try:
    from .core.manual_generator import ManualGenerator
except ImportError:
    ManualGenerator = None

try:
    from .infrastructure.openrouter_client import OpenRouterClient
except ImportError:
    OpenRouterClient = None

try:
    from .api import router
except ImportError:
    router = None

__all__ = [
    "ManualGenerator",
    "OpenRouterClient",
    "router",
]
