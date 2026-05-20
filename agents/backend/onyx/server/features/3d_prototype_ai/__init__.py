"""
3D Prototype AI - Sistema de Generación de Prototipos y Modelos 3D
===================================================================

Sistema de IA que genera prototipos completos de productos 3D incluyendo:
- Documentación completa (materiales, precios, especificaciones)
- Modelos CAD por partes
- Instrucciones de ensamblaje
- Opciones según presupuesto
- Fuentes de materiales
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI system for generating complete 3D product prototypes with documentation, CAD models, and assembly instructions"

# Try to import components with error handling
try:
    from .core.prototype_generator import PrototypeGenerator
except ImportError:
    PrototypeGenerator = None

__all__ = [
    "PrototypeGenerator",
]
