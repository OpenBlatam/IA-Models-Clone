"""
Physical Store Designer AI
==========================

Sistema de IA para diseñar locales físicos completos, incluyendo:
- Diseño visual del local (layout, decoración, distribución)
- Plan de marketing y ventas
- Estrategia de decoración física
- Chat interactivo para recopilar información del cliente
"""

__version__ = "1.45.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI system for designing complete physical stores with visual layouts, marketing plans, and decoration strategies"

# Try to import components with error handling
try:
    from .api import app
except ImportError:
    app = None

try:
    from .core import (
        StoreType,
        DesignStyle,
        StoreDesignRequest,
        StoreDesign,
        MarketingPlan,
        DecorationPlan,
    )
except ImportError:
    StoreType = None
    DesignStyle = None
    StoreDesignRequest = None
    StoreDesign = None
    MarketingPlan = None
    DecorationPlan = None

__all__ = [
    "app",
    "StoreType",
    "DesignStyle",
    "StoreDesignRequest",
    "StoreDesign",
    "MarketingPlan",
    "DecorationPlan",
]
