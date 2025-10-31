"""
API Module - Módulo de API
=========================

Módulo que contiene la API del sistema ultra-ultra-ultra-ultra-refactorizado
con endpoints para todas las funcionalidades avanzadas.
"""

from .app import create_app
from .endpoints import (
    time_dilation_router,
    consciousness_router,
    dimensional_portals_router,
    quantum_teleportation_router,
    reality_manipulation_router,
    transcendent_ai_router
)
from .middleware import (
    TimeDilationMiddleware,
    ConsciousnessMiddleware,
    DimensionalPortalMiddleware,
    QuantumTeleportationMiddleware,
    RealityManipulationMiddleware,
    TranscendentAIMiddleware
)

__all__ = [
    "create_app",
    "time_dilation_router",
    "consciousness_router",
    "dimensional_portals_router",
    "quantum_teleportation_router",
    "reality_manipulation_router",
    "transcendent_ai_router",
    "TimeDilationMiddleware",
    "ConsciousnessMiddleware",
    "DimensionalPortalMiddleware",
    "QuantumTeleportationMiddleware",
    "RealityManipulationMiddleware",
    "TranscendentAIMiddleware"
]




