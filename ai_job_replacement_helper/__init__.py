"""
AI Job Replacement Helper 🎯
============================

Sistema inteligente que ayuda a las personas cuando una IA les quita su trabajo.
Incluye gamificación, pasos guiados y búsqueda de trabajo estilo Tinder con LinkedIn API.

Características principales:
- 🎮 Sistema de gamificación (puntos, niveles, badges, logros)
- 📋 Pasos guiados personalizados y roadmap
- 💼 Integración con LinkedIn API estilo Tinder (swipe de trabajos)
- 🤖 Recomendaciones inteligentes basadas en IA
- 📊 Tracking de progreso y métricas
- 🎓 Sistema de mentoría y coaching
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "Intelligent system to help people when AI replaces their jobs, with gamification, guided steps, and Tinder-style job search"

# Try to import components with error handling
try:
    from .core.gamification import GamificationService
except ImportError:
    GamificationService = None

try:
    from .core.steps_guide import StepsGuideService
except ImportError:
    StepsGuideService = None

try:
    from .core.linkedin_integration import LinkedInIntegrationService
except ImportError:
    LinkedInIntegrationService = None

try:
    from .core.recommendations import RecommendationService
except ImportError:
    RecommendationService = None

__all__ = [
    "GamificationService",
    "StepsGuideService",
    "LinkedInIntegrationService",
    "RecommendationService",
]
