"""
Dog Training Coaching AI
========================

AI-powered dog training assistant with personalized coaching.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered dog training assistant with personalized coaching"

# Try to import components with error handling
try:
    from .services.coaching_service import DogTrainingCoach
except ImportError:
    DogTrainingCoach = None

try:
    from .api.routes import router
except ImportError:
    router = None

__all__ = [
    "DogTrainingCoach",
    "router",
]