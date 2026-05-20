"""
AI Tutor Educacional con Open Router
====================================

Sistema de tutoría educacional inteligente que utiliza Open Router
para proporcionar asistencia educativa personalizada.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "Intelligent educational tutoring system using Open Router for personalized educational assistance"

# Try to import components with error handling
try:
    from .core.tutor import AITutor
except ImportError:
    AITutor = None

try:
    from .config.tutor_config import TutorConfig
except ImportError:
    TutorConfig = None

__all__ = [
    "AITutor",
    "TutorConfig",
]
