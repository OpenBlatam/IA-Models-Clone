"""
AI Project Generator - Generador Automático de Proyectos de IA
===============================================================

Sistema que genera automáticamente la estructura completa de backend y frontend
para proyectos de IA basándose en una descripción del usuario.
Funciona de forma continua sin parar, generando proyectos automáticamente.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "Automatic AI project generator that creates complete backend and frontend structures based on user descriptions"

# Try to import components with error handling
try:
    from .core.project_generator import ProjectGenerator
except ImportError:
    ProjectGenerator = None

try:
    from .core.backend_generator import BackendGenerator
except ImportError:
    BackendGenerator = None

try:
    from .core.frontend_generator import FrontendGenerator
except ImportError:
    FrontendGenerator = None

try:
    from .core.continuous_generator import ContinuousGenerator
except ImportError:
    ContinuousGenerator = None

try:
    from .core.deep_learning_generator import DeepLearningGenerator
except ImportError:
    DeepLearningGenerator = None

try:
    from .api.generator_api import create_generator_app
except ImportError:
    create_generator_app = None

__all__ = [
    "ProjectGenerator",
    "BackendGenerator",
    "FrontendGenerator",
    "ContinuousGenerator",
    "DeepLearningGenerator",
    "create_generator_app",
]
