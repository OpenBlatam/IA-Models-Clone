"""
Constants - Constantes compartidas
===================================

Constantes utilizadas en todo el generador de proyectos.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

from enum import Enum
from typing import Final

DEFAULT_PROJECT_VERSION: Final[str] = "1.0.0"
DEFAULT_AUTHOR: Final[str] = "Blatam Academy"
DEFAULT_BACKEND_PORT: Final[int] = 8000
DEFAULT_FRONTEND_PORT: Final[int] = 3000
MAX_PROJECT_NAME_LENGTH: Final[int] = 50


class FrameworkType(str, Enum):
    """Tipos de frameworks soportados."""
    FASTAPI = "fastapi"
    FLASK = "flask"
    DJANGO = "django"
    REACT = "react"
    VUE = "vue"
    NEXTJS = "nextjs"


class ProjectComplexity(str, Enum):
    """Niveles de complejidad del proyecto."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class AIType(str, Enum):
    """Tipos de IA soportados."""
    GENERAL = "general"
    CHAT = "chat"
    VISION = "vision"
    AUDIO = "audio"
    NLP = "nlp"
    VIDEO = "video"
    RECOMMENDATION = "recommendation"
    ANALYTICS = "analytics"
    GENERATION = "generation"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    QA = "qa"


class ModelArchitecture(str, Enum):
    """Arquitecturas de modelos soportadas."""
    TRANSFORMER = "transformer"
    DIFFUSION = "diffusion"
    LLM = "llm"
    CNN = "cnn"
    RNN = "rnn"
    CUSTOM = "custom"


SUPPORTED_BACKEND_FRAMEWORKS: Final[list[str]] = [FrameworkType.FASTAPI.value]
SUPPORTED_FRONTEND_FRAMEWORKS: Final[list[str]] = [FrameworkType.REACT.value]

COMMON_FEATURES: Final[list[str]] = [
    "dashboard",
    "rest_api",
    "graphql",
    "admin_panel",
    "monitoring",
    "logging",
    "testing",
    "docker",
]
