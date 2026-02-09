"""
Social Media Identity Clone AI
==============================

Sistema de IA para clonar identidades de perfiles de redes sociales
(TikTok, Instagram, YouTube) y generar contenido basado en esa identidad.

Características:
- Extracción de datos de perfiles de redes sociales
- Análisis de contenido de videos y posts
- Construcción de perfil de identidad
- Generación de contenido basado en identidad clonada
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI system for cloning social media identities and generating content based on cloned profiles"

# Try to import components with error handling
try:
    from .core.models import (
        SocialProfile,
        IdentityProfile,
        ContentAnalysis,
        GeneratedContent,
    )
except ImportError:
    SocialProfile = None
    IdentityProfile = None
    ContentAnalysis = None
    GeneratedContent = None

try:
    from .services.profile_extractor import ProfileExtractor
    from .services.identity_analyzer import IdentityAnalyzer
    from .services.content_generator import ContentGenerator
except ImportError:
    ProfileExtractor = None
    IdentityAnalyzer = None
    ContentGenerator = None

__all__ = [
    "SocialProfile",
    "IdentityProfile",
    "ContentAnalysis",
    "GeneratedContent",
    "ProfileExtractor",
    "IdentityAnalyzer",
    "ContentGenerator",
]
