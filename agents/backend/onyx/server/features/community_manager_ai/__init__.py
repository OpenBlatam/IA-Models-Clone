"""
Community Manager AI - Sistema de Gestión de Redes Sociales
============================================================

Sistema completo para gestión automatizada de redes sociales que incluye:
- Gestión de memes
- Calendario de publicaciones
- Scripts de automatización
- Conexiones a todas las plataformas sociales
- Programación y organización de publicaciones
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "Complete automated social media management system with meme management, post scheduling, and multi-platform connections"

# Try to import components with error handling
try:
    from .core.community_manager import CommunityManager
    from .core.scheduler import PostScheduler
except ImportError:
    CommunityManager = None
    PostScheduler = None

try:
    from .services.meme_manager import MemeManager
    from .services.social_media_connector import SocialMediaConnector
except ImportError:
    MemeManager = None
    SocialMediaConnector = None

try:
    from .api import create_app
except ImportError:
    create_app = None

__all__ = [
    "CommunityManager",
    "PostScheduler",
    "MemeManager",
    "SocialMediaConnector",
    "create_app",
]
