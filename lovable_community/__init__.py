"""
Lovable Community - Sistema de comunidad para compartir y remixar chats

Este módulo proporciona funcionalidades para:
- Publicar chats en la comunidad
- Remixar chats de otros usuarios
- Sistema de votación y ranking
- Descubrimiento de contenido popular
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered community platform for sharing and remixing chats"

# Try to import main components with error handling
try:
    from .services.chat import ChatService
except ImportError:
    ChatService = None

try:
    from .services.ranking import RankingService
except ImportError:
    RankingService = None

try:
    from .api.main import create_app
except ImportError:
    create_app = None

__all__ = [
    "ChatService",
    "RankingService",
    "create_app",
]

