"""
Modelos de base de datos para la comunidad Lovable (backward compatibility)

Este archivo mantiene compatibilidad hacia atrás importando desde el módulo models/.
Los modelos están ahora organizados en archivos separados en models/ para mejor modularidad.

Para nuevas importaciones, use:
    from .models import PublishedChat, ChatRemix, etc.
"""

# Import all models from the modular structure for backward compatibility
# Note: We import from models/ directory (not from this file)
from .models.published_chat import PublishedChat
from .models.chat_remix import ChatRemix
from .models.chat_vote import ChatVote
from .models.chat_view import ChatView
from .models.chat_embedding import ChatEmbedding
from .models.chat_ai_metadata import ChatAIMetadata
from .models.base import Base

__all__ = [
    "Base",
    "PublishedChat",
    "ChatRemix",
    "ChatVote",
    "ChatView",
    "ChatEmbedding",
    "ChatAIMetadata",
]

