"""
Chat AI Processing

Handles AI-related processing for chats (moderation, sentiment, embeddings).
"""

import logging
from typing import Optional
from ....models import PublishedChat

logger = logging.getLogger(__name__)


class ChatAIProcessor:
    """Handles AI processing for chats."""
    
    def __init__(self, chat_repository):
        """
        Initialize AI processor.
        
        Args:
            chat_repository: Chat repository to get database session
        """
        self.chat_repository = chat_repository
        self._ai_services_initialized = False
        self._embedding_service = None
        self._sentiment_service = None
        self._moderation_service = None
    
    def _init_ai_services(self):
        """Lazy initialization of AI services"""
        if self._ai_services_initialized:
            return
        
        try:
            from ....config import settings
            if settings.ai_enabled:
                from ...ai import EmbeddingService, SentimentService, ModerationService
                db = self.chat_repository.db
                self._embedding_service = EmbeddingService(db)
                self._sentiment_service = SentimentService(db)
                self._moderation_service = ModerationService(db)
            self._ai_services_initialized = True
        except Exception as e:
            logger.warning(f"Failed to initialize AI services: {e}")
            self._ai_services_initialized = True
    
    def process_chat(self, chat: PublishedChat) -> None:
        """
        Process chat with AI services (moderation, sentiment, embeddings).
        
        Args:
            chat: PublishedChat to process
            
        Raises:
            ValueError: If chat is None
        """
        if not chat:
            raise ValueError("chat cannot be None")
        
        try:
            from ....config import settings
            if not settings.ai_enabled:
                logger.debug(f"AI processing disabled, skipping chat {chat.id}")
                return
            
            self._init_ai_services()
            
            if self._moderation_service and settings.moderation_enabled:
                try:
                    moderation_result = self._moderation_service.moderate_chat(chat.id, chat=chat)
                    if moderation_result and moderation_result.is_toxic:
                        logger.warning(
                            f"Toxic content detected for chat {chat.id}: "
                            f"score={moderation_result.toxicity_score:.2f}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Moderation failed for chat {chat.id}: {e}",
                        exc_info=True
                    )
            
            if self._sentiment_service and settings.sentiment_enabled:
                try:
                    self._sentiment_service.analyze_chat_sentiment(chat.id, chat=chat)
                except Exception as e:
                    logger.warning(
                        f"Sentiment analysis failed for chat {chat.id}: {e}",
                        exc_info=True
                    )
            
            if self._embedding_service:
                try:
                    self._embedding_service.create_or_update_embedding(chat.id, chat=chat)
                except Exception as e:
                    logger.warning(
                        f"Embedding generation failed for chat {chat.id}: {e}",
                        exc_info=True
                    )
        except Exception as e:
            logger.error(
                f"Error in AI processing for chat {chat.id}: {e}",
                exc_info=True
            )






