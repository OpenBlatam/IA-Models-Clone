"""
Sentiment Analysis Service using transformers

Uses pre-trained transformer models for sentiment analysis
of chat content.
"""

import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

from ...config import settings
from ...models import ChatAIMetadata, PublishedChat
from ...exceptions import DatabaseError
from ...helpers import generate_id

logger = logging.getLogger(__name__)


class SentimentService:
    """Service for sentiment analysis of chat content"""
    
    def __init__(self, db: Session):
        """
        Initialize SentimentService.
        
        Args:
            db: Database session
            
        Raises:
            ValueError: If db is None
        """
        if db is None:
            raise ValueError("Database session (db) cannot be None")
        
        self.db = db
        self.device = settings.device if settings.ai_enabled and torch.cuda.is_available() and settings.use_gpu else "cpu"
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        if settings.sentiment_enabled and settings.ai_enabled:
            self._load_model()
    
    def _load_model(self) -> None:
        """Lazy load the sentiment analysis model"""
        try:
            logger.info(f"Loading sentiment model: {settings.sentiment_model} on {self.device}")
            
            # Use pipeline for easier inference
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=settings.sentiment_model,
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}", exc_info=True)
            # Fallback to basic model
            try:
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    device=-1,
                    return_all_scores=True
                )
                logger.warning("Using fallback sentiment model")
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                raise
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with sentiment label and score
            
        Raises:
            RuntimeError: If sentiment model is not loaded and sentiment is enabled
            ValueError: If text is not a string
        """
        if not self.pipeline:
            if not settings.sentiment_enabled:
                return {"label": "neutral", "score": 0.5, "enabled": False}
            raise RuntimeError("Sentiment model not loaded")
        
        if not isinstance(text, str):
            raise ValueError(f"text must be a string, got {type(text).__name__}")
        
        if not text or not text.strip():
            return {"label": "neutral", "score": 0.5}
        
        try:
            # Truncate text if too long (most models have token limits)
            max_length = 512
            text_to_analyze = text[:max_length] if len(text) > max_length else text
            
            results = self.pipeline(text_to_analyze)
            
            # Get the highest confidence prediction
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    # Multiple scores returned
                    best_result = max(results[0], key=lambda x: x['score'])
                else:
                    best_result = results[0]
                
                label = best_result['label'].lower()
                score = float(best_result['score'])
                
                # Normalize labels
                if 'positive' in label:
                    label = 'positive'
                elif 'negative' in label:
                    label = 'negative'
                else:
                    label = 'neutral'
                
                return {
                    "label": label,
                    "score": score,
                    "raw_results": results
                }
            else:
                return {"label": "neutral", "score": 0.5}
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}", exc_info=True)
            return {"label": "neutral", "score": 0.5, "error": str(e)}
    
    def analyze_chat_sentiment(
        self,
        chat_id: str,
        chat: Optional[PublishedChat] = None
    ) -> ChatAIMetadata:
        """
        Analyze sentiment of a chat and store results.
        
        Args:
            chat_id: ID of the chat
            chat: Optional chat object
            
        Returns:
            ChatAIMetadata with sentiment analysis
            
        Raises:
            ValueError: If chat_id is None, empty, or chat not found
        """
        if not chat_id or not isinstance(chat_id, str) or not chat_id.strip():
            raise ValueError(f"chat_id must be a non-empty string, got {type(chat_id).__name__}")
        
        chat_id = chat_id.strip()
        
        try:
            if not chat:
                chat = self.db.query(PublishedChat).filter(
                    PublishedChat.id == chat_id
                ).first()
            
            if not chat:
                raise ValueError(f"Chat {chat_id} not found")
            
            # Combine text for analysis
            text_parts = [chat.title]
            if chat.description:
                text_parts.append(chat.description)
            text_parts.append(chat.chat_content)
            full_text = " ".join(text_parts)
            
            # Analyze sentiment
            sentiment_result = self.analyze_sentiment(full_text)
            
            # Get or create metadata
            metadata = self.db.query(ChatAIMetadata).filter(
                ChatAIMetadata.chat_id == chat_id
            ).first()
            
            if metadata:
                metadata.sentiment_label = sentiment_result["label"]
                metadata.sentiment_score = sentiment_result["score"]
                self.db.commit()
                self.db.refresh(metadata)
            else:
                metadata_id = generate_id()
                metadata = ChatAIMetadata(
                    id=metadata_id,
                    chat_id=chat_id,
                    sentiment_label=sentiment_result["label"],
                    sentiment_score=sentiment_result["score"],
                    ai_analysis_version="1.0"
                )
                self.db.add(metadata)
                self.db.commit()
                self.db.refresh(metadata)
            
            logger.info(f"Analyzed sentiment for chat {chat_id}: {sentiment_result['label']}")
            return metadata
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error analyzing chat sentiment: {e}", exc_info=True)
            raise DatabaseError(f"Failed to analyze sentiment: {str(e)}")





