"""
Refactored Sentiment Analysis Service

Uses BaseAIService for consistent device management and mixed precision.
"""

import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from transformers import pipeline
import torch

from ...config import settings
from ...models import ChatAIMetadata, PublishedChat
from ...exceptions import DatabaseError
from ...helpers import generate_id
from .base_service import BaseAIService

logger = logging.getLogger(__name__)


class SentimentServiceRefactored(BaseAIService):
    """
    Refactored sentiment analysis service using BaseAIService
    
    Provides consistent device management, mixed precision, and error handling.
    """
    
    def __init__(self, db: Session):
        """
        Initialize sentiment service
        
        Args:
            db: Database session
        """
        super().__init__(
            model_name=settings.sentiment_model,
            model_type="transformer"
        )
        self.db = db
        self.pipeline = None
        if settings.sentiment_enabled and settings.ai_enabled:
            self._load_model()
    
    def _load_model_impl(self) -> None:
        """Load the sentiment analysis model"""
        if not settings.sentiment_enabled:
            logger.warning("Sentiment analysis is disabled")
            return
        
        try:
            logger.info(f"Loading sentiment model: {self.model_name} on {self.device}")
            
            device_id = 0 if self.device.type == "cuda" and torch.cuda.is_available() else -1
            
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=device_id,
                return_all_scores=True,
                model_kwargs={
                    "torch_dtype": torch.float16 if self.use_mixed_precision and self.device.type == "cuda" else torch.float32
                }
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
                self.pipeline = None
    
    def analyze_sentiment(
        self,
        text: str,
        return_all_scores: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            return_all_scores: Whether to return all class scores
            
        Returns:
            Dict with sentiment label and score
        """
        if not self.pipeline:
            if not settings.sentiment_enabled:
                return {"label": "neutral", "score": 0.5, "enabled": False}
            raise RuntimeError("Sentiment model not loaded")
        
        if not text or not text.strip():
            return {"label": "neutral", "score": 0.5}
        
        try:
            # Truncate text if too long
            max_length = 512
            text_to_analyze = text[:max_length] if len(text) > max_length else text
            
            with self.inference_context():
                results = self.pipeline(text_to_analyze)
            
            # Process results
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
                
                result = {
                    "label": label,
                    "score": score
                }
                
                if return_all_scores:
                    result["all_scores"] = results
                
                return result
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
        Analyze sentiment of a chat and store results
        
        Args:
            chat_id: ID of the chat
            chat: Optional chat object
            
        Returns:
            ChatAIMetadata with sentiment analysis
        """
        if not chat_id:
            raise ValueError("Chat ID cannot be empty")
        
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





