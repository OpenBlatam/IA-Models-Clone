"""
Refactored Content Moderation Service

Uses BaseAIService for consistent device management and mixed precision.
"""

import logging
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from transformers import pipeline
import torch

from ...config import settings
from ...models import ChatAIMetadata, PublishedChat
from ...exceptions import DatabaseError, InvalidChatError
from ...helpers import generate_id
from .base_service import BaseAIService

logger = logging.getLogger(__name__)


class ModerationServiceRefactored(BaseAIService):
    """
    Refactored content moderation service using BaseAIService
    
    Provides consistent device management, mixed precision, and error handling.
    """
    
    def __init__(self, db: Session):
        """
        Initialize moderation service
        
        Args:
            db: Database session
        """
        super().__init__(
            model_name=settings.moderation_model,
            model_type="transformer"
        )
        self.db = db
        self.pipeline = None
        if settings.moderation_enabled and settings.ai_enabled:
            self._load_model()
    
    def _load_model_impl(self) -> None:
        """Load the moderation model"""
        if not settings.moderation_enabled:
            logger.warning("Content moderation is disabled")
            return
        
        try:
            logger.info(f"Loading moderation model: {self.model_name} on {self.device}")
            
            device_id = 0 if self.device.type == "cuda" and torch.cuda.is_available() else -1
            
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=device_id,
                return_all_scores=True,
                model_kwargs={
                    "torch_dtype": torch.float16 if self.use_mixed_precision and self.device.type == "cuda" else torch.float32
                }
            )
            
            logger.info("Moderation model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading moderation model: {e}", exc_info=True)
            # Fallback to basic model
            try:
                self.pipeline = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert",
                    device=-1,
                    return_all_scores=True
                )
                logger.warning("Using fallback moderation model")
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                self.pipeline = None
    
    def moderate_content(
        self,
        text: str,
        return_all_scores: bool = False
    ) -> Dict[str, Any]:
        """
        Check content for toxicity and inappropriate material
        
        Args:
            text: Text to moderate
            return_all_scores: Whether to return all class scores
            
        Returns:
            Dict with moderation results including toxicity score and flags
        """
        if not self.pipeline:
            if not settings.moderation_enabled:
                return {
                    "is_toxic": False,
                    "toxicity_score": 0.0,
                    "flags": [],
                    "enabled": False
                }
            raise RuntimeError("Moderation model not loaded")
        
        if not text or not text.strip():
            return {"is_toxic": False, "toxicity_score": 0.0, "flags": []}
        
        try:
            # Truncate text if too long
            max_length = 512
            text_to_moderate = text[:max_length] if len(text) > max_length else text
            
            with self.inference_context():
                results = self.pipeline(text_to_moderate)
            
            # Extract toxicity scores
            toxicity_score = 0.0
            flags = []
            all_scores = {}
            
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    # Multiple labels returned
                    for result in results[0]:
                        label = result['label'].lower()
                        score = float(result['score'])
                        all_scores[label] = score
                        
                        # Check for toxic labels
                        toxic_labels = ['toxic', 'hate', 'threat', 'insult', 'obscene', 'severe_toxic']
                        if any(toxic_label in label for toxic_label in toxic_labels):
                            if score > toxicity_score:
                                toxicity_score = score
                            flags.append({"label": label, "score": score})
                else:
                    # Single result
                    result = results[0]
                    label = result['label'].lower()
                    score = float(result['score'])
                    all_scores[label] = score
                    
                    toxic_labels = ['toxic', 'hate', 'threat', 'insult', 'obscene']
                    if any(toxic_label in label for toxic_label in toxic_labels):
                        toxicity_score = score
                        flags.append({"label": label, "score": score})
            
            is_toxic = toxicity_score >= settings.moderation_threshold
            
            result = {
                "is_toxic": is_toxic,
                "toxicity_score": toxicity_score,
                "flags": flags,
                "threshold": settings.moderation_threshold
            }
            
            if return_all_scores:
                result["all_scores"] = all_scores
            
            return result
            
        except Exception as e:
            logger.error(f"Error moderating content: {e}", exc_info=True)
            # Fail safe - don't block content if moderation fails
            return {
                "is_toxic": False,
                "toxicity_score": 0.0,
                "flags": [],
                "error": str(e)
            }
    
    def moderate_chat(
        self,
        chat_id: str,
        chat: Optional[PublishedChat] = None
    ) -> ChatAIMetadata:
        """
        Moderate a chat and store results
        
        Args:
            chat_id: ID of the chat
            chat: Optional chat object
            
        Returns:
            ChatAIMetadata with moderation results
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
            
            # Combine text for moderation
            text_parts = [chat.title]
            if chat.description:
                text_parts.append(chat.description)
            text_parts.append(chat.chat_content)
            full_text = " ".join(text_parts)
            
            # Moderate content
            moderation_result = self.moderate_content(full_text)
            
            # Get or create metadata
            metadata = self.db.query(ChatAIMetadata).filter(
                ChatAIMetadata.chat_id == chat_id
            ).first()
            
            if metadata:
                metadata.is_toxic = moderation_result["is_toxic"]
                metadata.toxicity_score = moderation_result["toxicity_score"]
                metadata.moderation_flags = moderation_result["flags"]
                self.db.commit()
                self.db.refresh(metadata)
            else:
                metadata_id = generate_id()
                metadata = ChatAIMetadata(
                    id=metadata_id,
                    chat_id=chat_id,
                    is_toxic=moderation_result["is_toxic"],
                    toxicity_score=moderation_result["toxicity_score"],
                    moderation_flags=moderation_result["flags"],
                    ai_analysis_version="1.0"
                )
                self.db.add(metadata)
                self.db.commit()
                self.db.refresh(metadata)
            
            logger.info(
                f"Moderated chat {chat_id}: "
                f"toxic={moderation_result['is_toxic']}, "
                f"score={moderation_result['toxicity_score']:.2f}"
            )
            
            return metadata
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error moderating chat: {e}", exc_info=True)
            raise DatabaseError(f"Failed to moderate chat: {str(e)}")
    
    def should_block_content(self, text: str) -> bool:
        """
        Check if content should be blocked based on moderation
        
        Args:
            text: Text to check
            
        Returns:
            True if content should be blocked
        """
        result = self.moderate_content(text)
        return result.get("is_toxic", False)





