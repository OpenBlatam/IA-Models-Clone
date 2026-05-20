"""
Improved Embedding Service for semantic search using sentence transformers

This service follows PyTorch/Transformers best practices:
- Proper device management and mixed precision
- Efficient batch processing
- Gradient-free inference
- Error handling and logging
- Memory optimization
"""

import logging
from typing import List, Optional, Dict, Any, Union
import numpy as np
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
import torch
from torch.cuda.amp import autocast

from .base_service import BaseAIService
from ...config import settings
from ...models import ChatEmbedding, PublishedChat
from ...exceptions import DatabaseError
from ...helpers import generate_id

logger = logging.getLogger(__name__)


class EmbeddingServiceImproved(BaseAIService):
    """
    Improved embedding service with PyTorch best practices.
    
    Features:
    - Inherits from BaseAIService for device management
    - Mixed precision inference support
    - Efficient batch processing
    - Proper error handling
    - Memory optimization
    """
    
    def __init__(self, db: Session, model_name: Optional[str] = None):
        """
        Initialize embedding service.
        
        Args:
            db: Database session
            model_name: Optional model name (defaults to settings.embedding_model)
        """
        self.db = db
        model_name = model_name or settings.embedding_model
        super().__init__(model_name, model_type="transformer")
        self._load_model()
    
    def _load_model_impl(self) -> None:
        """
        Load the sentence transformer model.
        
        Implements abstract method from BaseAIService.
        """
        if not settings.ai_enabled:
            logger.warning("AI features are disabled")
            return
        
        try:
            logger.info(
                f"Loading embedding model: {self.model_name} "
                f"on device: {self.device}"
            )
            
            # Load model with proper device handling
            self.model = SentenceTransformer(
                self.model_name,
                device=str(self.device),
                cache_folder=settings.model_cache_dir
            )
            
            # Move model to device explicitly
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            
            # Set to evaluation mode for inference
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            logger.info(
                f"Embedding model loaded successfully. "
                f"Device: {self.device}, "
                f"Mixed precision: {self.use_mixed_precision}"
            )
            
        except Exception as e:
            logger.error(
                f"Error loading embedding model: {e}",
                exc_info=True
            )
            raise RuntimeError(f"Failed to load embedding model: {e}") from e
    
    def generate_embedding(
        self,
        text: str,
        normalize: bool = True,
        show_progress: bool = False
    ) -> List[float]:
        """
        Generate embedding for a text string.
        
        Uses proper inference context with no_grad and mixed precision.
        
        Args:
            text: Input text to embed
            normalize: Whether to normalize the embedding
            show_progress: Whether to show progress bar
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            RuntimeError: If model not loaded
            ValueError: If text is empty
        """
        if not self.model:
            raise RuntimeError(
                "Embedding model not loaded. AI features may be disabled."
            )
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            # Use inference context (no_grad + mixed precision)
            with self.inference_context():
                # Encode with proper settings
                embedding = self.model.encode(
                    text.strip(),
                    convert_to_numpy=True,
                    show_progress_bar=show_progress,
                    batch_size=1,
                    normalize_embeddings=normalize,
                    device=str(self.device)
                )
            
            return embedding.tolist()
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU out of memory: {e}")
            torch.cuda.empty_cache()
            raise RuntimeError("GPU out of memory. Try reducing batch size.") from e
        except Exception as e:
            logger.error(
                f"Error generating embedding: {e}",
                exc_info=True
            )
            raise RuntimeError(f"Failed to generate embedding: {e}") from e
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize: bool = True,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        More efficient than calling generate_embedding multiple times.
        Uses proper batch processing with mixed precision.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size (defaults to settings.batch_size_embeddings)
            normalize: Whether to normalize embeddings
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If model not loaded
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        if not texts:
            return []
        
        batch_size = batch_size or getattr(
            settings, 'batch_size_embeddings', 32
        )
        
        try:
            # Use inference context
            with self.inference_context():
                embeddings = self.model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=show_progress,
                    batch_size=batch_size,
                    normalize_embeddings=normalize,
                    device=str(self.device)
                )
            
            return embeddings.tolist()
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU out of memory during batch encoding: {e}")
            torch.cuda.empty_cache()
            # Try with smaller batch size
            if batch_size > 1:
                logger.info(f"Retrying with batch size {batch_size // 2}")
                return self.generate_embeddings_batch(
                    texts,
                    batch_size=batch_size // 2,
                    normalize=normalize,
                    show_progress=show_progress
                )
            raise RuntimeError("GPU out of memory. Try reducing batch size.") from e
        except Exception as e:
            logger.error(
                f"Error generating batch embeddings: {e}",
                exc_info=True
            )
            raise RuntimeError(f"Failed to generate batch embeddings: {e}") from e
    
    def create_or_update_embedding(
        self,
        chat_id: str,
        text: Optional[str] = None,
        chat: Optional[PublishedChat] = None
    ) -> ChatEmbedding:
        """
        Create or update embedding for a chat.
        
        Args:
            chat_id: ID of the chat
            text: Optional text to embed (if not provided, uses chat content)
            chat: Optional chat object (if not provided, fetches from DB)
            
        Returns:
            ChatEmbedding object
            
        Raises:
            ValueError: If chat_id is empty or chat not found
            DatabaseError: If database operation fails
        """
        if not chat_id:
            raise ValueError("Chat ID cannot be empty")
        
        try:
            # Fetch chat if not provided
            if not chat:
                chat = self.db.query(PublishedChat).filter(
                    PublishedChat.id == chat_id
                ).first()
            
            if not chat:
                raise ValueError(f"Chat {chat_id} not found")
            
            # Prepare text for embedding
            if not text:
                text_parts = [chat.title]
                if chat.description:
                    text_parts.append(chat.description)
                if chat.chat_content:
                    text_parts.append(str(chat.chat_content))
                text = " ".join(filter(None, text_parts))
            
            if not text.strip():
                raise ValueError(f"Chat {chat_id} has no content to embed")
            
            # Generate embedding
            embedding_vector = self.generate_embedding(text)
            
            # Check if embedding already exists
            existing = self.db.query(ChatEmbedding).filter(
                ChatEmbedding.chat_id == chat_id
            ).first()
            
            if existing:
                existing.embedding = embedding_vector
                existing.embedding_model = self.model_name
                self.db.commit()
                self.db.refresh(existing)
                logger.info(f"Updated embedding for chat {chat_id}")
                return existing
            else:
                embedding_id = generate_id()
                embedding = ChatEmbedding(
                    id=embedding_id,
                    chat_id=chat_id,
                    embedding=embedding_vector,
                    embedding_model=self.model_name
                )
                self.db.add(embedding)
                self.db.commit()
                self.db.refresh(embedding)
                logger.info(f"Created embedding for chat {chat_id}")
                return embedding
                
        except ValueError:
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(
                f"Error creating/updating embedding for chat {chat_id}: {e}",
                exc_info=True
            )
            raise DatabaseError(f"Failed to create embedding: {str(e)}") from e
    
    def find_similar_chats(
        self,
        query_text: str,
        limit: int = 10,
        min_similarity: float = 0.5,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find similar chats using semantic search with cosine similarity.
        
        Uses efficient batch processing and proper tensor operations.
        
        Args:
            query_text: Query text to search for
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)
            batch_size: Batch size for processing embeddings
            
        Returns:
            List of dicts with chat info and similarity scores
            
        Raises:
            RuntimeError: If model not loaded
            ValueError: If query_text is empty
            DatabaseError: If database operation fails
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")
        
        try:
            # Generate query embedding
            query_embedding = np.array(
                self.generate_embedding(query_text),
                dtype=np.float32
            )
            query_norm = np.linalg.norm(query_embedding)
            
            # Get all embeddings from database
            embeddings = self.db.query(ChatEmbedding).join(
                PublishedChat
            ).filter(
                PublishedChat.is_public == True
            ).all()
            
            if not embeddings:
                logger.info("No embeddings found in database")
                return []
            
            # Process in batches for efficiency
            batch_size = batch_size or getattr(
                settings, 'batch_size_embeddings', 32
            )
            similarities = []
            
            # Process embeddings in batches
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size]
                embedding_vectors = np.array(
                    [np.array(emb.embedding, dtype=np.float32) for emb in batch]
                )
                
                # Batch cosine similarity computation
                norms = np.linalg.norm(embedding_vectors, axis=1)
                dots = np.dot(embedding_vectors, query_embedding)
                batch_similarities = dots / (norms * query_norm)
                
                # Filter and collect results
                for emb, similarity in zip(batch, batch_similarities):
                    if similarity >= min_similarity:
                        similarities.append({
                            "chat_id": emb.chat_id,
                            "similarity": float(similarity),
                            "chat": emb.chat
                        })
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            logger.info(
                f"Found {len(similarities)} similar chats "
                f"(min_similarity={min_similarity})"
            )
            
            return similarities[:limit]
            
        except Exception as e:
            logger.error(
                f"Error finding similar chats: {e}",
                exc_info=True
            )
            raise DatabaseError(f"Failed to find similar chats: {str(e)}") from e
    
    def get_embedding(self, chat_id: str) -> Optional[ChatEmbedding]:
        """
        Get embedding for a chat.
        
        Args:
            chat_id: ID of the chat
            
        Returns:
            ChatEmbedding object or None if not found
        """
        return self.db.query(ChatEmbedding).filter(
            ChatEmbedding.chat_id == chat_id
        ).first()
    
    def delete_embedding(self, chat_id: str) -> bool:
        """
        Delete embedding for a chat.
        
        Args:
            chat_id: ID of the chat
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            DatabaseError: If database operation fails
        """
        try:
            embedding = self.get_embedding(chat_id)
            if embedding:
                self.db.delete(embedding)
                self.db.commit()
                logger.info(f"Deleted embedding for chat {chat_id}")
                return True
            return False
        except Exception as e:
            self.db.rollback()
            logger.error(
                f"Error deleting embedding: {e}",
                exc_info=True
            )
            raise DatabaseError(f"Failed to delete embedding: {str(e)}") from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        
        if self.model:
            # Add sentence transformer specific info
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                info["embedding_dimension"] = (
                    self.model.get_sentence_embedding_dimension()
                )
            if hasattr(self.model, 'max_seq_length'):
                info["max_seq_length"] = self.model.max_seq_length
        
        return info
