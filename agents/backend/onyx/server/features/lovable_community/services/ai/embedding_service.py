"""
Embedding Service for semantic search using sentence transformers

Uses sentence-transformers to generate embeddings for chats,
enabling semantic similarity search.
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
import torch

from ...config import settings
from ...models import ChatEmbedding, PublishedChat
from ...exceptions import DatabaseError
from ...helpers import generate_id

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing text embeddings"""
    
    def __init__(self, db: Session):
        """
        Initialize EmbeddingService.
        
        Args:
            db: Database session
            
        Raises:
            ValueError: If db is None
        """
        if db is None:
            raise ValueError("Database session (db) cannot be None")
        
        self.db = db
        self.model = None
        self.device = settings.device if settings.ai_enabled and torch.cuda.is_available() and settings.use_gpu else "cpu"
        self._load_model()
    
    def _load_model(self) -> None:
        """Lazy load the embedding model"""
        if not settings.ai_enabled:
            logger.warning("AI features are disabled")
            return
        
        try:
            logger.info(f"Loading embedding model: {settings.embedding_model} on {self.device}")
            self.model = SentenceTransformer(
                settings.embedding_model,
                device=self.device,
                cache_folder=settings.model_cache_dir
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}", exc_info=True)
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            RuntimeError: If embedding model is not loaded
            ValueError: If text is None, empty, or not a string
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded. AI features may be disabled.")
        
        if not isinstance(text, str):
            raise ValueError(f"text must be a string, got {type(text).__name__}")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        
        try:
            # Combine title, description, and content for better embeddings
            embedding = self.model.encode(
                text.strip(),
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=1
            )
            
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If embedding model is not loaded
            ValueError: If texts is None or contains invalid entries
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        if texts is None:
            raise ValueError("texts cannot be None")
        
        if not isinstance(texts, list):
            raise ValueError(f"texts must be a list, got {type(texts).__name__}")
        
        if not texts:
            return []
        
        # Validate all texts are strings and not empty
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"texts[{i}] must be a string, got {type(text).__name__}")
            if not text.strip():
                raise ValueError(f"texts[{i}] cannot be empty or only whitespace")
        
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=settings.batch_size_embeddings
            )
            
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}", exc_info=True)
            raise
    
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
            ValueError: If chat_id is None, empty, or chat not found
            RuntimeError: If embedding model is not loaded
        """
        if not chat_id or not isinstance(chat_id, str) or not chat_id.strip():
            raise ValueError(f"chat_id must be a non-empty string, got {type(chat_id).__name__}")
        
        chat_id = chat_id.strip()
        
        if text is not None and not isinstance(text, str):
            raise ValueError(f"text must be a string if provided, got {type(text).__name__}")
        
        try:
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
                text_parts.append(chat.chat_content)
                text = " ".join(text_parts)
            
            # Generate embedding
            embedding_vector = self.generate_embedding(text)
            
            # Check if embedding already exists
            existing = self.db.query(ChatEmbedding).filter(
                ChatEmbedding.chat_id == chat_id
            ).first()
            
            if existing:
                existing.embedding = embedding_vector
                existing.embedding_model = settings.embedding_model
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
                    embedding_model=settings.embedding_model
                )
                self.db.add(embedding)
                self.db.commit()
                self.db.refresh(embedding)
                logger.info(f"Created embedding for chat {chat_id}")
                return embedding
                
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating/updating embedding for chat {chat_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to create embedding: {str(e)}")
    
    def find_similar_chats(
        self,
        query_text: str,
        limit: int = 10,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find similar chats using semantic search.
        
        Args:
            query_text: Query text to search for
            limit: Maximum number of results (must be > 0)
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of dicts with chat info and similarity scores
            
        Raises:
            RuntimeError: If embedding model is not loaded
            ValueError: If query_text is invalid, limit <= 0, or min_similarity out of range
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        if not isinstance(query_text, str):
            raise ValueError(f"query_text must be a string, got {type(query_text).__name__}")
        
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty or only whitespace")
        
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"limit must be a positive integer, got {limit}")
        
        if not isinstance(min_similarity, (int, float)) or not (0.0 <= min_similarity <= 1.0):
            raise ValueError(f"min_similarity must be between 0.0 and 1.0, got {min_similarity}")
            raise ValueError("Query text cannot be empty")
        
        try:
            # Generate query embedding
            query_embedding = np.array(self.generate_embedding(query_text))
            
            # Get all embeddings from database
            embeddings = self.db.query(ChatEmbedding).join(
                PublishedChat
            ).filter(
                PublishedChat.is_public == True
            ).all()
            
            if not embeddings:
                return []
            
            # Calculate similarities
            similarities = []
            for emb in embeddings:
                embedding_vector = np.array(emb.embedding)
                # Cosine similarity
                similarity = np.dot(query_embedding, embedding_vector) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding_vector)
                )
                
                if similarity >= min_similarity:
                    similarities.append({
                        "chat_id": emb.chat_id,
                        "similarity": float(similarity),
                        "chat": emb.chat
                    })
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similarities[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar chats: {e}", exc_info=True)
            raise DatabaseError(f"Failed to find similar chats: {str(e)}")
    
    def get_embedding(self, chat_id: str) -> Optional[ChatEmbedding]:
        """Get embedding for a chat"""
        return self.db.query(ChatEmbedding).filter(
            ChatEmbedding.chat_id == chat_id
        ).first()
    
    def delete_embedding(self, chat_id: str) -> bool:
        """Delete embedding for a chat"""
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
            logger.error(f"Error deleting embedding: {e}", exc_info=True)
            raise DatabaseError(f"Failed to delete embedding: {str(e)}")





