"""
Refactored Embedding Service with best practices

Uses sentence-transformers with proper:
- Device management
- Batch processing
- Error handling
- Performance optimization
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

from ...config import settings
from ...models import ChatEmbedding, PublishedChat
from ...exceptions import DatabaseError
from ...helpers import generate_id, handle_db_error
from .base_service import BaseAIService
from .data_loader import BatchProcessor, preprocess_text

logger = logging.getLogger(__name__)


class EmbeddingService(BaseAIService):
    """
    Service for generating and managing text embeddings
    
    Uses sentence-transformers with proper batch processing,
    device management, and error handling.
    """
    
    def __init__(self, db: Session):
        """
        Initialize embedding service
        
        Args:
            db: Database session
        """
        super().__init__(
            model_name=settings.embedding_model,
            model_type="sentence_transformer"
        )
        self.db = db
        self.batch_processor = BatchProcessor(
            batch_size=settings.batch_size_embeddings,
            device=self.device
        )
        self._load_model()
    
    def _load_model_impl(self) -> None:
        """Load the sentence transformer model"""
        if not settings.ai_enabled:
            logger.warning("AI features are disabled")
            return
        
        try:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            
            # Use device string for sentence-transformers
            device_str = str(self.device) if self.device.type == "cuda" else "cpu"
            
            self.model = SentenceTransformer(
                self.model_name,
                device=device_str,
                cache_folder=settings.model_cache_dir
            )
            
            # Move to device explicitly
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            
            # Set to eval mode for inference
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            logger.info(f"Embedding model loaded. Dimension: {self.get_embedding_dimension()}")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}", exc_info=True)
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if self.model is None:
            return settings.embedding_dimension
        
        # Try to get dimension from model
        try:
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                return self.model.get_sentence_embedding_dimension()
            elif hasattr(self.model, 'max_seq_length'):
                # Estimate from model config
                return 384  # Default for MiniLM models
        except:
            pass
        
        return settings.embedding_dimension
    
    def generate_embedding(
        self,
        text: str,
        normalize: bool = True,
        show_progress: bool = False
    ) -> List[float]:
        """
        Generate embedding for a text string
        
        Args:
            text: Input text to embed
            normalize: Whether to normalize the embedding
            show_progress: Whether to show progress bar
            
        Returns:
            List of floats representing the embedding vector
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded. AI features may be disabled.")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            # Preprocess text
            processed_text = preprocess_text(text, max_length=512)
            
            with self.inference_context():
                embedding = self.model.encode(
                    processed_text,
                    convert_to_numpy=True,
                    show_progress_bar=show_progress,
                    batch_size=1,
                    normalize_embeddings=normalize
                )
            
            # Check for NaN/Inf
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                logger.warning("Embedding contains NaN or Inf values")
                embedding = np.nan_to_num(embedding, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)
            raise
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch
        
        Args:
            texts: List of texts to embed
            normalize: Whether to normalize embeddings
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        if not texts:
            return []
        
        try:
            # Preprocess texts
            processed_texts = [
                preprocess_text(text, max_length=512)
                for text in texts
            ]
            
            with self.inference_context():
                embeddings = self.model.encode(
                    processed_texts,
                    convert_to_numpy=True,
                    show_progress_bar=show_progress,
                    batch_size=settings.batch_size_embeddings,
                    normalize_embeddings=normalize
                )
            
            # Check for NaN/Inf
            if np.isnan(embeddings).any() or np.isinf(embeddings).any():
                logger.warning("Embeddings contain NaN or Inf values")
                embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}", exc_info=True)
            raise
    
    def create_or_update_embedding(
        self,
        chat_id: str,
        text: Optional[str] = None,
        chat: Optional[PublishedChat] = None,
        force_update: bool = False
    ) -> ChatEmbedding:
        """
        Create or update embedding for a chat
        
        Args:
            chat_id: ID of the chat
            text: Optional text to embed (if not provided, uses chat content)
            chat: Optional chat object (if not provided, fetches from DB)
            force_update: Force update even if embedding exists
            
        Returns:
            ChatEmbedding object
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
            
            # Check if embedding exists and model hasn't changed
            existing = self.db.query(ChatEmbedding).filter(
                ChatEmbedding.chat_id == chat_id
            ).first()
            
            if existing and not force_update:
                if existing.embedding_model == settings.embedding_model:
                    logger.debug(f"Embedding already exists for chat {chat_id}")
                    return existing
            
            # Prepare text for embedding
            if not text:
                text_parts = [chat.title]
                if chat.description:
                    text_parts.append(chat.description)
                text_parts.append(chat.chat_content)
                text = " ".join(text_parts)
            
            # Generate embedding
            embedding_vector = self.generate_embedding(text)
            
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
        min_similarity: float = 0.5,
        use_faiss: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Find similar chats using semantic search
        
        Args:
            query_text: Query text to search for
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)
            use_faiss: Whether to use FAISS for faster search (if available)
            
        Returns:
            List of dicts with chat info and similarity scores
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")
        
        try:
            # Generate query embedding
            query_embedding = np.array(self.generate_embedding(query_text))
            query_norm = np.linalg.norm(query_embedding)
            
            # Get all embeddings from database
            embeddings = self.db.query(ChatEmbedding).join(
                PublishedChat
            ).filter(
                PublishedChat.is_public == True
            ).all()
            
            if not embeddings:
                return []
            
            # Use FAISS for faster search if available
            if use_faiss:
                try:
                    import faiss
                    return self._search_with_faiss(
                        query_embedding, embeddings, limit, min_similarity
                    )
                except ImportError:
                    logger.warning("FAISS not available, using numpy search")
            
            # Calculate similarities using numpy (vectorized)
            embedding_vectors = np.array([emb.embedding for emb in embeddings])
            embedding_norms = np.linalg.norm(embedding_vectors, axis=1)
            
            # Cosine similarity (vectorized)
            similarities = np.dot(embedding_vectors, query_embedding) / (
                embedding_norms * query_norm
            )
            
            # Filter and sort
            valid_indices = np.where(similarities >= min_similarity)[0]
            sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
            
            results = []
            for idx in sorted_indices[:limit]:
                emb = embeddings[idx]
                results.append({
                    "chat_id": emb.chat_id,
                    "similarity": float(similarities[idx]),
                    "chat": emb.chat
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar chats: {e}", exc_info=True)
            raise DatabaseError(f"Failed to find similar chats: {str(e)}")
    
    def _search_with_faiss(
        self,
        query_embedding: np.ndarray,
        embeddings: List[ChatEmbedding],
        limit: int,
        min_similarity: float
    ) -> List[Dict[str, Any]]:
        """Use FAISS for faster similarity search"""
        import faiss
        
        dim = len(query_embedding)
        index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        embedding_vectors = np.array([emb.embedding for emb in embeddings])
        faiss.normalize_L2(embedding_vectors)
        index.add(embedding_vectors.astype('float32'))
        
        # Normalize query
        query_normalized = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_normalized)
        
        # Search
        similarities, indices = index.search(query_normalized, limit * 2)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim >= min_similarity:
                emb = embeddings[idx]
                results.append({
                    "chat_id": emb.chat_id,
                    "similarity": float(sim),
                    "chat": emb.chat
                })
                if len(results) >= limit:
                    break
        
        return results
    
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
    
    def batch_create_embeddings(
        self,
        chat_ids: List[str],
        show_progress: bool = True
    ) -> Dict[str, ChatEmbedding]:
        """
        Create embeddings for multiple chats in batch
        
        Args:
            chat_ids: List of chat IDs
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping chat_id to ChatEmbedding
        """
        results = {}
        
        # Fetch all chats
        chats = self.db.query(PublishedChat).filter(
            PublishedChat.id.in_(chat_ids)
        ).all()
        
        if not chats:
            return results
        
        # Prepare texts
        texts = []
        chat_map = {}
        for chat in chats:
            text_parts = [chat.title]
            if chat.description:
                text_parts.append(chat.description)
            text_parts.append(chat.chat_content)
            text = " ".join(text_parts)
            texts.append(text)
            chat_map[text] = chat
        
        # Generate embeddings in batch
        if show_progress:
            iterator = tqdm(range(0, len(texts), settings.batch_size_embeddings), desc="Creating embeddings")
        else:
            iterator = range(0, len(texts), settings.batch_size_embeddings)
        
        for i in iterator:
            batch_texts = texts[i:i + settings.batch_size_embeddings]
            batch_chats = [chat_map[text] for text in batch_texts]
            
            embeddings = self.generate_embeddings_batch(batch_texts, show_progress=False)
            
            # Save embeddings
            for chat, embedding_vector in zip(batch_chats, embeddings):
                try:
                    embedding = self.create_or_update_embedding(
                        chat.id,
                        text=None,
                        chat=chat,
                        force_update=True
                    )
                    results[chat.id] = embedding
                except Exception as e:
                    logger.warning(f"Failed to create embedding for chat {chat.id}: {e}")
        
        return results





