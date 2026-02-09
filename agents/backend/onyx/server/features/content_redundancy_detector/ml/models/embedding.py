"""
Embedding Model for Semantic Similarity
Uses Sentence Transformers for high-quality embeddings
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseModel

logger = logging.getLogger(__name__)


class EmbeddingModel(BaseModel):
    """
    Sentence Transformer model for generating semantic embeddings
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = False,
    ):
        """
        Initialize embedding model
        
        Args:
            model_name: HuggingFace model identifier
            device: PyTorch device
            use_mixed_precision: Use mixed precision for inference
        """
        super().__init__(model_name, device, use_mixed_precision)
        self.transformer_model_name = model_name
    
    async def load(self) -> None:
        """Load Sentence Transformer model"""
        if self.is_loaded:
            return
        
        try:
            logger.info(f"Loading Sentence Transformer: {self.transformer_model_name}")
            self.model = SentenceTransformer(
                self.transformer_model_name,
                device=str(self.device)
            )
            self.is_loaded = True
            logger.info(f"Successfully loaded {self.transformer_model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    async def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for input texts
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if not self.is_loaded:
            await self.load()
        
        if isinstance(texts, str):
            texts = [texts]
        
        with torch.no_grad():
            if self.use_mixed_precision and self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    embeddings = self.model.encode(
                        texts,
                        batch_size=batch_size,
                        normalize_embeddings=normalize,
                        show_progress_bar=show_progress,
                        convert_to_numpy=True,
                    )
            else:
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=normalize,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                )
        
        return embeddings
    
    async def predict(self, inputs: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Generate embeddings (alias for encode for BaseModel compatibility)
        
        Args:
            inputs: Text or list of texts
            
        Returns:
            Dictionary with embeddings
        """
        embeddings = await self.encode(inputs)
        return {
            "embeddings": embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings,
            "shape": list(embeddings.shape) if isinstance(embeddings, np.ndarray) else None,
        }
    
    async def compute_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """
        Compute cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        embeddings = await self.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    async def compute_batch_similarity(
        self,
        texts1: List[str],
        texts2: List[str],
    ) -> np.ndarray:
        """
        Compute similarity matrix between two lists of texts
        
        Args:
            texts1: First list of texts
            texts2: Second list of texts
            
        Returns:
            Similarity matrix
        """
        embeddings1 = await self.encode(texts1)
        embeddings2 = await self.encode(texts2)
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        return similarity_matrix

