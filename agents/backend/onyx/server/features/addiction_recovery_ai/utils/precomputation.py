"""
Precomputation and Embedding Caching for Faster Inference
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from functools import lru_cache
import pickle
import os
import hashlib
import logging

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for precomputed embeddings"""
    
    def __init__(self, cache_dir: str = ".cache", max_size: int = 10000):
        """
        Initialize embedding cache
        
        Args:
            cache_dir: Directory for cache files
            max_size: Maximum cache size
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.cache = {}
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"EmbeddingCache initialized: {cache_dir}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[torch.Tensor]:
        """Get cached embedding"""
        key = self._get_cache_key(text)
        if key in self.cache:
            return self.cache[key]
        
        # Try to load from disk
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                self.cache[key] = embedding
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def set(self, text: str, embedding: torch.Tensor):
        """Set cached embedding"""
        key = self._get_cache_key(text)
        
        # Check cache size
        if len(self.cache) >= self.max_size:
            # Remove oldest (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = embedding
        
        # Save to disk
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        logger.info("Embedding cache cleared")


class FeaturePreprocessor:
    """Preprocessor for feature extraction and caching"""
    
    def __init__(self, cache: Optional[EmbeddingCache] = None):
        """
        Initialize feature preprocessor
        
        Args:
            cache: Embedding cache
        """
        self.cache = cache or EmbeddingCache()
    
    def preprocess_features(
        self,
        features: Dict[str, float]
    ) -> torch.Tensor:
        """
        Preprocess features to tensor
        
        Args:
            features: Feature dictionary
        
        Returns:
            Preprocessed tensor
        """
        feature_list = [
            features.get("days_sober", 0) / 365.0,
            features.get("cravings_level", 5) / 10.0,
            features.get("stress_level", 5) / 10.0,
            features.get("support_level", 5) / 10.0,
            features.get("mood_score", 5) / 10.0,
            features.get("sleep_quality", 5) / 10.0,
            features.get("exercise_frequency", 2) / 7.0,
            features.get("therapy_sessions", 0) / 10.0,
            features.get("medication_compliance", 1.0),
            features.get("social_activity", 3) / 7.0
        ]
        
        return torch.tensor([feature_list], dtype=torch.float32)
    
    def preprocess_sequence(
        self,
        sequence: List[Dict[str, float]],
        max_length: int = 30
    ) -> torch.Tensor:
        """
        Preprocess sequence to tensor
        
        Args:
            sequence: Sequence of daily features
            max_length: Maximum sequence length
        
        Returns:
            Preprocessed tensor
        """
        seq_data = []
        for day in sequence[-max_length:]:
            seq_data.append([
                day.get("cravings_level", 5) / 10.0,
                day.get("stress_level", 5) / 10.0,
                day.get("mood_score", 5) / 10.0,
                day.get("triggers_count", 0) / 10.0,
                day.get("consumed", 0.0)
            ])
        
        # Pad to fixed length
        while len(seq_data) < max_length:
            seq_data.insert(0, [0.0] * 5)
        
        return torch.tensor([seq_data], dtype=torch.float32)


class BatchPreprocessor:
    """Batch preprocessor for efficient batch processing"""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize batch preprocessor
        
        Args:
            device: Device to use
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_preprocessor = FeaturePreprocessor()
    
    def preprocess_batch(
        self,
        items: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> List[torch.Tensor]:
        """
        Preprocess batch of items
        
        Args:
            items: List of items to preprocess
            batch_size: Batch size
        
        Returns:
            List of preprocessed tensors
        """
        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Preprocess each item
            preprocessed = []
            for item in batch:
                if "features" in item:
                    tensor = self.feature_preprocessor.preprocess_features(item["features"])
                elif "sequence" in item:
                    tensor = self.feature_preprocessor.preprocess_sequence(item["sequence"])
                else:
                    raise ValueError("Item must have 'features' or 'sequence'")
                
                preprocessed.append(tensor)
            
            # Stack into batch
            batch_tensor = torch.cat(preprocessed, dim=0).to(self.device)
            batches.append(batch_tensor)
        
        return batches


def precompute_embeddings(
    texts: List[str],
    model,
    tokenizer,
    batch_size: int = 32,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Precompute embeddings for texts
    
    Args:
        texts: List of texts
        model: Embedding model
        tokenizer: Tokenizer
        batch_size: Batch size
        device: Device to use
    
    Returns:
        Dictionary of text -> embedding
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    embeddings = {}
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(device)
            
            # Get embeddings
            outputs = model(**inputs)
            if hasattr(outputs, 'last_hidden_state'):
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                batch_embeddings = outputs.logits
            
            # Store
            for j, text in enumerate(batch_texts):
                embeddings[text] = batch_embeddings[j].cpu()
    
    logger.info(f"Precomputed {len(embeddings)} embeddings")
    return embeddings

