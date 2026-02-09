"""
Feature Store for Recovery AI
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import pickle
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FeatureStore:
    """Feature store for storing and retrieving features"""
    
    def __init__(self, store_dir: str = "feature_store"):
        """
        Initialize feature store
        
        Args:
            store_dir: Store directory
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True)
        
        self.features = {}
        self.metadata = {}
        
        logger.info(f"FeatureStore initialized: {store_dir}")
    
    def store_features(
        self,
        feature_id: str,
        features: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Store features
        
        Args:
            feature_id: Feature identifier
            features: Feature dictionary
            metadata: Optional metadata
        """
        self.features[feature_id] = {
            "features": features,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Persist to disk
        self._save_features(feature_id)
    
    def get_features(self, feature_id: str) -> Optional[Dict[str, Any]]:
        """Get features"""
        if feature_id in self.features:
            return self.features[feature_id]
        
        # Try to load from disk
        return self._load_features(feature_id)
    
    def _save_features(self, feature_id: str):
        """Save features to disk"""
        if feature_id not in self.features:
            return
        
        filepath = self.store_dir / f"{feature_id}.json"
        try:
            with open(filepath, 'w') as f:
                json.dump(self.features[feature_id], f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save features: {e}")
    
    def _load_features(self, feature_id: str) -> Optional[Dict[str, Any]]:
        """Load features from disk"""
        filepath = self.store_dir / f"{feature_id}.json"
        
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                self.features[feature_id] = data
                return data
            except Exception as e:
                logger.warning(f"Failed to load features: {e}")
        
        return None
    
    def get_feature_history(
        self,
        user_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get feature history for user
        
        Args:
            user_id: User identifier
            days: Number of days
        
        Returns:
            List of feature records
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        history = []
        for feature_id, data in self.features.items():
            if feature_id.startswith(user_id):
                timestamp = datetime.fromisoformat(data["timestamp"])
                if timestamp >= cutoff:
                    history.append(data)
        
        return sorted(history, key=lambda x: x["timestamp"])


class EmbeddingStore:
    """Store for embeddings"""
    
    def __init__(self, store_dir: str = "embedding_store"):
        """
        Initialize embedding store
        
        Args:
            store_dir: Store directory
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True)
        
        self.embeddings = {}
        logger.info(f"EmbeddingStore initialized: {store_dir}")
    
    def store_embedding(
        self,
        key: str,
        embedding: torch.Tensor
    ):
        """Store embedding"""
        self.embeddings[key] = embedding.cpu()
        
        # Persist
        filepath = self.store_dir / f"{key}.pth"
        torch.save(embedding.cpu(), filepath)
    
    def get_embedding(self, key: str) -> Optional[torch.Tensor]:
        """Get embedding"""
        if key in self.embeddings:
            return self.embeddings[key]
        
        # Try to load from disk
        filepath = self.store_dir / f"{key}.pth"
        if filepath.exists():
            try:
                embedding = torch.load(filepath)
                self.embeddings[key] = embedding
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load embedding: {e}")
        
        return None
    
    def search_similar(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 10
    ) -> List[tuple]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results
        
        Returns:
            List of (key, similarity_score) tuples
        """
        similarities = []
        
        for key, embedding in self.embeddings.items():
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                embedding.unsqueeze(0)
            ).item()
            
            similarities.append((key, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

