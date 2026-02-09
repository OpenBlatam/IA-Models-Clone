"""
Analyzer Interfaces - Define contracts for analysis components
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np


class IMusicAnalyzer(ABC):
    """
    Base interface for music analyzers
    """
    
    @abstractmethod
    def analyze(self, audio_data: Any) -> Dict[str, Any]:
        """Analyze audio data"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get supported audio formats"""
        pass


class IFeatureExtractor(ABC):
    """
    Interface for feature extractors
    """
    
    @abstractmethod
    def extract(self, audio_data: Any) -> np.ndarray:
        """Extract features from audio"""
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Get feature dimension"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        pass


class IEmbeddingExtractor(IFeatureExtractor):
    """
    Interface for embedding extractors
    """
    
    @abstractmethod
    def extract_embeddings(self, audio_data: Any) -> np.ndarray:
        """Extract embeddings"""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        pass













