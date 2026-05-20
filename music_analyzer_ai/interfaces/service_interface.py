"""
Service Interfaces - Define contracts for services
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class IService(ABC):
    """
    Base interface for all services
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize service"""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown service"""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        pass


class IFeatureService(IService):
    """
    Interface for feature extraction services
    """
    
    @abstractmethod
    def extract_features(self, audio_data: Any) -> Dict[str, Any]:
        """Extract features from audio"""
        pass
    
    @abstractmethod
    def get_available_features(self) -> List[str]:
        """Get list of available features"""
        pass


class IAnalysisService(IService):
    """
    Interface for analysis services
    """
    
    @abstractmethod
    def analyze(self, track_id: str, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze a track"""
        pass
    
    @abstractmethod
    def batch_analyze(self, track_ids: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple tracks"""
        pass


class IRecommendationService(IService):
    """
    Interface for recommendation services
    """
    
    @abstractmethod
    def recommend(self, track_id: str, num_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Get recommendations for a track"""
        pass
    
    @abstractmethod
    def recommend_by_features(self, features: Dict[str, Any], num_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Get recommendations by features"""
        pass









