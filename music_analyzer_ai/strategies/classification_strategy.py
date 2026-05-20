"""
Classification Strategies
"""

from typing import Any, Dict
import logging

from .strategy import IStrategy

logger = logging.getLogger(__name__)


class ClassificationStrategy(IStrategy):
    """Base strategy for classification"""
    
    @property
    def name(self) -> str:
        return "BaseClassification"


class NeuralStrategy(ClassificationStrategy):
    """Classification using neural networks"""
    
    @property
    def name(self) -> str:
        return "NeuralClassification"
    
    def execute(self, data: Any) -> Dict[str, Any]:
        """Classify using neural network"""
        try:
            from ..core.deep_models import DeepMusicAnalyzer
            
            analyzer = DeepMusicAnalyzer(device="cuda")
            result = analyzer.predict_genre(data)
            return result
        
        except Exception as e:
            logger.error(f"Neural classification failed: {str(e)}")
            raise


class MLStrategy(ClassificationStrategy):
    """Classification using traditional ML"""
    
    @property
    def name(self) -> str:
        return "MLClassification"
    
    def execute(self, data: Any) -> Dict[str, Any]:
        """Classify using ML models"""
        try:
            from ..core.ml_audio_analyzer import MLMusicAnalyzer
            
            analyzer = MLMusicAnalyzer()
            result = analyzer.predict_genre(data)
            return result
        
        except Exception as e:
            logger.error(f"ML classification failed: {str(e)}")
            raise








