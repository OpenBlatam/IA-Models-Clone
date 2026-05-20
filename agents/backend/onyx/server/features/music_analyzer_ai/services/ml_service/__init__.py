"""
ML Service Submodule
Aggregates ML service components.
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    from ...core.models import get_deep_analyzer, DeepMusicAnalyzer
    from ...core.ml_audio import get_ml_analyzer, MLMusicAnalyzer, AudioFeatureExtractor
    from ...core.processing import create_default_pipeline, ProcessingPipeline
    # Try to import transformer analyzer (may not exist)
    try:
        from ...core.transformer_analyzer import get_transformer_analyzer
        transformer_available = True
    except ImportError:
        try:
            from ...core.transformers import MusicTransformerEncoder
            transformer_available = True
            get_transformer_analyzer = lambda: MusicTransformerEncoder()
        except ImportError:
            transformer_available = False
            get_transformer_analyzer = None
            MusicTransformerEncoder = None
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML components not available")

from .analysis import AnalysisMixin
from .feature_extraction import FeatureExtractionMixin
from .comparison import ComparisonMixin


class MLService(AnalysisMixin, FeatureExtractionMixin, ComparisonMixin):
    """
    High-level ML service with multiple layers of abstraction:
    - Feature extraction
    - Model inference
    - Result aggregation
    - Caching
    """
    
    def __init__(self):
        if not ML_AVAILABLE:
            raise ImportError("ML components not available")
        
        self.deep_analyzer = get_deep_analyzer()
        self.ml_analyzer = get_ml_analyzer()
        if transformer_available and get_transformer_analyzer:
            self.transformer_analyzer = get_transformer_analyzer()
        else:
            self.transformer_analyzer = None
            logger.warning("Transformer analyzer not available")
        self.feature_extractor = AudioFeatureExtractor()
        self.pipeline = create_default_pipeline()
        
        # Cache for results
        self.cache: Dict[str, Any] = {}


# Global instance
_ml_service: Optional[MLService] = None


def get_ml_service() -> MLService:
    """Get or create ML service instance"""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service


__all__ = [
    "MLService",
    "AnalysisMixin",
    "FeatureExtractionMixin",
    "ComparisonMixin",
    "get_ml_service",
]

