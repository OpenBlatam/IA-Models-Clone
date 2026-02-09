"""
Analyzer Factory - Create analyzer instances
"""

from typing import Dict, Any, Optional, Type
import logging

from ..interfaces.analyzer_interface import IMusicAnalyzer, IFeatureExtractor

# Lazy imports to avoid circular dependencies
try:
    from ..core.deep_models import DeepMusicAnalyzer
except ImportError:
    DeepMusicAnalyzer = None

try:
    from ..core.transformer_analyzer import TransformerMusicAnalyzer
except ImportError:
    TransformerMusicAnalyzer = None

try:
    from ..core.ml_audio_analyzer import MLMusicAnalyzer
except ImportError:
    MLMusicAnalyzer = None

logger = logging.getLogger(__name__)


class AnalyzerFactory:
    """
    Factory for creating analyzer instances
    """
    
    _analyzer_registry: Dict[str, Type[IMusicAnalyzer]] = {}
    
    @classmethod
    def register(cls, name: str, analyzer_class: Type[IMusicAnalyzer]):
        """Register an analyzer class"""
        cls._analyzer_registry[name] = analyzer_class
        logger.info(f"Registered analyzer: {name}")
    
    @classmethod
    def create(
        cls,
        analyzer_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> IMusicAnalyzer:
        """
        Create analyzer instance
        
        Args:
            analyzer_type: Type of analyzer
            config: Analyzer configuration
        
        Returns:
            Analyzer instance
        """
        if analyzer_type not in cls._analyzer_registry:
            raise ValueError(f"Unknown analyzer type: {analyzer_type}")
        
        if config is None:
            config = {}
        
        analyzer_class = cls._analyzer_registry[analyzer_type]
        analyzer = analyzer_class(**config)
        
        logger.info(f"Created {analyzer_type} analyzer")
        return analyzer
    
    @classmethod
    def list_available(cls) -> list:
        """List available analyzer types"""
        return list(cls._analyzer_registry.keys())


# Register default analyzers (if available)
if DeepMusicAnalyzer is not None:
    AnalyzerFactory.register("deep", DeepMusicAnalyzer)
if TransformerMusicAnalyzer is not None:
    AnalyzerFactory.register("transformer", TransformerMusicAnalyzer)
if MLMusicAnalyzer is not None:
    AnalyzerFactory.register("ml", MLMusicAnalyzer)


def create_analyzer(
    analyzer_type: str = "deep",
    config: Optional[Dict[str, Any]] = None
) -> IMusicAnalyzer:
    """
    Convenience function to create an analyzer
    
    Args:
        analyzer_type: Type of analyzer
        config: Analyzer configuration
    
    Returns:
        Analyzer instance
    """
    return AnalyzerFactory.create(analyzer_type, config)

