"""
Base Analyzer - Shared base implementation for all analyzers
"""

from typing import Dict, Any, List
import logging

from ..interfaces.analyzer_interface import IMusicAnalyzer

logger = logging.getLogger(__name__)


class BaseMusicAnalyzer(IMusicAnalyzer):
    """
    Base class for all music analyzers with common functionality
    """
    
    def __init__(self, name: str = "BaseMusicAnalyzer"):
        self.name = name
        self.supported_formats = ["wav", "mp3", "flac"]
        self._cache: Dict[str, Any] = {}
    
    def analyze(self, audio_data: Any) -> Dict[str, Any]:
        """Base analyze implementation"""
        # Subclasses should implement actual analysis logic
        return {
            "analyzer": self.name,
            "status": "success"
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get supported audio formats"""
        return self.supported_formats
    
    def _cache_result(self, key: str, result: Any):
        """Cache analysis result"""
        self._cache[key] = result
    
    def _get_cached_result(self, key: str) -> Any:
        """Get cached result"""
        return self._cache.get(key)
    
    def clear_cache(self):
        """Clear analysis cache"""
        self._cache.clear()
        logger.info(f"Cache cleared for {self.name}")













