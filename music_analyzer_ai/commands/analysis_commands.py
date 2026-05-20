"""
Analysis Commands - Commands for analysis operations
"""

from typing import Any, List, Dict, Optional
import logging

from .command import ICommand

logger = logging.getLogger(__name__)


class AnalyzeTrackCommand(ICommand):
    """
    Command to analyze a track
    """
    
    def __init__(self, analyzer, track_id: str, features: Optional[Dict] = None):
        self.analyzer = analyzer
        self.track_id = track_id
        self.features = features
        self._result = None
    
    @property
    def name(self) -> str:
        return "AnalyzeTrack"
    
    def execute(self) -> Dict[str, Any]:
        """Execute analysis"""
        if hasattr(self.analyzer, 'analyze'):
            self._result = self.analyzer.analyze(self.track_id, self.features)
        elif hasattr(self.analyzer, 'predict_genre'):
            self._result = self.analyzer.predict_genre(self.features)
        else:
            raise ValueError("Analyzer does not support analysis")
        
        return self._result
    
    def undo(self) -> Any:
        """Analysis cannot be undone"""
        logger.warning("Analysis cannot be undone")
        return None


class BatchAnalyzeCommand(ICommand):
    """
    Command to analyze multiple tracks
    """
    
    def __init__(self, analyzer, track_ids: List[str]):
        self.analyzer = analyzer
        self.track_ids = track_ids
        self._results = []
    
    @property
    def name(self) -> str:
        return "BatchAnalyze"
    
    def execute(self) -> List[Dict[str, Any]]:
        """Execute batch analysis"""
        results = []
        for track_id in self.track_ids:
            try:
                if hasattr(self.analyzer, 'analyze'):
                    result = self.analyzer.analyze(track_id)
                else:
                    result = {"track_id": track_id, "error": "Analyzer not supported"}
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {track_id}: {str(e)}")
                results.append({"track_id": track_id, "error": str(e)})
        
        self._results = results
        return results
    
    def undo(self) -> Any:
        """Batch analysis cannot be undone"""
        logger.warning("Batch analysis cannot be undone")
        return None








