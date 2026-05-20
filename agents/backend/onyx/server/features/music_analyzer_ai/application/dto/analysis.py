"""
Analysis DTOs

Data Transfer Objects for analysis results.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class TrackAnalysisDTO:
    """DTO for track analysis data"""
    track_id: str
    track_name: str
    artists: List[str]
    album: Optional[str] = None
    duration_ms: Optional[int] = None
    preview_url: Optional[str] = None
    popularity: Optional[int] = None


@dataclass
class AnalysisResultDTO:
    """DTO for complete analysis result"""
    track_id: str
    track_name: str
    artists: List[str]
    analysis: Dict[str, Any]
    coaching: Optional[Dict[str, Any]] = None
    album: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "track_id": self.track_id,
            "track_name": self.track_name,
            "artists": self.artists,
            "album": self.album,
            "duration_seconds": self.duration_seconds,
            "analysis": self.analysis,
            "coaching": self.coaching
        }




