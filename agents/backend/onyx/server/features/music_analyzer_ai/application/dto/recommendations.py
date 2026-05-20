"""
Recommendation DTOs

Data Transfer Objects for recommendations.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class RecommendationDTO:
    """DTO for a single recommendation"""
    track_id: str
    track_name: str
    artists: List[str]
    similarity_score: Optional[float] = None
    reason: Optional[str] = None
    album: Optional[str] = None
    preview_url: Optional[str] = None
    popularity: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "track_id": self.track_id,
            "track_name": self.track_name,
            "artists": self.artists,
            "similarity_score": self.similarity_score,
            "reason": self.reason,
            "album": self.album,
            "preview_url": self.preview_url,
            "popularity": self.popularity
        }


@dataclass
class PlaylistDTO:
    """DTO for a generated playlist"""
    tracks: List[RecommendationDTO]
    total_tracks: int
    criteria: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tracks": [track.to_dict() for track in self.tracks],
            "total_tracks": self.total_tracks,
            "criteria": self.criteria
        }




