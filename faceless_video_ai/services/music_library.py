"""
Music Library Service
Manages background music library
"""

from typing import List, Dict, Any, Optional
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MusicStyle(str, Enum):
    """Music style categories"""
    AMBIENT = "ambient"
    ENERGETIC = "energetic"
    CORPORATE = "corporate"
    FUN = "fun"
    TRENDY = "trendy"
    VIRAL = "viral"
    CALM = "calm"
    EPIC = "epic"
    UPBEAT = "upbeat"
    RELAXING = "relaxing"


class MusicTrack:
    """Represents a music track"""
    
    def __init__(
        self,
        track_id: str,
        name: str,
        style: MusicStyle,
        file_path: str,
        duration: float,
        bpm: Optional[int] = None,
        mood: Optional[str] = None
    ):
        self.track_id = track_id
        self.name = name
        self.style = style
        self.file_path = file_path
        self.duration = duration
        self.bpm = bpm
        self.mood = mood
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "track_id": self.track_id,
            "name": self.name,
            "style": self.style.value,
            "file_path": self.file_path,
            "duration": self.duration,
            "bpm": self.bpm,
            "mood": self.mood,
        }


class MusicLibrary:
    """Manages music library"""
    
    def __init__(self, music_dir: Optional[str] = None):
        self.music_dir = Path(music_dir) if music_dir else Path("/tmp/faceless_video/music")
        self.music_dir.mkdir(parents=True, exist_ok=True)
        self.tracks: Dict[str, MusicTrack] = {}
        self._load_library()
    
    def _load_library(self):
        """Load music library from directory"""
        # In production, load from database or config file
        # For now, create placeholder tracks
        
        placeholder_tracks = [
            {
                "track_id": "ambient_001",
                "name": "Ambient Background",
                "style": MusicStyle.AMBIENT,
                "file_path": str(self.music_dir / "ambient_001.mp3"),
                "duration": 300.0,
                "bpm": 60,
                "mood": "calm",
            },
            {
                "track_id": "energetic_001",
                "name": "Energetic Beat",
                "style": MusicStyle.ENERGETIC,
                "file_path": str(self.music_dir / "energetic_001.mp3"),
                "duration": 180.0,
                "bpm": 120,
                "mood": "energetic",
            },
            {
                "track_id": "corporate_001",
                "name": "Corporate Background",
                "style": MusicStyle.CORPORATE,
                "file_path": str(self.music_dir / "corporate_001.mp3"),
                "duration": 240.0,
                "bpm": 90,
                "mood": "professional",
            },
        ]
        
        for track_data in placeholder_tracks:
            track = MusicTrack(**track_data)
            self.tracks[track.track_id] = track
        
        logger.info(f"Loaded {len(self.tracks)} music tracks")
    
    def get_tracks_by_style(self, style: MusicStyle) -> List[MusicTrack]:
        """Get tracks by style"""
        return [track for track in self.tracks.values() if track.style == style]
    
    def get_track(self, track_id: str) -> Optional[MusicTrack]:
        """Get track by ID"""
        return self.tracks.get(track_id)
    
    def find_track(
        self,
        style: Optional[MusicStyle] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        bpm: Optional[int] = None
    ) -> Optional[MusicTrack]:
        """
        Find track matching criteria
        
        Args:
            style: Music style
            min_duration: Minimum duration
            max_duration: Maximum duration
            bpm: BPM range
            
        Returns:
            Matching track or None
        """
        candidates = list(self.tracks.values())
        
        if style:
            candidates = [t for t in candidates if t.style == style]
        
        if min_duration:
            candidates = [t for t in candidates if t.duration >= min_duration]
        
        if max_duration:
            candidates = [t for t in candidates if t.duration <= max_duration]
        
        if bpm:
            # Find closest BPM
            candidates = sorted(candidates, key=lambda t: abs((t.bpm or 0) - bpm))
        
        return candidates[0] if candidates else None
    
    def list_tracks(self) -> List[Dict[str, Any]]:
        """List all tracks"""
        return [track.to_dict() for track in self.tracks.values()]
    
    def add_track(
        self,
        name: str,
        style: MusicStyle,
        file_path: str,
        duration: float,
        bpm: Optional[int] = None,
        mood: Optional[str] = None
    ) -> MusicTrack:
        """Add new track to library"""
        track_id = f"{style.value}_{len([t for t in self.tracks.values() if t.style == style]) + 1:03d}"
        
        track = MusicTrack(
            track_id=track_id,
            name=name,
            style=style,
            file_path=file_path,
            duration=duration,
            bpm=bpm,
            mood=mood
        )
        
        self.tracks[track_id] = track
        logger.info(f"Added track to library: {track_id}")
        
        return track


_music_library: Optional[MusicLibrary] = None


def get_music_library(music_dir: Optional[str] = None) -> MusicLibrary:
    """Get music library instance (singleton)"""
    global _music_library
    if _music_library is None:
        _music_library = MusicLibrary(music_dir=music_dir)
    return _music_library

