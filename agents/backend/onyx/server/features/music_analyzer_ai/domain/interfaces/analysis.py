"""
Analysis Service Interfaces

Defines contracts for music analysis services.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class IAnalysisService(ABC):
    """Interface for music analysis service"""
    
    @abstractmethod
    async def analyze_track(self, spotify_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a track with complete Spotify data.
        
        Args:
            spotify_data: Dictionary containing track_info, audio_features, and audio_analysis
        
        Returns:
            Complete analysis dictionary with musical, technical, and composition analysis
        """
        pass
    
    @abstractmethod
    async def analyze_tracks_batch(
        self,
        tracks_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple tracks in batch.
        
        Args:
            tracks_data: List of track data dictionaries
        
        Returns:
            List of analysis dictionaries
        """
        pass


class IHarmonicAnalyzer(ABC):
    """Interface for harmonic analysis"""
    
    @abstractmethod
    async def analyze_harmony(
        self,
        audio_analysis: Dict[str, Any],
        key: int,
        mode: int
    ) -> Dict[str, Any]:
        """
        Analyze harmonic progression and patterns.
        
        Args:
            audio_analysis: Audio analysis data from Spotify
            key: Key signature (0-11)
            mode: Mode (0=minor, 1=major)
        
        Returns:
            Harmonic analysis with chord progressions, cadences, etc.
        """
        pass
    
    @abstractmethod
    async def detect_chord_progressions(
        self,
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect chord progressions from audio segments.
        
        Args:
            segments: List of audio segments with pitch data
        
        Returns:
            List of detected chord progressions
        """
        pass


class IStructureAnalyzer(ABC):
    """Interface for musical structure analysis"""
    
    @abstractmethod
    async def analyze_structure(
        self,
        audio_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze the structure of a track (sections, transitions, etc.).
        
        Args:
            audio_analysis: Audio analysis data from Spotify
        
        Returns:
            Structure analysis with sections, transitions, build-ups, drops
        """
        pass
    
    @abstractmethod
    async def detect_sections(
        self,
        sections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect and categorize musical sections.
        
        Args:
            sections: List of section data from Spotify
        
        Returns:
            List of categorized sections (intro, verse, chorus, etc.)
        """
        pass


class IEmotionAnalyzer(ABC):
    """Interface for emotion analysis"""
    
    @abstractmethod
    async def analyze_emotions(
        self,
        audio_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze emotional characteristics of a track.
        
        Args:
            audio_features: Audio features from Spotify
        
        Returns:
            Emotion analysis with primary emotions and emotional profile
        """
        pass
    
    @abstractmethod
    async def detect_primary_emotion(
        self,
        audio_features: Dict[str, Any]
    ) -> str:
        """
        Detect the primary emotion of a track.
        
        Args:
            audio_features: Audio features from Spotify
        
        Returns:
            Primary emotion (happy, sad, energetic, calm, etc.)
        """
        pass


class IGenreDetector(ABC):
    """Interface for genre detection"""
    
    @abstractmethod
    async def detect_genre(
        self,
        audio_features: Dict[str, Any],
        audio_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect the genre of a track.
        
        Args:
            audio_features: Audio features from Spotify
            audio_analysis: Optional detailed audio analysis
        
        Returns:
            Genre detection with primary genre and confidence
        """
        pass
    
    @abstractmethod
    async def detect_genres_batch(
        self,
        tracks_features: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect genres for multiple tracks.
        
        Args:
            tracks_features: List of audio features dictionaries
        
        Returns:
            List of genre detection results
        """
        pass




