"""
Spotify Adapter - Adapt Spotify API responses
"""

from typing import Any, Dict
import logging

from .adapter import BaseAdapter

logger = logging.getLogger(__name__)


class SpotifyAdapter(BaseAdapter):
    """
    Adapter for Spotify API responses to internal format
    """
    
    def __init__(self):
        super().__init__("spotify_api", "internal_format")
    
    def adapt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt Spotify response to internal format"""
        try:
            # Extract audio features
            audio_features = data.get("audio_features", {})
            
            # Adapt to internal format
            adapted = {
                "track_id": data.get("id"),
                "name": data.get("name"),
                "artist": data.get("artists", [{}])[0].get("name") if data.get("artists") else None,
                "features": {
                    "danceability": audio_features.get("danceability", 0.0),
                    "energy": audio_features.get("energy", 0.0),
                    "valence": audio_features.get("valence", 0.0),
                    "tempo": audio_features.get("tempo", 0.0),
                    "loudness": audio_features.get("loudness", 0.0),
                    "acousticness": audio_features.get("acousticness", 0.0),
                    "instrumentalness": audio_features.get("instrumentalness", 0.0),
                    "liveness": audio_features.get("liveness", 0.0),
                    "speechiness": audio_features.get("speechiness", 0.0),
                    "key": audio_features.get("key", 0),
                    "mode": audio_features.get("mode", 0),
                    "time_signature": audio_features.get("time_signature", 4)
                },
                "popularity": data.get("popularity", 0),
                "duration_ms": data.get("duration_ms", 0)
            }
            
            return adapted
        
        except Exception as e:
            logger.error(f"Error adapting Spotify data: {str(e)}")
            raise








