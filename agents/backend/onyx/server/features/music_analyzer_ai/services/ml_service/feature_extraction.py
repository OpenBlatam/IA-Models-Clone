"""
ML Service Feature Extraction Module

Feature extraction and vector creation.
"""

from typing import Dict, Any, Optional
import numpy as np


class FeatureExtractionMixin:
    """Feature extraction mixin for MLService."""
    
    def _create_feature_vector(
        self,
        audio_features: Any,
        spotify_features: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Create feature vector from audio and Spotify features.
        
        Args:
            audio_features: Audio features object
            spotify_features: Spotify API features dictionary
        
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Audio features
        if hasattr(audio_features, "mfcc"):
            features.extend(audio_features.mfcc.mean(axis=1))
            features.extend(audio_features.chroma.mean(axis=1))
            features.extend(audio_features.spectral_contrast.mean(axis=1))
            features.extend(audio_features.tonnetz.mean(axis=1))
            features.append(audio_features.tempo)
        else:
            features = audio_features
        
        # Spotify features
        if spotify_features:
            features.append(spotify_features.get("energy", 0.5))
            features.append(spotify_features.get("danceability", 0.5))
            features.append(spotify_features.get("valence", 0.5))
            features.append(spotify_features.get("acousticness", 0.5))
            features.append(spotify_features.get("instrumentalness", 0.5))
        
        return np.array(features)



