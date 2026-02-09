"""
ML Music Analyzer Module

ML-powered music analyzer with genre classification, mood detection, etc.
"""

from typing import Dict, Any, Optional, List
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    from transformers import AutoModel, AutoFeatureExtractor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available")

from .dataclasses import AudioFeatures, MLPrediction
from .feature_extractor import AudioFeatureExtractor

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class MLMusicAnalyzer:
    """
    ML-powered music analyzer with:
    - Genre classification
    - Mood detection
    - Instrument detection
    - Complexity analysis
    - Transformer-based features
    
    Args:
        device: Device to run models on.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.feature_extractor = AudioFeatureExtractor() if LIBROSA_AVAILABLE else None
        
        # Initialize models
        self.genre_classifier: Optional[Any] = None
        self.transformer_model = None
        self.transformer_extractor = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use a pre-trained audio model
                model_name = "facebook/wav2vec2-base"
                self.transformer_extractor = AutoFeatureExtractor.from_pretrained(model_name)
                self.transformer_model = AutoModel.from_pretrained(model_name)
                if TORCH_AVAILABLE:
                    self.transformer_model = self.transformer_model.to(device)
                logger.info("Loaded transformer model for audio analysis")
            except Exception as e:
                logger.warning(f"Could not load transformer model: {str(e)}")
    
    def analyze_with_ml(
        self,
        audio_features: AudioFeatures,
        spotify_features: Optional[Dict[str, Any]] = None
    ) -> MLPrediction:
        """
        Perform ML-based analysis.
        
        Args:
            audio_features: Extracted audio features.
            spotify_features: Optional Spotify audio features.
        
        Returns:
            MLPrediction with analysis results.
        """
        start_time = time.time()
        
        # Combine features
        feature_vector = self._create_feature_vector(audio_features, spotify_features)
        
        # Genre prediction
        genre = self._predict_genre(feature_vector)
        
        # Mood detection
        mood = self._detect_mood(feature_vector, spotify_features)
        
        # Energy level
        energy = self._calculate_energy(audio_features, spotify_features)
        
        # Complexity score
        complexity = self._calculate_complexity(audio_features)
        
        # Instrument detection (simplified)
        instruments = self._detect_instruments(audio_features)
        
        processing_time = time.time() - start_time
        logger.info(f"ML analysis completed in {processing_time:.2f}s")
        
        return MLPrediction(
            genre=genre,
            mood=mood,
            energy_level=energy,
            complexity_score=complexity,
            instrument_detection=instruments,
            confidence=0.85
        )
    
    def _create_feature_vector(
        self,
        audio_features: AudioFeatures,
        spotify_features: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Create feature vector from audio and Spotify features."""
        features = []
        
        # Audio features
        features.extend(audio_features.mfcc.mean(axis=1))
        features.extend(audio_features.chroma.mean(axis=1))
        features.extend(audio_features.spectral_contrast.mean(axis=1))
        features.extend(audio_features.tonnetz.mean(axis=1))
        features.append(audio_features.tempo)
        
        # Spotify features if available
        if spotify_features:
            features.append(spotify_features.get("energy", 0.5))
            features.append(spotify_features.get("danceability", 0.5))
            features.append(spotify_features.get("valence", 0.5))
            features.append(spotify_features.get("acousticness", 0.5))
            features.append(spotify_features.get("instrumentalness", 0.5))
        
        return np.array(features)
    
    def _predict_genre(self, feature_vector: np.ndarray) -> str:
        """Predict music genre."""
        # Simplified genre prediction
        genres = [
            "Pop", "Rock", "Jazz", "Classical", "Electronic",
            "Hip-Hop", "Country", "Blues", "Reggae", "Metal"
        ]
        
        # Simple heuristic based on features
        if feature_vector[0] > 0.5:
            return genres[1]  # Rock
        elif feature_vector[1] > 0.5:
            return genres[0]  # Pop
        else:
            return genres[2]  # Jazz
    
    def _detect_mood(
        self,
        feature_vector: np.ndarray,
        spotify_features: Optional[Dict[str, Any]]
    ) -> str:
        """Detect musical mood."""
        moods = ["Happy", "Sad", "Energetic", "Calm", "Aggressive", "Melancholic"]
        
        if spotify_features:
            valence = spotify_features.get("valence", 0.5)
            energy = spotify_features.get("energy", 0.5)
            
            if valence > 0.7 and energy > 0.6:
                return "Happy"
            elif valence < 0.3 and energy < 0.4:
                return "Sad"
            elif energy > 0.7:
                return "Energetic"
            elif energy < 0.3:
                return "Calm"
        
        return "Neutral"
    
    def _calculate_energy(
        self,
        audio_features: AudioFeatures,
        spotify_features: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate energy level."""
        if spotify_features:
            return spotify_features.get("energy", 0.5)
        
        # Calculate from audio features
        mfcc_energy = np.mean(np.abs(audio_features.mfcc))
        return float(np.clip(mfcc_energy / 10.0, 0.0, 1.0))
    
    def _calculate_complexity(self, audio_features: AudioFeatures) -> float:
        """Calculate musical complexity score."""
        # Complexity based on feature variance
        mfcc_var = np.var(audio_features.mfcc)
        chroma_var = np.var(audio_features.chroma)
        
        complexity = (mfcc_var + chroma_var) / 2.0
        return float(np.clip(complexity / 5.0, 0.0, 1.0))
    
    def _detect_instruments(self, audio_features: AudioFeatures) -> List[str]:
        """Detect instruments (simplified)."""
        instruments = []
        
        # Heuristic-based detection
        if np.mean(audio_features.chroma) > 0.5:
            instruments.append("Piano")
        if audio_features.tempo > 120:
            instruments.append("Drums")
        if np.mean(audio_features.spectral_contrast) > 0.3:
            instruments.append("Guitar")
        
        return instruments if instruments else ["Unknown"]
    
    def analyze_with_transformer(self, audio_path: str) -> Dict[str, Any]:
        """Analyze audio using transformer model."""
        if not TRANSFORMERS_AVAILABLE or self.transformer_model is None:
            return {"error": "Transformer model not available"}
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Extract features with transformer
            inputs = self.transformer_extractor(
                y, sampling_rate=sr, return_tensors="pt"
            )
            
            if TORCH_AVAILABLE:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.transformer_model(**inputs)
                
                # Extract embeddings
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                
                return {
                    "embeddings": embeddings.tolist(),
                    "embedding_dim": embeddings.shape[1]
                }
        except Exception as e:
            logger.error(f"Transformer analysis error: {str(e)}")
            return {"error": str(e)}


# Global instance
_ml_analyzer: Optional[MLMusicAnalyzer] = None


def get_ml_analyzer(device: str = "cpu") -> MLMusicAnalyzer:
    """
    Get or create ML analyzer instance.
    
    Args:
        device: Device to run models on.
    
    Returns:
        MLMusicAnalyzer instance.
    """
    global _ml_analyzer
    if _ml_analyzer is None:
        _ml_analyzer = MLMusicAnalyzer(device=device)
    return _ml_analyzer



