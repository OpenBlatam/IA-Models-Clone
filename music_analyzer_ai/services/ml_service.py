"""
ML Service - High-level service layer for ML operations
"""

from typing import Dict, Any, Optional, List
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)

try:
    from ..core.deep_models import get_deep_analyzer, DeepMusicAnalyzer
    from ..core.ml_audio_analyzer import get_ml_analyzer, MLMusicAnalyzer, AudioFeatureExtractor
    from ..core.transformer_analyzer import get_transformer_analyzer
    from ..core.processing_layers import create_default_pipeline, ProcessingPipeline
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML components not available")


class MLService:
    """
    High-level ML service with multiple layers of abstraction:
    - Feature extraction
    - Model inference
    - Result aggregation
    - Caching
    """
    
    def __init__(self):
        if not ML_AVAILABLE:
            raise ImportError("ML components not available")
        
        self.deep_analyzer = get_deep_analyzer()
        self.ml_analyzer = get_ml_analyzer()
        self.transformer_analyzer = get_transformer_analyzer()
        self.feature_extractor = AudioFeatureExtractor()
        self.pipeline = create_default_pipeline()
        
        # Cache for results
        self.cache: Dict[str, Any] = {}
    
    def analyze_track_comprehensive(
        self,
        audio_path: Optional[str] = None,
        audio_features: Optional[Any] = None,
        spotify_features: Optional[Dict[str, Any]] = None,
        use_pipeline: bool = True
    ) -> Dict[str, Any]:
        """Comprehensive analysis using all ML models"""
        start_time = time.time()
        
        try:
            # Extract features if needed
            if audio_features is None and audio_path:
                audio_features = self.feature_extractor.extract_features(audio_path)
            
            # Use pipeline or direct analysis
            if use_pipeline and audio_path:
                result = self.pipeline.process(audio_path)
                if result["success"]:
                    return {
                        "success": True,
                        "analysis": result["data"],
                        "processing_time": result["total_time"],
                        "pipeline_used": True
                    }
            
            # Direct analysis
            results = {}
            
            # Deep model predictions
            if audio_features:
                feature_vector = self._create_feature_vector(audio_features, spotify_features)
                
                # Multi-task prediction
                multi_task = self.deep_analyzer.predict_multi_task(feature_vector)
                results["multi_task"] = multi_task
                
                # Genre prediction
                genre = self.deep_analyzer.predict_genre(feature_vector)
                results["genre"] = genre
            
            # Transformer analysis
            if audio_path:
                try:
                    transformer_result = self.transformer_analyzer.extract_embeddings(audio_path)
                    results["embeddings"] = {
                        "dimension": len(transformer_result),
                        "sample": transformer_result[:10].tolist() if isinstance(transformer_result, np.ndarray) else transformer_result[:10]
                    }
                except Exception as e:
                    logger.warning(f"Transformer analysis failed: {str(e)}")
            
            # ML analyzer
            if audio_features:
                ml_prediction = self.ml_analyzer.analyze_with_ml(
                    audio_features,
                    spotify_features
                )
                results["ml_analysis"] = {
                    "genre": ml_prediction.genre,
                    "mood": ml_prediction.mood,
                    "energy": ml_prediction.energy_level,
                    "complexity": ml_prediction.complexity_score,
                    "instruments": ml_prediction.instrument_detection
                }
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "analysis": results,
                "processing_time": processing_time,
                "pipeline_used": False
            }
        
        except Exception as e:
            logger.error(f"Comprehensive analysis error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _create_feature_vector(
        self,
        audio_features: Any,
        spotify_features: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Create feature vector from audio and Spotify features"""
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
    
    def compare_tracks(
        self,
        track1_path: str,
        track2_path: str
    ) -> Dict[str, Any]:
        """Compare two tracks using ML"""
        try:
            # Extract embeddings
            emb1 = self.transformer_analyzer.extract_embeddings(track1_path)
            emb2 = self.transformer_analyzer.extract_embeddings(track2_path)
            
            # Calculate similarity
            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )
            
            # Analyze each track
            analysis1 = self.analyze_track_comprehensive(audio_path=track1_path)
            analysis2 = self.analyze_track_comprehensive(audio_path=track2_path)
            
            return {
                "success": True,
                "similarity": float(similarity),
                "track1_analysis": analysis1.get("analysis", {}),
                "track2_analysis": analysis2.get("analysis", {}),
                "comparison": {
                    "genre_match": (
                        analysis1.get("analysis", {}).get("genre", {}).get("genre_id") ==
                        analysis2.get("analysis", {}).get("genre", {}).get("genre_id")
                    ),
                    "energy_diff": abs(
                        analysis1.get("analysis", {}).get("ml_analysis", {}).get("energy", 0) -
                        analysis2.get("analysis", {}).get("ml_analysis", {}).get("energy", 0)
                    )
                }
            }
        
        except Exception as e:
            logger.error(f"Track comparison error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


# Global instance
_ml_service: Optional[MLService] = None


def get_ml_service() -> MLService:
    """Get or create ML service instance"""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service

