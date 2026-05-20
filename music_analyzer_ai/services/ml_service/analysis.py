"""
ML Service Analysis Module

Track analysis functionality.
"""

from typing import Dict, Any, Optional
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


class AnalysisMixin:
    """Analysis mixin for MLService."""
    
    def analyze_track_comprehensive(
        self,
        audio_path: Optional[str] = None,
        audio_features: Optional[Any] = None,
        spotify_features: Optional[Dict[str, Any]] = None,
        use_pipeline: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis using all ML models.
        
        Args:
            audio_path: Path to audio file
            audio_features: Pre-extracted audio features
            spotify_features: Spotify API features
            use_pipeline: Whether to use processing pipeline
        
        Returns:
            Analysis results dictionary
        """
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



