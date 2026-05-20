"""
ML Service Comparison Module

Track comparison functionality.
"""

from typing import Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ComparisonMixin:
    """Comparison mixin for MLService."""
    
    def compare_tracks(
        self,
        track1_path: str,
        track2_path: str
    ) -> Dict[str, Any]:
        """
        Compare two tracks using ML.
        
        Args:
            track1_path: Path to first track
            track2_path: Path to second track
        
        Returns:
            Comparison results dictionary
        """
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



