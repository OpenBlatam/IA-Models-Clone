"""
ML Optimizer for Color Grading AI
==================================

Machine learning optimizations for color grading.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class MLOptimizer:
    """
    ML-based optimizations for color grading.
    
    Features:
    - Auto-tuning parameters
    - Learning from user preferences
    - Predictive color matching
    - Quality prediction
    """
    
    def __init__(self):
        """Initialize ML optimizer."""
        self._preferences: Dict[str, List[Dict[str, Any]]] = {}
        self._model_weights: Dict[str, float] = {}
    
    def learn_from_preference(
        self,
        user_id: str,
        input_analysis: Dict[str, Any],
        applied_params: Dict[str, Any],
        user_rating: float
    ):
        """
        Learn from user preference.
        
        Args:
            user_id: User identifier
            input_analysis: Input media analysis
            applied_params: Applied color parameters
            user_rating: User rating (0-1)
        """
        if user_id not in self._preferences:
            self._preferences[user_id] = []
        
        self._preferences[user_id].append({
            "input": input_analysis,
            "params": applied_params,
            "rating": user_rating,
            "timestamp": None  # Will be set by caller
        })
        
        logger.info(f"Learned preference from user {user_id} (rating: {user_rating})")
    
    def predict_optimal_params(
        self,
        user_id: str,
        input_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict optimal parameters based on user history.
        
        Args:
            user_id: User identifier
            input_analysis: Input media analysis
            
        Returns:
            Predicted optimal parameters
        """
        if user_id not in self._preferences or not self._preferences[user_id]:
            # No history, return default
            return self._get_default_params(input_analysis)
        
        # Find similar inputs
        preferences = self._preferences[user_id]
        similar = self._find_similar_inputs(input_analysis, preferences)
        
        if not similar:
            return self._get_default_params(input_analysis)
        
        # Weighted average of similar preferences
        total_weight = sum(s["similarity"] * s["rating"] for s in similar)
        if total_weight == 0:
            return self._get_default_params(input_analysis)
        
        # Calculate weighted parameters
        optimal = {}
        for key in ["brightness", "contrast", "saturation"]:
            weighted_sum = sum(
                s["params"].get(key, 0) * s["similarity"] * s["rating"]
                for s in similar
            )
            optimal[key] = weighted_sum / total_weight
        
        return optimal
    
    def _find_similar_inputs(
        self,
        input_analysis: Dict[str, Any],
        preferences: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find similar input analyses."""
        similar = []
        
        input_temp = input_analysis.get("color_temperature", {}).get("temperature_k", 5500)
        input_brightness = input_analysis.get("statistics", {}).get("overall", {}).get("mean", 128)
        
        for pref in preferences:
            pref_input = pref["input"]
            pref_temp = pref_input.get("color_temperature", {}).get("temperature_k", 5500)
            pref_brightness = pref_input.get("statistics", {}).get("overall", {}).get("mean", 128)
            
            # Calculate similarity
            temp_diff = abs(input_temp - pref_temp) / 1000  # Normalize
            brightness_diff = abs(input_brightness - pref_brightness) / 255  # Normalize
            
            similarity = 1.0 - (temp_diff + brightness_diff) / 2.0
            similarity = max(0, similarity)  # Clamp to 0-1
            
            if similarity > 0.5:  # Threshold
                similar.append({
                    **pref,
                    "similarity": similarity
                })
        
        # Sort by similarity * rating
        similar.sort(key=lambda x: x["similarity"] * x["rating"], reverse=True)
        
        return similar[:5]  # Top 5 similar
    
    def _get_default_params(self, input_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get default parameters based on analysis."""
        # Simple heuristic-based defaults
        brightness = input_analysis.get("statistics", {}).get("overall", {}).get("mean", 128)
        
        if brightness < 100:  # Dark
            return {"brightness": 0.2, "contrast": 1.1, "saturation": 1.0}
        elif brightness > 180:  # Bright
            return {"brightness": -0.1, "contrast": 1.0, "saturation": 1.0}
        else:
            return {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0}
    
    def predict_quality(
        self,
        input_analysis: Dict[str, Any],
        color_params: Dict[str, Any]
    ) -> float:
        """
        Predict output quality score.
        
        Args:
            input_analysis: Input media analysis
            color_params: Color parameters to apply
            
        Returns:
            Predicted quality score (0-1)
        """
        # Simple quality prediction based on parameters
        score = 0.5  # Base score
        
        # Adjust based on parameter ranges
        brightness = color_params.get("brightness", 0)
        contrast = color_params.get("contrast", 1.0)
        saturation = color_params.get("saturation", 1.0)
        
        # Penalize extreme values
        if abs(brightness) > 0.5:
            score -= 0.1
        if contrast < 0.5 or contrast > 2.0:
            score -= 0.1
        if saturation < 0.5 or saturation > 2.0:
            score -= 0.1
        
        # Boost for balanced parameters
        if -0.2 <= brightness <= 0.2 and 0.8 <= contrast <= 1.2 and 0.8 <= saturation <= 1.2:
            score += 0.2
        
        return max(0, min(1, score))
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for user preferences."""
        if user_id not in self._preferences:
            return {"total_preferences": 0}
        
        prefs = self._preferences[user_id]
        ratings = [p["rating"] for p in prefs]
        
        return {
            "total_preferences": len(prefs),
            "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
            "min_rating": min(ratings) if ratings else 0,
            "max_rating": max(ratings) if ratings else 0,
        }




