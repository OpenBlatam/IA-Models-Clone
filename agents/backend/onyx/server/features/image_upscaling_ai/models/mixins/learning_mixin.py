"""
Learning Mixin

Contains machine learning and adaptive learning functionality.
"""

import logging
from typing import Union, Dict, Any, Optional, List, Tuple
from pathlib import Path
from PIL import Image

from ..helpers import (
    QualityCalculator,
    MethodSelector,
)

logger = logging.getLogger(__name__)


class LearningMixin:
    """
    Mixin providing machine learning and adaptive learning functionality.
    
    This mixin contains:
    - Adaptive method selection
    - Learning from results
    - Performance learning
    - Quality learning
    - Method recommendation learning
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize learning mixin."""
        super().__init__(*args, **kwargs)
        if not hasattr(self, '_learning_data'):
            self._learning_data = {
                "method_performance": {},
                "quality_history": [],
                "preferences": {},
            }
    
    def learn_from_result(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        method: str,
        result: Image.Image,
        processing_time: float,
        user_feedback: Optional[float] = None
    ):
        """
        Learn from upscaling result.
        
        Args:
            image: Input image
            scale_factor: Scale factor used
            method: Method used
            result: Resulting image
            processing_time: Processing time
            user_feedback: Optional user feedback (0.0-1.0)
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        if not hasattr(self, '_learning_data'):
            self._learning_data = {
                "method_performance": {},
                "quality_history": [],
                "preferences": {},
            }
        
        # Calculate quality
        quality = QualityCalculator.calculate_quality_metrics(result)
        
        # Update method performance
        if method not in self._learning_data["method_performance"]:
            self._learning_data["method_performance"][method] = {
                "count": 0,
                "total_quality": 0.0,
                "total_time": 0.0,
                "total_feedback": 0.0,
                "feedback_count": 0,
            }
        
        perf = self._learning_data["method_performance"][method]
        perf["count"] += 1
        perf["total_quality"] += quality.overall_quality
        perf["total_time"] += processing_time
        
        if user_feedback is not None:
            perf["total_feedback"] += user_feedback
            perf["feedback_count"] += 1
        
        # Store quality history
        self._learning_data["quality_history"].append({
            "method": method,
            "scale_factor": scale_factor,
            "quality": quality.overall_quality,
            "time": processing_time,
            "image_size": pil_image.size,
        })
        
        # Keep only recent history (last 1000)
        if len(self._learning_data["quality_history"]) > 1000:
            self._learning_data["quality_history"] = self._learning_data["quality_history"][-1000:]
    
    def get_learned_recommendation(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        priority: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Get recommendation based on learned data.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            priority: Priority ('speed', 'quality', 'balanced')
            
        Returns:
            Dictionary with learned recommendation
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        if not hasattr(self, '_learning_data') or not self._learning_data["method_performance"]:
            # Fallback to standard recommendation
            return {
                "method": MethodSelector.select_best_method(pil_image, scale_factor),
                "confidence": 0.5,
                "source": "default",
            }
        
        # Analyze learned performance
        method_scores = {}
        
        for method, perf in self._learning_data["method_performance"].items():
            if perf["count"] == 0:
                continue
            
            avg_quality = perf["total_quality"] / perf["count"]
            avg_time = perf["total_time"] / perf["count"]
            
            # Calculate score based on priority
            if priority == "quality":
                score = avg_quality
            elif priority == "speed":
                score = 1.0 / (avg_time + 0.1)  # Inverse of time
            else:  # balanced
                score = avg_quality / (avg_time + 0.1)
            
            # Add user feedback if available
            if perf["feedback_count"] > 0:
                avg_feedback = perf["total_feedback"] / perf["feedback_count"]
                score = score * 0.7 + avg_feedback * 0.3
            
            method_scores[method] = {
                "score": score,
                "avg_quality": avg_quality,
                "avg_time": avg_time,
                "count": perf["count"],
            }
        
        if not method_scores:
            return {
                "method": MethodSelector.select_best_method(pil_image, scale_factor),
                "confidence": 0.5,
                "source": "default",
            }
        
        # Get best method
        best_method = max(method_scores.items(), key=lambda x: x[1]["score"])
        
        return {
            "method": best_method[0],
            "confidence": min(best_method[1]["score"] / 10.0, 1.0),  # Normalize
            "source": "learned",
            "method_stats": method_scores,
            "recommendation": best_method[0],
        }
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get learning statistics.
        
        Returns:
            Dictionary with learning statistics
        """
        if not hasattr(self, '_learning_data'):
            return {
                "total_operations": 0,
                "methods_learned": 0,
                "quality_history_size": 0,
            }
        
        total_ops = sum(
            perf["count"] for perf in self._learning_data["method_performance"].values()
        )
        
        return {
            "total_operations": total_ops,
            "methods_learned": len(self._learning_data["method_performance"]),
            "quality_history_size": len(self._learning_data["quality_history"]),
            "method_performance": {
                method: {
                    "count": perf["count"],
                    "avg_quality": perf["total_quality"] / perf["count"] if perf["count"] > 0 else 0.0,
                    "avg_time": perf["total_time"] / perf["count"] if perf["count"] > 0 else 0.0,
                }
                for method, perf in self._learning_data["method_performance"].items()
            }
        }
    
    def clear_learning_data(self):
        """Clear all learning data."""
        if hasattr(self, '_learning_data'):
            self._learning_data = {
                "method_performance": {},
                "quality_history": [],
                "preferences": {},
            }
            logger.info("Learning data cleared")


