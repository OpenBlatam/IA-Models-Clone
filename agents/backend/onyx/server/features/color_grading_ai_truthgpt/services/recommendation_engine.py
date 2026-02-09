"""
Recommendation Engine for Color Grading AI
==========================================

Intelligent recommendations for color grading based on content analysis.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Color grading recommendation."""
    template_name: Optional[str] = None
    color_params: Dict[str, Any] = None
    confidence: float = 0.0
    reason: str = ""
    category: str = ""


class RecommendationEngine:
    """
    Intelligent recommendation engine.
    
    Features:
    - Content-based recommendations
    - Template recommendations
    - Parameter suggestions
    - Learning from history
    """
    
    def __init__(self, template_manager, history_manager, metrics_collector):
        """
        Initialize recommendation engine.
        
        Args:
            template_manager: Template manager instance
            history_manager: History manager instance
            metrics_collector: Metrics collector instance
        """
        self.template_manager = template_manager
        self.history_manager = history_manager
        self.metrics_collector = metrics_collector
    
    async def recommend_for_media(
        self,
        media_analysis: Dict[str, Any],
        media_type: str = "video"
    ) -> List[Recommendation]:
        """
        Recommend color grading for media based on analysis.
        
        Args:
            media_analysis: Media analysis results
            media_type: Media type
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Analyze color temperature
        color_temp = media_analysis.get("color_temperature", {})
        temp_k = color_temp.get("temperature_k", 5500)
        
        # Analyze exposure
        exposure = media_analysis.get("exposure", {})
        exposure_level = exposure.get("level", "normal")
        
        # Analyze brightness
        stats = media_analysis.get("statistics", {})
        brightness = stats.get("overall", {}).get("mean", 128)
        
        # Recommend based on temperature
        if temp_k < 5000:  # Cool
            recommendations.append(Recommendation(
                template_name="Cool Blue",
                confidence=0.8,
                reason="Cool color temperature detected",
                category="temperature"
            ))
        elif temp_k > 6000:  # Warm
            recommendations.append(Recommendation(
                template_name="Cinematic Warm",
                confidence=0.8,
                reason="Warm color temperature detected",
                category="temperature"
            ))
        
        # Recommend based on exposure
        if exposure_level == "underexposed":
            recommendations.append(Recommendation(
                color_params={"brightness": 0.2, "contrast": 1.1},
                confidence=0.7,
                reason="Underexposed image - increase brightness",
                category="exposure"
            ))
        elif exposure_level == "overexposed":
            recommendations.append(Recommendation(
                color_params={"brightness": -0.2, "contrast": 0.9},
                confidence=0.7,
                reason="Overexposed image - decrease brightness",
                category="exposure"
            ))
        
        # Recommend based on popular templates
        template_stats = self.metrics_collector.get_template_stats()
        if template_stats:
            most_used = max(template_stats.items(), key=lambda x: x[1]["count"])
            recommendations.append(Recommendation(
                template_name=most_used[0],
                confidence=0.6,
                reason="Most used template in your workflow",
                category="popular"
            ))
        
        # Sort by confidence
        recommendations.sort(key=lambda r: r.confidence, reverse=True)
        
        return recommendations
    
    def recommend_based_on_history(
        self,
        similar_media_id: Optional[str] = None
    ) -> List[Recommendation]:
        """
        Recommend based on processing history.
        
        Args:
            similar_media_id: Optional similar media ID
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Get recent successful operations
        history = self.history_manager.get_recent(limit=20)
        successful = [h for h in history if h.success]
        
        if successful:
            # Find most common template
            template_counts = {}
            for entry in successful:
                template = entry.template_used
                if template:
                    template_counts[template] = template_counts.get(template, 0) + 1
            
            if template_counts:
                most_common = max(template_counts.items(), key=lambda x: x[1])
                recommendations.append(Recommendation(
                    template_name=most_common[0],
                    confidence=0.7,
                    reason=f"Used successfully {most_common[1]} times",
                    category="history"
                ))
        
        return recommendations




