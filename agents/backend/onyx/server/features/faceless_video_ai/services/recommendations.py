"""
Recommendations Service
Provides intelligent recommendations for video generation
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class RecommendationService:
    """Provides intelligent recommendations"""
    
    def __init__(self):
        self.analytics = None  # Will be injected
    
    def recommend_style(self, script_text: str, language: str = "es") -> str:
        """
        Recommend video style based on script content
        
        Args:
            script_text: Script text
            language: Language code
            
        Returns:
            Recommended style
        """
        text_lower = script_text.lower()
        
        # Keywords for different styles
        style_keywords = {
            "realistic": ["real", "actual", "foto", "fotografía", "verdadero", "realidad"],
            "animated": ["animado", "dibujo", "cartoon", "ilustración", "colorido"],
            "abstract": ["abstracto", "arte", "creativo", "artístico", "diseño"],
            "minimalist": ["simple", "minimalista", "limpio", "elegante", "moderno"],
            "dynamic": ["energía", "dinámico", "movimiento", "acción", "rápido"],
        }
        
        scores = {style: 0 for style in style_keywords.keys()}
        
        for style, keywords in style_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[style] += 1
        
        # Return style with highest score, default to realistic
        recommended = max(scores.items(), key=lambda x: x[1])
        return recommended[0] if recommended[1] > 0 else "realistic"
    
    def recommend_voice(self, script_text: str, language: str = "es") -> str:
        """
        Recommend voice based on script content
        
        Args:
            script_text: Script text
            language: Language code
            
        Returns:
            Recommended voice
        """
        text_lower = script_text.lower()
        
        # Determine tone
        if any(word in text_lower for word in ["profesional", "negocio", "corporativo", "empresa"]):
            return "male_1"  # Professional male voice
        elif any(word in text_lower for word in ["amigable", "cálido", "personal", "cercano"]):
            return "female_1"  # Friendly female voice
        elif any(word in text_lower for word in ["técnico", "científico", "educativo"]):
            return "neutral"  # Neutral voice
        else:
            return "neutral"  # Default
    
    def recommend_subtitle_style(self, video_style: str) -> str:
        """
        Recommend subtitle style based on video style
        
        Args:
            video_style: Video style
            
        Returns:
            Recommended subtitle style
        """
        style_mapping = {
            "realistic": "modern",
            "animated": "neon",
            "abstract": "glass",
            "minimalist": "minimal",
            "dynamic": "bold",
        }
        
        return style_mapping.get(video_style, "modern")
    
    def recommend_resolution(self, platform: Optional[str] = None) -> str:
        """
        Recommend resolution based on platform
        
        Args:
            platform: Target platform
            
        Returns:
            Recommended resolution
        """
        platform_resolutions = {
            "youtube": "1920x1080",
            "youtube_short": "1080x1920",
            "instagram": "1080x1080",
            "instagram_story": "1080x1920",
            "tiktok": "1080x1920",
            "facebook": "1920x1080",
            "twitter": "1280x720",
        }
        
        return platform_resolutions.get(platform, "1920x1080")
    
    def recommend_music_style(self, video_style: str, content_type: str = "general") -> str:
        """
        Recommend music style
        
        Args:
            video_style: Video style
            content_type: Content type (educational, marketing, etc.)
            
        Returns:
            Recommended music style
        """
        if content_type == "educational":
            return "ambient"
        elif content_type == "marketing":
            return "energetic"
        elif content_type == "corporate":
            return "corporate"
        elif content_type == "entertainment":
            return "fun"
        else:
            return "ambient"
    
    def get_full_recommendations(
        self,
        script_text: str,
        language: str = "es",
        platform: Optional[str] = None,
        content_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Get full set of recommendations
        
        Args:
            script_text: Script text
            language: Language code
            platform: Target platform
            content_type: Content type
            
        Returns:
            Dictionary with all recommendations
        """
        style = self.recommend_style(script_text, language)
        voice = self.recommend_voice(script_text, language)
        subtitle_style = self.recommend_subtitle_style(style)
        resolution = self.recommend_resolution(platform)
        music_style = self.recommend_music_style(style, content_type)
        
        return {
            "video_style": style,
            "voice": voice,
            "subtitle_style": subtitle_style,
            "resolution": resolution,
            "music_style": music_style,
            "recommendations": {
                "fps": 30,
                "image_duration": 3.0,
                "transition_duration": 0.5,
                "subtitle_position": "bottom",
                "background_music": True,
            }
        }


_recommendation_service: Optional[RecommendationService] = None


def get_recommendation_service() -> RecommendationService:
    """Get recommendation service instance (singleton)"""
    global _recommendation_service
    if _recommendation_service is None:
        _recommendation_service = RecommendationService()
    return _recommendation_service

