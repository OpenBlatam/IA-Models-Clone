"""
Video Templates Service
Pre-configured templates for common video types
"""

from typing import Dict, Any, Optional
from enum import Enum
import logging

from ..core.models import VideoConfig, AudioConfig, SubtitleConfig

logger = logging.getLogger(__name__)


class VideoTemplate(str, Enum):
    """Available video templates"""
    EDUCATIONAL = "educational"
    MARKETING = "marketing"
    NEWS = "news"
    ENTERTAINMENT = "entertainment"
    CORPORATE = "corporate"
    SOCIAL_MEDIA = "social_media"
    YOUTUBE_SHORT = "youtube_short"
    INSTAGRAM_STORY = "instagram_story"
    TIKTOK = "tiktok"
    PODCAST = "podcast"


class TemplateService:
    """Manages video generation templates"""
    
    TEMPLATES: Dict[str, Dict[str, Any]] = {
        "educational": {
            "name": "Educational Video",
            "description": "Template for educational content",
            "video_config": {
                "resolution": "1920x1080",
                "fps": 30,
                "style": "realistic",
                "image_duration": 4.0,
                "transition_duration": 0.5,
            },
            "audio_config": {
                "voice": "neutral",
                "speed": 1.0,
                "background_music": True,
                "music_style": "ambient",
                "music_volume": 0.2,
            },
            "subtitle_config": {
                "enabled": True,
                "style": "modern",
                "font_size": 48,
                "position": "bottom",
            },
        },
        "marketing": {
            "name": "Marketing Video",
            "description": "Template for marketing and promotional content",
            "video_config": {
                "resolution": "1920x1080",
                "fps": 30,
                "style": "dynamic",
                "image_duration": 3.0,
                "transition_duration": 0.8,
            },
            "audio_config": {
                "voice": "female_1",
                "speed": 1.1,
                "background_music": True,
                "music_style": "energetic",
                "music_volume": 0.3,
            },
            "subtitle_config": {
                "enabled": True,
                "style": "bold",
                "font_size": 52,
                "position": "bottom",
            },
        },
        "news": {
            "name": "News Video",
            "description": "Template for news and journalism",
            "video_config": {
                "resolution": "1920x1080",
                "fps": 30,
                "style": "realistic",
                "image_duration": 3.5,
                "transition_duration": 0.3,
            },
            "audio_config": {
                "voice": "male_1",
                "speed": 1.0,
                "background_music": False,
            },
            "subtitle_config": {
                "enabled": True,
                "style": "simple",
                "font_size": 44,
                "position": "bottom",
            },
        },
        "entertainment": {
            "name": "Entertainment Video",
            "description": "Template for entertainment content",
            "video_config": {
                "resolution": "1920x1080",
                "fps": 30,
                "style": "animated",
                "image_duration": 2.5,
                "transition_duration": 1.0,
            },
            "audio_config": {
                "voice": "female_2",
                "speed": 1.0,
                "background_music": True,
                "music_style": "fun",
                "music_volume": 0.4,
            },
            "subtitle_config": {
                "enabled": True,
                "style": "neon",
                "font_size": 50,
                "position": "bottom",
            },
        },
        "corporate": {
            "name": "Corporate Video",
            "description": "Template for corporate and business content",
            "video_config": {
                "resolution": "1920x1080",
                "fps": 30,
                "style": "minimalist",
                "image_duration": 4.0,
                "transition_duration": 0.5,
            },
            "audio_config": {
                "voice": "neutral",
                "speed": 1.0,
                "background_music": True,
                "music_style": "corporate",
                "music_volume": 0.25,
            },
            "subtitle_config": {
                "enabled": True,
                "style": "elegant",
                "font_size": 46,
                "position": "bottom",
            },
        },
        "social_media": {
            "name": "Social Media Video",
            "description": "Template for social media posts",
            "video_config": {
                "resolution": "1080x1080",
                "fps": 30,
                "style": "dynamic",
                "image_duration": 2.0,
                "transition_duration": 0.6,
            },
            "audio_config": {
                "voice": "female_1",
                "speed": 1.1,
                "background_music": True,
                "music_style": "trendy",
                "music_volume": 0.35,
            },
            "subtitle_config": {
                "enabled": True,
                "style": "bold",
                "font_size": 56,
                "position": "center",
            },
        },
        "youtube_short": {
            "name": "YouTube Short",
            "description": "Template optimized for YouTube Shorts",
            "video_config": {
                "resolution": "1080x1920",
                "fps": 30,
                "style": "dynamic",
                "image_duration": 1.5,
                "transition_duration": 0.4,
            },
            "audio_config": {
                "voice": "female_1",
                "speed": 1.2,
                "background_music": True,
                "music_style": "viral",
                "music_volume": 0.4,
            },
            "subtitle_config": {
                "enabled": True,
                "style": "bold",
                "font_size": 60,
                "position": "center",
            },
        },
        "instagram_story": {
            "name": "Instagram Story",
            "description": "Template for Instagram Stories",
            "video_config": {
                "resolution": "1080x1920",
                "fps": 30,
                "style": "dynamic",
                "image_duration": 2.0,
                "transition_duration": 0.5,
            },
            "audio_config": {
                "voice": "female_1",
                "speed": 1.1,
                "background_music": True,
                "music_style": "trendy",
                "music_volume": 0.3,
            },
            "subtitle_config": {
                "enabled": True,
                "style": "modern",
                "font_size": 58,
                "position": "center",
            },
        },
        "tiktok": {
            "name": "TikTok Video",
            "description": "Template optimized for TikTok",
            "video_config": {
                "resolution": "1080x1920",
                "fps": 30,
                "style": "dynamic",
                "image_duration": 1.0,
                "transition_duration": 0.3,
            },
            "audio_config": {
                "voice": "female_1",
                "speed": 1.3,
                "background_music": True,
                "music_style": "viral",
                "music_volume": 0.5,
            },
            "subtitle_config": {
                "enabled": True,
                "style": "neon",
                "font_size": 64,
                "position": "center",
            },
        },
        "podcast": {
            "name": "Podcast Video",
            "description": "Template for podcast-style videos",
            "video_config": {
                "resolution": "1920x1080",
                "fps": 30,
                "style": "minimalist",
                "image_duration": 5.0,
                "transition_duration": 0.5,
            },
            "audio_config": {
                "voice": "male_1",
                "speed": 1.0,
                "background_music": False,
            },
            "subtitle_config": {
                "enabled": True,
                "style": "simple",
                "font_size": 42,
                "position": "bottom",
            },
        },
    }
    
    @classmethod
    def get_template(cls, template_name: str) -> Optional[Dict[str, Any]]:
        """Get template configuration by name"""
        return cls.TEMPLATES.get(template_name.lower())
    
    @classmethod
    def list_templates(cls) -> List[Dict[str, Any]]:
        """List all available templates"""
        return [
            {
                "name": name,
                "display_name": config["name"],
                "description": config["description"],
            }
            for name, config in cls.TEMPLATES.items()
        ]
    
    @classmethod
    def apply_template(
        cls,
        template_name: str,
        script_text: str,
        language: str = "es"
    ) -> Dict[str, Any]:
        """
        Apply template to create a video generation request
        
        Args:
            template_name: Name of the template
            script_text: Script text
            language: Language code
            
        Returns:
            Dictionary with video_config, audio_config, subtitle_config
        """
        template = cls.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        return {
            "video_config": VideoConfig(**template["video_config"]),
            "audio_config": AudioConfig(**template["audio_config"]),
            "subtitle_config": SubtitleConfig(**template["subtitle_config"]),
        }

