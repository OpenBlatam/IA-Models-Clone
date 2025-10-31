"""
Video Template Service for HeyGen AI
Manages professional video templates, presets, and customization options.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VideoAspectRatio(str, Enum):
    """Video aspect ratio options"""
    LANDSCAPE_16_9 = "16:9"
    PORTRAIT_9_16 = "9:16"
    SQUARE_1_1 = "1:1"
    LANDSCAPE_21_9 = "21:9"
    PORTRAIT_4_5 = "4:5"


class VideoQuality(str, Enum):
    """Video quality presets"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"
    CUSTOM = "custom"


class VideoStyle(str, Enum):
    """Video style categories"""
    CORPORATE = "corporate"
    CREATIVE = "creative"
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    MARKETING = "marketing"
    PRESENTATION = "presentation"


class TransitionType(str, Enum):
    """Video transition types"""
    FADE = "fade"
    SLIDE = "slide"
    ZOOM = "zoom"
    DISSOLVE = "dissolve"
    WIPE = "wipe"
    NONE = "none"


@dataclass
class VideoTransition:
    """Video transition configuration"""
    type: TransitionType = TransitionType.FADE
    duration: float = 0.5
    easing: str = "ease-in-out"
    direction: str = "in"


@dataclass
class VideoEffect:
    """Video effect configuration"""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    intensity: float = 1.0


@dataclass
class VideoLayout:
    """Video layout configuration"""
    aspect_ratio: VideoAspectRatio = VideoAspectRatio.LANDSCAPE_16_9
    background_color: str = "#000000"
    background_image: Optional[str] = None
    padding: Dict[str, int] = field(default_factory=lambda: {"top": 0, "bottom": 0, "left": 0, "right": 0})
    avatar_position: str = "center"
    text_position: str = "bottom"
    logo_position: Optional[str] = None


@dataclass
class VideoPreset:
    """Video quality preset configuration"""
    name: str
    quality: VideoQuality
    resolution: Tuple[int, int]
    fps: int
    bitrate: str
    codec: str = "h264"
    audio_codec: str = "aac"
    audio_bitrate: str = "128k"


@dataclass
class VideoTemplate:
    """Video template configuration"""
    id: str
    name: str
    description: str
    style: VideoStyle
    layout: VideoLayout
    transitions: List[VideoTransition] = field(default_factory=list)
    effects: List[VideoEffect] = field(default_factory=list)
    preset: VideoPreset = None
    duration_range: Tuple[float, float] = (10.0, 300.0)
    max_text_length: int = 500
    supported_languages: List[str] = field(default_factory=lambda: ["en", "es"])
    tags: List[str] = field(default_factory=list)
    thumbnail: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


class VideoTemplateService:
    """Service for managing video templates and presets"""
    
    def __init__(self, templates_path: str = "./data/video_templates"):
        self.templates_path = Path(templates_path)
        self.templates: Dict[str, VideoTemplate] = {}
        self.presets: Dict[str, VideoPreset] = {}
        self.custom_templates: Dict[str, VideoTemplate] = {}
        
        self._initialize_service()
        self._load_default_presets()
        self._load_default_templates()
    
    def _initialize_service(self):
        """Initialize the video template service"""
        try:
            self.templates_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.templates_path / "templates").mkdir(exist_ok=True)
            (self.templates_path / "presets").mkdir(exist_ok=True)
            (self.templates_path / "custom").mkdir(exist_ok=True)
            (self.templates_path / "thumbnails").mkdir(exist_ok=True)
            
            logger.info(f"Video template service initialized at {self.templates_path}")
        except Exception as e:
            logger.error(f"Failed to initialize video template service: {e}")
            raise
    
    def _load_default_presets(self):
        """Load default video quality presets"""
        self.presets = {
            "low": VideoPreset(
                name="Low Quality",
                quality=VideoQuality.LOW,
                resolution=(640, 360),
                fps=24,
                bitrate="500k"
            ),
            "medium": VideoPreset(
                name="Medium Quality",
                quality=VideoQuality.MEDIUM,
                resolution=(1280, 720),
                fps=30,
                bitrate="2000k"
            ),
            "high": VideoPreset(
                name="High Quality",
                quality=VideoQuality.HIGH,
                resolution=(1920, 1080),
                fps=30,
                bitrate="5000k"
            ),
            "ultra": VideoPreset(
                name="Ultra Quality",
                quality=VideoQuality.ULTRA,
                resolution=(3840, 2160),
                fps=60,
                bitrate="15000k"
            ),
            "social_media": VideoPreset(
                name="Social Media",
                quality=VideoQuality.MEDIUM,
                resolution=(1080, 1920),
                fps=30,
                bitrate="3000k"
            ),
            "presentation": VideoPreset(
                name="Presentation",
                quality=VideoQuality.HIGH,
                resolution=(1920, 1080),
                fps=30,
                bitrate="4000k"
            )
        }
        
        # Save presets to file
        self._save_presets()
    
    def _load_default_templates(self):
        """Load default video templates"""
        default_templates = [
            VideoTemplate(
                id="corporate_presentation",
                name="Corporate Presentation",
                description="Professional corporate presentation template",
                style=VideoStyle.CORPORATE,
                layout=VideoLayout(
                    aspect_ratio=VideoAspectRatio.LANDSCAPE_16_9,
                    background_color="#1a1a1a",
                    avatar_position="left",
                    text_position="right"
                ),
                transitions=[
                    VideoTransition(type=TransitionType.FADE, duration=0.8),
                    VideoTransition(type=TransitionType.SLIDE, duration=0.6)
                ],
                effects=[
                    VideoEffect(name="color_correction", parameters={"contrast": 1.1, "saturation": 0.9}),
                    VideoEffect(name="subtle_blur", parameters={"radius": 2}, intensity=0.3)
                ],
                preset=self.presets["presentation"],
                duration_range=(30.0, 600.0),
                max_text_length=300,
                tags=["corporate", "business", "professional", "presentation"]
            ),
            VideoTemplate(
                id="social_media_story",
                name="Social Media Story",
                description="Vertical video template for social media stories",
                style=VideoStyle.SOCIAL_MEDIA,
                layout=VideoLayout(
                    aspect_ratio=VideoAspectRatio.PORTRAIT_9_16,
                    background_color="#000000",
                    avatar_position="center",
                    text_position="bottom"
                ),
                transitions=[
                    VideoTransition(type=TransitionType.ZOOM, duration=0.5),
                    VideoTransition(type=TransitionType.FADE, duration=0.3)
                ],
                effects=[
                    VideoEffect(name="vibrant_colors", parameters={"saturation": 1.2}),
                    VideoEffect(name="grain", parameters={"intensity": 0.1})
                ],
                preset=self.presets["social_media"],
                duration_range=(5.0, 60.0),
                max_text_length=100,
                tags=["social", "story", "vertical", "trendy"]
            ),
            VideoTemplate(
                id="educational_lesson",
                name="Educational Lesson",
                description="Template for educational content and lessons",
                style=VideoStyle.EDUCATIONAL,
                layout=VideoLayout(
                    aspect_ratio=VideoAspectRatio.LANDSCAPE_16_9,
                    background_color="#f8f9fa",
                    avatar_position="right",
                    text_position="left"
                ),
                transitions=[
                    VideoTransition(type=TransitionType.SLIDE, duration=0.7),
                    VideoTransition(type=TransitionType.DISSOLVE, duration=0.5)
                ],
                effects=[
                    VideoEffect(name="bright_colors", parameters={"brightness": 1.05}),
                    VideoEffect(name="text_highlight", parameters={"color": "#007bff"})
                ],
                preset=self.presets["high"],
                duration_range=(60.0, 1800.0),
                max_text_length=800,
                tags=["education", "learning", "lesson", "tutorial"]
            ),
            VideoTemplate(
                id="marketing_ad",
                name="Marketing Advertisement",
                description="High-impact marketing and advertising template",
                style=VideoStyle.MARKETING,
                layout=VideoLayout(
                    aspect_ratio=VideoAspectRatio.LANDSCAPE_16_9,
                    background_color="#000000",
                    avatar_position="center",
                    text_position="overlay"
                ),
                transitions=[
                    VideoTransition(type=TransitionType.WIPE, duration=0.4),
                    VideoTransition(type=TransitionType.ZOOM, duration=0.6)
                ],
                effects=[
                    VideoEffect(name="dynamic_colors", parameters={"saturation": 1.3}),
                    VideoEffect(name="motion_blur", parameters={"intensity": 0.2}),
                    VideoEffect(name="glow", parameters={"color": "#ff6b6b", "radius": 10})
                ],
                preset=self.presets["high"],
                duration_range=(15.0, 120.0),
                max_text_length=200,
                tags=["marketing", "advertisement", "promotional", "dynamic"]
            ),
            VideoTemplate(
                id="news_broadcast",
                name="News Broadcast",
                description="Professional news and broadcast template",
                style=VideoStyle.NEWS,
                layout=VideoLayout(
                    aspect_ratio=VideoAspectRatio.LANDSCAPE_16_9,
                    background_color="#2c3e50",
                    avatar_position="left",
                    text_position="right",
                    logo_position="top-right"
                ),
                transitions=[
                    VideoTransition(type=TransitionType.FADE, duration=0.5),
                    VideoTransition(type=TransitionType.SLIDE, duration=0.4)
                ],
                effects=[
                    VideoEffect(name="news_style", parameters={"contrast": 1.2}),
                    VideoEffect(name="lower_third", parameters={"color": "#34495e"})
                ],
                preset=self.presets["high"],
                duration_range=(30.0, 900.0),
                max_text_length=400,
                tags=["news", "broadcast", "journalism", "professional"]
            )
        ]
        
        for template in default_templates:
            self.templates[template.id] = template
        
        # Save templates to file
        self._save_templates()
    
    def _save_presets(self):
        """Save presets to JSON file"""
        try:
            presets_data = {}
            for key, preset in self.presets.items():
                presets_data[key] = {
                    "name": preset.name,
                    "quality": preset.quality.value,
                    "resolution": preset.resolution,
                    "fps": preset.fps,
                    "bitrate": preset.bitrate,
                    "codec": preset.codec,
                    "audio_codec": preset.audio_codec,
                    "audio_bitrate": preset.audio_bitrate
                }
            
            with open(self.templates_path / "presets" / "video_presets.json", "w") as f:
                json.dump(presets_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save presets: {e}")
    
    def _save_templates(self):
        """Save templates to JSON file"""
        try:
            templates_data = {}
            for template_id, template in self.templates.items():
                templates_data[template_id] = {
                    "id": template.id,
                    "name": template.name,
                    "description": template.description,
                    "style": template.style.value,
                    "layout": {
                        "aspect_ratio": template.layout.aspect_ratio.value,
                        "background_color": template.layout.background_color,
                        "background_image": template.layout.background_image,
                        "padding": template.layout.padding,
                        "avatar_position": template.layout.avatar_position,
                        "text_position": template.layout.text_position,
                        "logo_position": template.layout.logo_position
                    },
                    "transitions": [
                        {
                            "type": t.type.value,
                            "duration": t.duration,
                            "easing": t.easing,
                            "direction": t.direction
                        } for t in template.transitions
                    ],
                    "effects": [
                        {
                            "name": e.name,
                            "parameters": e.parameters,
                            "enabled": e.enabled,
                            "intensity": e.intensity
                        } for e in template.effects
                    ],
                    "preset": template.preset.name if template.preset else None,
                    "duration_range": template.duration_range,
                    "max_text_length": template.max_text_length,
                    "supported_languages": template.supported_languages,
                    "tags": template.tags,
                    "thumbnail": template.thumbnail,
                    "created_at": template.created_at,
                    "updated_at": template.updated_at
                }
            
            with open(self.templates_path / "templates" / "video_templates.json", "w") as f:
                json.dump(templates_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save templates: {e}")
    
    def get_template(self, template_id: str) -> Optional[VideoTemplate]:
        """Get a video template by ID"""
        return self.templates.get(template_id) or self.custom_templates.get(template_id)
    
    def get_all_templates(self) -> Dict[str, VideoTemplate]:
        """Get all available templates"""
        return {**self.templates, **self.custom_templates}
    
    def get_templates_by_style(self, style: VideoStyle) -> List[VideoTemplate]:
        """Get templates filtered by style"""
        all_templates = self.get_all_templates()
        return [t for t in all_templates.values() if t.style == style]
    
    def get_templates_by_aspect_ratio(self, aspect_ratio: VideoAspectRatio) -> List[VideoTemplate]:
        """Get templates filtered by aspect ratio"""
        all_templates = self.get_all_templates()
        return [t for t in all_templates.values() if t.layout.aspect_ratio == aspect_ratio]
    
    def search_templates(self, query: str) -> List[VideoTemplate]:
        """Search templates by name, description, or tags"""
        query = query.lower()
        all_templates = self.get_all_templates()
        
        results = []
        for template in all_templates.values():
            if (query in template.name.lower() or 
                query in template.description.lower() or
                any(query in tag.lower() for tag in template.tags)):
                results.append(template)
        
        return results
    
    def create_custom_template(self, template: VideoTemplate) -> str:
        """Create a custom video template"""
        try:
            # Validate template
            if not template.id or template.id in self.templates:
                raise ValueError("Template ID must be unique")
            
            # Add to custom templates
            self.custom_templates[template.id] = template
            
            # Save to file
            self._save_custom_templates()
            
            logger.info(f"Created custom template: {template.id}")
            return template.id
        except Exception as e:
            logger.error(f"Failed to create custom template: {e}")
            raise
    
    def update_template(self, template_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing template"""
        try:
            template = self.get_template(template_id)
            if not template:
                return False
            
            # Update fields
            for key, value in updates.items():
                if hasattr(template, key):
                    setattr(template, key, value)
            
            # Save changes
            if template_id in self.templates:
                self._save_templates()
            else:
                self._save_custom_templates()
            
            logger.info(f"Updated template: {template_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update template: {e}")
            return False
    
    def delete_template(self, template_id: str) -> bool:
        """Delete a custom template"""
        try:
            if template_id in self.custom_templates:
                del self.custom_templates[template_id]
                self._save_custom_templates()
                logger.info(f"Deleted custom template: {template_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete template: {e}")
            return False
    
    def get_preset(self, preset_name: str) -> Optional[VideoPreset]:
        """Get a video preset by name"""
        return self.presets.get(preset_name)
    
    def get_all_presets(self) -> Dict[str, VideoPreset]:
        """Get all available presets"""
        return self.presets.copy()
    
    def create_custom_preset(self, preset: VideoPreset) -> str:
        """Create a custom video preset"""
        try:
            preset_key = preset.name.lower().replace(" ", "_")
            self.presets[preset_key] = preset
            self._save_presets()
            logger.info(f"Created custom preset: {preset_key}")
            return preset_key
        except Exception as e:
            logger.error(f"Failed to create custom preset: {e}")
            raise
    
    def _save_custom_templates(self):
        """Save custom templates to JSON file"""
        try:
            custom_data = {}
            for template_id, template in self.custom_templates.items():
                custom_data[template_id] = {
                    "id": template.id,
                    "name": template.name,
                    "description": template.description,
                    "style": template.style.value,
                    "layout": {
                        "aspect_ratio": template.layout.aspect_ratio.value,
                        "background_color": template.layout.background_color,
                        "background_image": template.layout.background_image,
                        "padding": template.layout.padding,
                        "avatar_position": template.layout.avatar_position,
                        "text_position": template.layout.text_position,
                        "logo_position": template.layout.logo_position
                    },
                    "transitions": [
                        {
                            "type": t.type.value,
                            "duration": t.duration,
                            "easing": t.easing,
                            "direction": t.direction
                        } for t in template.transitions
                    ],
                    "effects": [
                        {
                            "name": e.name,
                            "parameters": e.parameters,
                            "enabled": e.enabled,
                            "intensity": e.intensity
                        } for e in template.effects
                    ],
                    "preset": template.preset.name if template.preset else None,
                    "duration_range": template.duration_range,
                    "max_text_length": template.max_text_length,
                    "supported_languages": template.supported_languages,
                    "tags": template.tags,
                    "thumbnail": template.thumbnail,
                    "created_at": template.created_at,
                    "updated_at": template.updated_at
                }
            
            with open(self.templates_path / "custom" / "custom_templates.json", "w") as f:
                json.dump(custom_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save custom templates: {e}")
    
    def validate_template_for_content(self, template_id: str, content_length: int, duration: float, language: str = "en") -> Tuple[bool, List[str]]:
        """Validate if a template is suitable for given content"""
        template = self.get_template(template_id)
        if not template:
            return False, ["Template not found"]
        
        errors = []
        
        # Check content length
        if content_length > template.max_text_length:
            errors.append(f"Content too long. Max: {template.max_text_length}, Got: {content_length}")
        
        # Check duration
        min_duration, max_duration = template.duration_range
        if duration < min_duration or duration > max_duration:
            errors.append(f"Duration out of range. Range: {min_duration}-{max_duration}s, Got: {duration}s")
        
        # Check language support
        if language not in template.supported_languages:
            errors.append(f"Language '{language}' not supported. Supported: {template.supported_languages}")
        
        return len(errors) == 0, errors
    
    def get_recommended_templates(self, content_length: int, duration: float, style: Optional[VideoStyle] = None, aspect_ratio: Optional[VideoAspectRatio] = None) -> List[VideoTemplate]:
        """Get recommended templates based on content characteristics"""
        all_templates = self.get_all_templates()
        recommendations = []
        
        for template in all_templates.values():
            # Filter by style if specified
            if style and template.style != style:
                continue
            
            # Filter by aspect ratio if specified
            if aspect_ratio and template.layout.aspect_ratio != aspect_ratio:
                continue
            
            # Check if template is suitable
            is_valid, _ = self.validate_template_for_content(template.id, content_length, duration)
            if is_valid:
                recommendations.append(template)
        
        # Sort by relevance (you could implement more sophisticated scoring)
        recommendations.sort(key=lambda t: len(t.tags), reverse=True)
        
        return recommendations
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the video template service"""
        try:
            stats = {
                "status": "healthy",
                "total_templates": len(self.templates),
                "total_custom_templates": len(self.custom_templates),
                "total_presets": len(self.presets),
                "templates_by_style": {},
                "templates_by_aspect_ratio": {},
                "errors": []
            }
            
            # Count templates by style
            for style in VideoStyle:
                stats["templates_by_style"][style.value] = len(self.get_templates_by_style(style))
            
            # Count templates by aspect ratio
            for aspect_ratio in VideoAspectRatio:
                stats["templates_by_aspect_ratio"][aspect_ratio.value] = len(self.get_templates_by_aspect_ratio(aspect_ratio))
            
            # Check file system
            if not self.templates_path.exists():
                stats["status"] = "error"
                stats["errors"].append("Templates directory does not exist")
            
            return stats
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "errors": [str(e)]
            }


# Example usage
if __name__ == "__main__":
    # Initialize service
    service = VideoTemplateService()
    
    # Get all templates
    templates = service.get_all_templates()
    print(f"Available templates: {len(templates)}")
    
    # Search for corporate templates
    corporate_templates = service.get_templates_by_style(VideoStyle.CORPORATE)
    print(f"Corporate templates: {len(corporate_templates)}")
    
    # Get recommendations
    recommendations = service.get_recommended_templates(
        content_length=200,
        duration=60.0,
        style=VideoStyle.CORPORATE
    )
    print(f"Recommended templates: {len(recommendations)}")
    
    # Health check
    health = service.health_check()
    print(f"Service health: {health['status']}")


