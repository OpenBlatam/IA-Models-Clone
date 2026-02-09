"""
Upscaling Presets
==================

Pre-configured presets for different use cases.
"""

import logging
from typing import Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UpscalingPreset:
    """Upscaling preset configuration."""
    name: str
    scale_factor: float
    quality_mode: str
    use_ai: bool
    use_optimization_core: bool
    anti_aliasing_strength: float
    sharpness_factor: float
    description: str


class PresetManager:
    """
    Manage upscaling presets.
    
    Presets:
    - photo_enhancement: For photos
    - artwork_upscale: For artwork
    - pixel_art: For pixel art
    - document_scan: For scanned documents
    - video_frame: For video frames
    """
    
    PRESETS = {
        "photo_enhancement": UpscalingPreset(
            name="photo_enhancement",
            scale_factor=2.0,
            quality_mode="high",
            use_ai=True,
            use_optimization_core=True,
            anti_aliasing_strength=0.4,
            sharpness_factor=1.2,
            description="High-quality photo enhancement with AI"
        ),
        "artwork_upscale": UpscalingPreset(
            name="artwork_upscale",
            scale_factor=4.0,
            quality_mode="ultra",
            use_ai=True,
            use_optimization_core=True,
            anti_aliasing_strength=0.3,
            sharpness_factor=1.3,
            description="Ultra-quality artwork upscaling"
        ),
        "pixel_art": UpscalingPreset(
            name="pixel_art",
            scale_factor=4.0,
            quality_mode="balanced",
            use_ai=False,
            use_optimization_core=False,
            anti_aliasing_strength=0.0,
            sharpness_factor=1.0,
            description="Pixel art upscaling without smoothing"
        ),
        "document_scan": UpscalingPreset(
            name="document_scan",
            scale_factor=2.0,
            quality_mode="high",
            use_ai=True,
            use_optimization_core=True,
            anti_aliasing_strength=0.2,
            sharpness_factor=1.4,
            description="Document scan enhancement"
        ),
        "video_frame": UpscalingPreset(
            name="video_frame",
            scale_factor=2.0,
            quality_mode="balanced",
            use_ai=False,
            use_optimization_core=True,
            anti_aliasing_strength=0.5,
            sharpness_factor=1.1,
            description="Fast video frame upscaling"
        ),
        # Real-ESRGAN specific presets
        "realesrgan_photo": UpscalingPreset(
            name="realesrgan_photo",
            scale_factor=4.0,
            quality_mode="ultra",
            use_ai=True,
            use_optimization_core=True,
            anti_aliasing_strength=0.3,
            sharpness_factor=1.2,
            description="Real-ESRGAN photo upscaling (4x)"
        ),
        "realesrgan_anime": UpscalingPreset(
            name="realesrgan_anime",
            scale_factor=4.0,
            quality_mode="ultra",
            use_ai=True,
            use_optimization_core=True,
            anti_aliasing_strength=0.2,
            sharpness_factor=1.3,
            description="Real-ESRGAN anime upscaling (4x, optimized for anime)"
        ),
        "realesrgan_artwork": UpscalingPreset(
            name="realesrgan_artwork",
            scale_factor=4.0,
            quality_mode="ultra",
            use_ai=True,
            use_optimization_core=True,
            anti_aliasing_strength=0.4,
            sharpness_factor=1.25,
            description="Real-ESRGAN artwork upscaling (4x)"
        ),
        "realesrgan_fast": UpscalingPreset(
            name="realesrgan_fast",
            scale_factor=2.0,
            quality_mode="high",
            use_ai=False,
            use_optimization_core=False,
            anti_aliasing_strength=0.3,
            sharpness_factor=1.1,
            description="Real-ESRGAN fast upscaling (2x, RealESRNet)"
        ),
    }
    
    @classmethod
    def get_preset(cls, name: str) -> Optional[UpscalingPreset]:
        """
        Get preset by name.
        
        Args:
            name: Preset name
            
        Returns:
            Preset or None if not found
        """
        return cls.PRESETS.get(name)
    
    @classmethod
    def list_presets(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all available presets.
        
        Returns:
            Dictionary of preset information
        """
        return {
            name: {
                "name": preset.name,
                "scale_factor": preset.scale_factor,
                "quality_mode": preset.quality_mode,
                "use_ai": preset.use_ai,
                "use_optimization_core": preset.use_optimization_core,
                "description": preset.description
            }
            for name, preset in cls.PRESETS.items()
        }
    
    @classmethod
    def apply_preset(
        cls,
        preset_name: str,
        **overrides
    ) -> Dict[str, Any]:
        """
        Get preset configuration with optional overrides.
        
        Args:
            preset_name: Preset name
            **overrides: Override values
            
        Returns:
            Configuration dictionary
        """
        preset = cls.get_preset(preset_name)
        if not preset:
            raise ValueError(f"Preset '{preset_name}' not found")
        
        config = {
            "scale_factor": preset.scale_factor,
            "quality_mode": preset.quality_mode,
            "use_ai": preset.use_ai,
            "use_optimization_core": preset.use_optimization_core,
            "anti_aliasing_strength": preset.anti_aliasing_strength,
            "sharpness_factor": preset.sharpness_factor,
        }
        
        # Apply overrides
        config.update(overrides)
        
        return config

