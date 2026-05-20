"""
Avatar Types
============

Data types for avatar generation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from shared.enums import AvatarStyle, AvatarQuality, Resolution


@dataclass
class AvatarGenerationConfig:
    """Configuration for avatar generation using diffusion models.
    
    Attributes:
        resolution: Output resolution (720p, 1080p, 4k)
        style: Avatar style (realistic, cartoon, anime, artistic)
        quality: Generation quality level
        enable_lip_sync: Enable lip-sync functionality
        enable_expressions: Enable facial expressions
        seed: Random seed for reproducibility
        num_inference_steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale
        scheduler: Diffusion scheduler type
        use_mixed_precision: Use FP16/BF16 for inference
    """
    resolution: Resolution = Resolution.P1080
    style: AvatarStyle = AvatarStyle.REALISTIC
    quality: AvatarQuality = AvatarQuality.HIGH
    enable_lip_sync: bool = True
    enable_expressions: bool = True
    seed: int | None = None
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    scheduler: str = "ddim"  # ddim, dpm_solver
    use_mixed_precision: bool = True

    def get_resolution_tuple(self) -> Tuple[int, int]:
        """Get resolution as (width, height) tuple."""
        resolution_map = {
            Resolution.P720: (1280, 720),
            Resolution.P1080: (1920, 1080),
            Resolution.P4K: (3840, 2160),
        }
        return resolution_map.get(self.resolution, (1920, 1080))

    def get_inference_steps(self) -> int:
        """Get number of inference steps based on quality."""
        quality_steps = {
            AvatarQuality.LOW: 20,
            AvatarQuality.MEDIUM: 30,
            AvatarQuality.HIGH: 50,
            AvatarQuality.ULTRA: 100,
        }
        return quality_steps.get(self.quality, self.num_inference_steps)


@dataclass
class AvatarModel:
    """Avatar model configuration.
    
    Attributes:
        id: Unique identifier
        name: Display name
        style: Avatar style
        model_path: Path to model weights or HuggingFace model ID
        characteristics: Additional characteristics dictionary
    """
    id: str
    name: str
    style: AvatarStyle
    model_path: str
    characteristics: Dict[str, Any] = field(default_factory=dict)



