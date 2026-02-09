"""
Clothing Changer Configuration
==============================

Configuration management for character clothing changer AI feature.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class ClothingChangerConfig(BaseModel):
    """
    Configuration for Character Clothing Changer AI.
    """
    
    # Model configuration
    model_id: str = Field(
        default="black-forest-labs/flux2-dev",
        description="HuggingFace model ID for Flux2"
    )
    
    device: Optional[str] = Field(
        default=None,
        description="Device to use (cuda/cpu/auto). None for auto-detection"
    )
    
    dtype: Optional[str] = Field(
        default=None,
        description="Data type (float16/float32). None for auto-detection"
    )
    
    enable_optimizations: bool = Field(
        default=True,
        description="Enable memory and speed optimizations"
    )
    
    use_inpainting: bool = Field(
        default=True,
        description="Use inpainting pipeline for clothing replacement"
    )
    
    use_controlnet: bool = Field(
        default=False,
        description="Use ControlNet for better control (experimental)"
    )
    
    # Output configuration
    output_dir: str = Field(
        default="./comfyui_tensors",
        description="Directory to save ComfyUI safe tensors"
    )
    
    # API configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="API host address"
    )
    
    api_port: int = Field(
        default=8002,
        description="API port"
    )
    
    # Performance configuration
    max_batch_size: int = Field(
        default=4,
        description="Maximum batch size for processing"
    )
    
    max_image_size: int = Field(
        default=1024,
        description="Maximum image size (width/height)"
    )
    
    # Generation configuration
    default_num_inference_steps: int = Field(
        default=50,
        description="Default number of inference steps"
    )
    
    default_guidance_scale: float = Field(
        default=7.5,
        description="Default guidance scale"
    )
    
    default_strength: float = Field(
        default=0.8,
        description="Default inpainting strength"
    )
    
    default_negative_prompt: str = Field(
        default="blurry, low quality, distorted, deformed, bad anatomy",
        description="Default negative prompt"
    )
    
    @classmethod
    def from_env(cls) -> "ClothingChangerConfig":
        """
        Create configuration from environment variables.
        
        Returns:
            ClothingChangerConfig instance
        """
        return cls(
            model_id=os.getenv("CLOTHING_CHANGER_MODEL_ID", "black-forest-labs/flux2-dev"),
            device=os.getenv("CLOTHING_CHANGER_DEVICE"),
            dtype=os.getenv("CLOTHING_CHANGER_DTYPE"),
            enable_optimizations=os.getenv("CLOTHING_CHANGER_OPTIMIZATIONS", "true").lower() == "true",
            use_inpainting=os.getenv("CLOTHING_CHANGER_USE_INPAINTING", "true").lower() == "true",
            use_controlnet=os.getenv("CLOTHING_CHANGER_USE_CONTROLNET", "false").lower() == "true",
            output_dir=os.getenv("CLOTHING_CHANGER_OUTPUT_DIR", "./comfyui_tensors"),
            api_host=os.getenv("CLOTHING_CHANGER_API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("CLOTHING_CHANGER_API_PORT", "8002")),
            max_batch_size=int(os.getenv("CLOTHING_CHANGER_MAX_BATCH_SIZE", "4")),
            max_image_size=int(os.getenv("CLOTHING_CHANGER_MAX_IMAGE_SIZE", "1024")),
            default_num_inference_steps=int(os.getenv("CLOTHING_CHANGER_INFERENCE_STEPS", "50")),
            default_guidance_scale=float(os.getenv("CLOTHING_CHANGER_GUIDANCE_SCALE", "7.5")),
            default_strength=float(os.getenv("CLOTHING_CHANGER_STRENGTH", "0.8")),
            default_negative_prompt=os.getenv(
                "CLOTHING_CHANGER_NEGATIVE_PROMPT",
                "blurry, low quality, distorted, deformed, bad anatomy"
            ),
        )
    
    def validate(self) -> None:
        """Validate configuration."""
        # Validate output directory
        output_path = Path(self.output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Invalid output directory: {e}")
        
        # Validate device
        if self.device and self.device not in ["cuda", "cpu", "auto"]:
            raise ValueError(f"Invalid device: {self.device}. Must be cuda, cpu, or auto")
        
        # Validate dtype
        if self.dtype and self.dtype not in ["float16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}. Must be float16 or float32")
        
        logger.info("Configuration validated successfully")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    class Config:
        """Pydantic config."""
        extra = "forbid"
        use_enum_values = True


