"""
Image Upscaling Configuration
==============================

Configuration management for image upscaling AI feature.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class OpenRouterConfig(BaseModel):
    """OpenRouter API configuration."""
    
    api_key: Optional[str] = Field(
        default=None,
        description="OpenRouter API key"
    )
    
    model: str = Field(
        default="anthropic/claude-3.5-sonnet",
        description="OpenRouter model to use for AI processing"
    )
    
    timeout: float = Field(
        default=120.0,
        description="Request timeout in seconds"
    )
    
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts"
    )
    
    retry_delay: float = Field(
        default=1.0,
        description="Retry delay in seconds"
    )


class UpscalingConfig(BaseModel):
    """
    Configuration for Image Upscaling AI.
    """
    
    # OpenRouter configuration
    openrouter: OpenRouterConfig = Field(
        default_factory=OpenRouterConfig,
        description="OpenRouter API configuration"
    )
    
    # Upscaling configuration
    default_scale_factor: float = Field(
        default=2.0,
        description="Default upscaling scale factor (2x, 4x, etc.)"
    )
    
    max_scale_factor: float = Field(
        default=8.0,
        description="Maximum upscaling scale factor"
    )
    
    min_scale_factor: float = Field(
        default=1.5,
        description="Minimum upscaling scale factor"
    )
    
    # Quality settings
    quality_mode: str = Field(
        default="high",
        description="Quality mode: 'fast', 'balanced', 'high', 'ultra'"
    )
    
    use_ai_enhancement: bool = Field(
        default=True,
        description="Use AI enhancement via OpenRouter"
    )
    
    use_optimization_core: bool = Field(
        default=True,
        description="Use optimization_core for processing"
    )
    
    # Output configuration
    output_dir: str = Field(
        default="./upscaled_images",
        description="Directory to save upscaled images"
    )
    
    # API configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="API host address"
    )
    
    api_port: int = Field(
        default=8003,
        description="API port"
    )
    
    # Performance configuration
    max_image_size: int = Field(
        default=4096,
        description="Maximum output image size (width/height)"
    )
    
    batch_size: int = Field(
        default=1,
        description="Batch size for processing"
    )
    
    # Optimization core settings
    optimization_core_path: Optional[str] = Field(
        default=None,
        description="Path to optimization_core module"
    )
    
    # Real-ESRGAN settings
    use_realesrgan: bool = Field(
        default=False,
        description="Use Real-ESRGAN for upscaling (requires model download)"
    )
    
    realesrgan_model: str = Field(
        default="RealESRGAN_x4plus",
        description="Real-ESRGAN model to use"
    )
    
    realesrgan_model_path: Optional[str] = Field(
        default=None,
        description="Path to Real-ESRGAN model file (auto-download if None)"
    )
    
    realesrgan_auto_download: bool = Field(
        default=False,
        description="Automatically download Real-ESRGAN models if not found"
    )
    
    @classmethod
    def from_env(cls) -> "UpscalingConfig":
        """
        Create configuration from environment variables.
        
        Returns:
            UpscalingConfig instance
        """
        openrouter_config = OpenRouterConfig(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=os.getenv("UPSCALING_OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"),
            timeout=float(os.getenv("UPSCALING_TIMEOUT", "120.0")),
            max_retries=int(os.getenv("UPSCALING_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("UPSCALING_RETRY_DELAY", "1.0")),
        )
        
        return cls(
            openrouter=openrouter_config,
            default_scale_factor=float(os.getenv("UPSCALING_SCALE_FACTOR", "2.0")),
            max_scale_factor=float(os.getenv("UPSCALING_MAX_SCALE", "8.0")),
            min_scale_factor=float(os.getenv("UPSCALING_MIN_SCALE", "1.5")),
            quality_mode=os.getenv("UPSCALING_QUALITY_MODE", "high"),
            use_ai_enhancement=os.getenv("UPSCALING_USE_AI", "true").lower() == "true",
            use_optimization_core=os.getenv("UPSCALING_USE_OPTIMIZATION_CORE", "true").lower() == "true",
            output_dir=os.getenv("UPSCALING_OUTPUT_DIR", "./upscaled_images"),
            api_host=os.getenv("UPSCALING_API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("UPSCALING_API_PORT", "8003")),
            max_image_size=int(os.getenv("UPSCALING_MAX_IMAGE_SIZE", "4096")),
            batch_size=int(os.getenv("UPSCALING_BATCH_SIZE", "1")),
            optimization_core_path=os.getenv("OPTIMIZATION_CORE_PATH"),
            use_realesrgan=os.getenv("UPSCALING_USE_REALESRGAN", "false").lower() == "true",
            realesrgan_model=os.getenv("REALESRGAN_MODEL", "RealESRGAN_x4plus"),
            realesrgan_model_path=os.getenv("REALESRGAN_MODEL_PATH"),
            realesrgan_auto_download=os.getenv("REALESRGAN_AUTO_DOWNLOAD", "false").lower() == "true",
        )
    
    def validate(self) -> None:
        """Validate configuration."""
        # Validate output directory
        output_path = Path(self.output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Invalid output directory: {e}")
        
        # Validate scale factor
        if self.default_scale_factor < self.min_scale_factor:
            raise ValueError(f"Default scale factor {self.default_scale_factor} is below minimum {self.min_scale_factor}")
        
        if self.default_scale_factor > self.max_scale_factor:
            raise ValueError(f"Default scale factor {self.default_scale_factor} exceeds maximum {self.max_scale_factor}")
        
        # Validate quality mode
        if self.quality_mode not in ["fast", "balanced", "high", "ultra"]:
            raise ValueError(f"Invalid quality mode: {self.quality_mode}")
        
        logger.info("Configuration validated successfully")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    class Config:
        """Pydantic config."""
        extra = "forbid"
        use_enum_values = True

