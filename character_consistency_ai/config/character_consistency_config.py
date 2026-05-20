"""
Character Consistency Configuration
====================================

Configuration management for character consistency AI feature.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class CharacterConsistencyConfig(BaseModel):
    """
    Configuration for Character Consistency AI.
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
    
    embedding_dim: int = Field(
        default=768,
        description="Dimension of character embeddings"
    )
    
    # Output configuration
    output_dir: str = Field(
        default="./character_embeddings",
        description="Directory to save safe tensors"
    )
    
    # API configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="API host address"
    )
    
    api_port: int = Field(
        default=8001,
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
    
    # Workflow configuration
    default_num_inference_steps: int = Field(
        default=50,
        description="Default number of inference steps"
    )
    
    default_guidance_scale: float = Field(
        default=7.5,
        description="Default guidance scale"
    )
    
    @classmethod
    def from_env(cls) -> "CharacterConsistencyConfig":
        """
        Create configuration from environment variables.
        
        Returns:
            CharacterConsistencyConfig instance
        """
        return cls(
            model_id=os.getenv("CHARACTER_CONSISTENCY_MODEL_ID", "black-forest-labs/flux2-dev"),
            device=os.getenv("CHARACTER_CONSISTENCY_DEVICE"),
            dtype=os.getenv("CHARACTER_CONSISTENCY_DTYPE"),
            enable_optimizations=os.getenv("CHARACTER_CONSISTENCY_OPTIMIZATIONS", "true").lower() == "true",
            embedding_dim=int(os.getenv("CHARACTER_CONSISTENCY_EMBEDDING_DIM", "768")),
            output_dir=os.getenv("CHARACTER_CONSISTENCY_OUTPUT_DIR", "./character_embeddings"),
            api_host=os.getenv("CHARACTER_CONSISTENCY_API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("CHARACTER_CONSISTENCY_API_PORT", "8001")),
            max_batch_size=int(os.getenv("CHARACTER_CONSISTENCY_MAX_BATCH_SIZE", "4")),
            max_image_size=int(os.getenv("CHARACTER_CONSISTENCY_MAX_IMAGE_SIZE", "1024")),
            default_num_inference_steps=int(os.getenv("CHARACTER_CONSISTENCY_INFERENCE_STEPS", "50")),
            default_guidance_scale=float(os.getenv("CHARACTER_CONSISTENCY_GUIDANCE_SCALE", "7.5")),
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


