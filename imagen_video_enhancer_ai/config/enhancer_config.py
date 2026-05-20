"""
Configuration for Imagen Video Enhancer AI
==========================================
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class OpenRouterConfig:
    """OpenRouter configuration."""
    api_key: Optional[str] = None
    model: str = "anthropic/claude-3.5-sonnet"
    timeout: float = 120.0  # Más tiempo para procesamiento de imágenes/videos
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class TruthGPTConfig:
    """TruthGPT configuration."""
    enabled: bool = True
    endpoint: Optional[str] = None
    timeout: float = 120.0
    optimization_type: str = "standard"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "truthgpt_endpoint": self.endpoint,
            "timeout": self.timeout,
            "optimization_type": self.optimization_type,
        }


@dataclass
class EnhancerConfig:
    """Configuration for Enhancer Agent."""
    
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    truthgpt: TruthGPTConfig = field(default_factory=TruthGPTConfig)
    max_parallel_tasks: int = 5  # Menos tareas paralelas para procesamiento pesado
    output_dir: str = "enhancer_output"
    debug: bool = False
    
    # Configuración específica para imágenes/videos
    max_file_size_mb: int = 100  # Tamaño máximo de archivo en MB
    allowed_image_formats: list = field(default_factory=lambda: ["jpg", "jpeg", "png", "webp", "bmp"])
    allowed_video_formats: list = field(default_factory=lambda: ["mp4", "avi", "mov", "mkv", "webm"])
    max_image_dimensions: tuple = (8192, 8192)  # Máximo 8K
    max_video_duration_seconds: int = 300  # 5 minutos máximo
    
    def __post_init__(self):
        """Initialize configuration from environment variables."""
        # OpenRouter
        if not self.openrouter.api_key:
            self.openrouter.api_key = os.getenv("OPENROUTER_API_KEY")
        
        # TruthGPT
        if self.truthgpt.enabled:
            if not self.truthgpt.endpoint:
                self.truthgpt.endpoint = os.getenv("TRUTHGPT_ENDPOINT")
    
    def validate(self):
        """Validate configuration."""
        if not self.openrouter.api_key:
            raise ValueError("OpenRouter API key is required")
        
        if self.truthgpt.enabled and not self.truthgpt.endpoint:
            # TruthGPT is optional, just log a warning
            import logging
            logging.getLogger(__name__).warning(
                "TruthGPT endpoint not configured, running without TruthGPT optimization"
            )




