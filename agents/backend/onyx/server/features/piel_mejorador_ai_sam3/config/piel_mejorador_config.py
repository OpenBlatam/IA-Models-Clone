"""
Configuration for Piel Mejorador AI SAM3
=========================================
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class OpenRouterConfig:
    """OpenRouter configuration."""
    api_key: Optional[str] = None
    model: str = "anthropic/claude-3.5-sonnet"
    timeout: float = 120.0  # Longer timeout for image/video processing
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
class PielMejoradorConfig:
    """Configuration for Piel Mejorador SAM3 Agent."""
    
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    truthgpt: TruthGPTConfig = field(default_factory=TruthGPTConfig)
    max_parallel_tasks: int = 5  # Lower for image/video processing
    output_dir: str = "piel_mejorador_output"
    debug: bool = False
    
    # Skin enhancement specific settings
    max_image_size_mb: int = 50
    max_video_size_mb: int = 500
    supported_image_formats: list = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"])
    supported_video_formats: list = field(default_factory=lambda: [".mp4", ".mov", ".avi", ".webm"])
    
    # Enhancement levels
    enhancement_levels: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "low": {"intensity": 0.3, "realism": 0.5},
        "medium": {"intensity": 0.6, "realism": 0.7},
        "high": {"intensity": 0.9, "realism": 0.9},
        "ultra": {"intensity": 1.0, "realism": 1.0},
    })
    
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
    
    def get_enhancement_config(self, level: str) -> Dict[str, Any]:
        """Get enhancement configuration for a level."""
        if level not in self.enhancement_levels:
            raise ValueError(f"Unknown enhancement level: {level}. Available: {list(self.enhancement_levels.keys())}")
        return self.enhancement_levels[level]




