"""
Settings Module
===============

Centralized settings management with environment variable support.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """Application settings."""
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8002
    api_reload: bool = False
    api_log_level: str = "info"
    
    # Model Settings
    model_id: str = "black-forest-labs/flux2-dev"
    device: Optional[str] = None
    dtype: Optional[str] = None
    
    # Path Settings
    output_dir: str = "./comfyui_tensors"
    cache_dir: str = "./embedding_cache"
    temp_dir: str = "./temp"
    
    # Feature Flags
    enable_cache: bool = True
    enable_validation: bool = True
    enable_enhancement: bool = False
    
    # Performance Settings
    max_batch_size: int = 4
    max_workers: int = 4
    
    # Security Settings
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    
    # Logging Settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Environment
    environment: str = "development"
    debug: bool = False
    
    @classmethod
    def from_env(cls) -> "Settings":
        """
        Create settings from environment variables.
        
        Returns:
            Settings instance
        """
        return cls(
            api_host=os.getenv("CLOTHING_CHANGER_API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("CLOTHING_CHANGER_API_PORT", "8002")),
            api_reload=os.getenv("CLOTHING_CHANGER_API_RELOAD", "false").lower() == "true",
            api_log_level=os.getenv("CLOTHING_CHANGER_API_LOG_LEVEL", "info"),
            model_id=os.getenv("CLOTHING_CHANGER_MODEL_ID", "black-forest-labs/flux2-dev"),
            device=os.getenv("CLOTHING_CHANGER_DEVICE"),
            dtype=os.getenv("CLOTHING_CHANGER_DTYPE"),
            output_dir=os.getenv("CLOTHING_CHANGER_OUTPUT_DIR", "./comfyui_tensors"),
            cache_dir=os.getenv("CLOTHING_CHANGER_CACHE_DIR", "./embedding_cache"),
            temp_dir=os.getenv("CLOTHING_CHANGER_TEMP_DIR", "./temp"),
            enable_cache=os.getenv("CLOTHING_CHANGER_ENABLE_CACHE", "true").lower() == "true",
            enable_validation=os.getenv("CLOTHING_CHANGER_ENABLE_VALIDATION", "true").lower() == "true",
            enable_enhancement=os.getenv("CLOTHING_CHANGER_ENABLE_ENHANCEMENT", "false").lower() == "true",
            max_batch_size=int(os.getenv("CLOTHING_CHANGER_MAX_BATCH_SIZE", "4")),
            max_workers=int(os.getenv("CLOTHING_CHANGER_MAX_WORKERS", "4")),
            enable_rate_limiting=os.getenv("CLOTHING_CHANGER_ENABLE_RATE_LIMITING", "true").lower() == "true",
            rate_limit_requests=int(os.getenv("CLOTHING_CHANGER_RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window=int(os.getenv("CLOTHING_CHANGER_RATE_LIMIT_WINDOW", "60")),
            log_level=os.getenv("CLOTHING_CHANGER_LOG_LEVEL", "INFO"),
            log_file=os.getenv("CLOTHING_CHANGER_LOG_FILE"),
            environment=os.getenv("CLOTHING_CHANGER_ENVIRONMENT", "development"),
            debug=os.getenv("CLOTHING_CHANGER_DEBUG", "false").lower() == "true",
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary.
        
        Returns:
            Settings dictionary
        """
        return {
            "api_host": self.api_host,
            "api_port": self.api_port,
            "api_reload": self.api_reload,
            "api_log_level": self.api_log_level,
            "model_id": self.model_id,
            "device": self.device,
            "dtype": self.dtype,
            "output_dir": self.output_dir,
            "cache_dir": self.cache_dir,
            "temp_dir": self.temp_dir,
            "enable_cache": self.enable_cache,
            "enable_validation": self.enable_validation,
            "enable_enhancement": self.enable_enhancement,
            "max_batch_size": self.max_batch_size,
            "max_workers": self.max_workers,
            "enable_rate_limiting": self.enable_rate_limiting,
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window": self.rate_limit_window,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "environment": self.environment,
            "debug": self.debug,
        }
    
    def validate(self) -> None:
        """Validate settings."""
        if self.api_port < 1 or self.api_port > 65535:
            raise ValueError(f"Invalid API port: {self.api_port}")
        
        if self.max_batch_size < 1:
            raise ValueError(f"Invalid max batch size: {self.max_batch_size}")
        
        if self.max_workers < 1:
            raise ValueError(f"Invalid max workers: {self.max_workers}")


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    _settings = Settings.from_env()
    return _settings

