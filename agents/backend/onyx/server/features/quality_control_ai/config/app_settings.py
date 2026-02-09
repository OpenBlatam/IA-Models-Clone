"""
Application Settings

Centralized application settings with environment variable support.
"""

import os
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppSettings:
    """
    Application settings with environment variable support.
    
    Settings can be loaded from environment variables or defaults.
    """
    
    # Application
    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "Quality Control AI"))
    app_version: str = field(default_factory=lambda: os.getenv("APP_VERSION", "2.2.0"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "False").lower() == "true")
    
    # API
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    api_workers: int = field(default_factory=lambda: int(os.getenv("API_WORKERS", "1")))
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "json"))
    log_file: Optional[str] = field(default_factory=lambda: os.getenv("LOG_FILE"))
    
    # Cache
    cache_enabled: bool = field(default_factory=lambda: os.getenv("CACHE_ENABLED", "True").lower() == "true")
    cache_max_size: int = field(default_factory=lambda: int(os.getenv("CACHE_MAX_SIZE", "1000")))
    cache_default_ttl: int = field(default_factory=lambda: int(os.getenv("CACHE_DEFAULT_TTL", "3600")))
    
    # Metrics
    metrics_enabled: bool = field(default_factory=lambda: os.getenv("METRICS_ENABLED", "True").lower() == "true")
    
    # ML Models
    model_device: str = field(default_factory=lambda: os.getenv("MODEL_DEVICE", "auto"))  # auto, cpu, cuda
    model_cache_dir: str = field(default_factory=lambda: os.getenv("MODEL_CACHE_DIR", "./models"))
    
    # Image Processing
    image_max_size: int = field(default_factory=lambda: int(os.getenv("IMAGE_MAX_SIZE", "10000")))
    image_default_size: tuple = field(default_factory=lambda: (
        int(os.getenv("IMAGE_DEFAULT_WIDTH", "224")),
        int(os.getenv("IMAGE_DEFAULT_HEIGHT", "224"))
    ))
    
    # Inspection
    inspection_timeout: float = field(default_factory=lambda: float(os.getenv("INSPECTION_TIMEOUT", "30.0")))
    inspection_batch_size: int = field(default_factory=lambda: int(os.getenv("INSPECTION_BATCH_SIZE", "8")))
    
    # Storage
    storage_path: str = field(default_factory=lambda: os.getenv("STORAGE_PATH", "./storage"))
    
    # Database (if needed)
    database_url: Optional[str] = field(default_factory=lambda: os.getenv("DATABASE_URL"))
    
    @classmethod
    def from_env(cls) -> 'AppSettings':
        """
        Create settings from environment variables.
        
        Returns:
            AppSettings instance
        """
        return cls()
    
    def to_dict(self) -> dict:
        """
        Convert settings to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "debug": self.debug,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "log_level": self.log_level,
            "cache_enabled": self.cache_enabled,
            "metrics_enabled": self.metrics_enabled,
        }
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate settings.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.api_port < 1 or self.api_port > 65535:
            return False, "API port must be between 1 and 65535"
        
        if self.cache_max_size < 1:
            return False, "Cache max size must be positive"
        
        if self.inspection_timeout <= 0:
            return False, "Inspection timeout must be positive"
        
        if self.image_max_size < 32:
            return False, "Image max size must be at least 32"
        
        return True, None


# Global settings instance
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """
    Get global settings instance.
    
    Returns:
        AppSettings instance
    """
    global _settings
    if _settings is None:
        _settings = AppSettings.from_env()
    return _settings



