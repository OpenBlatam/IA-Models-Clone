"""
Application Configuration
==========================
Application configuration management using Pydantic Settings.

Provides singleton pattern for configuration access and
environment variable support with validation.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """
    Application configuration (optimized with Pydantic v2).
    
    Loads configuration from environment variables with fallback defaults.
    Uses singleton pattern for efficient access throughout the application.
    """
    
    # Server configuration
    host: str = "0.0.0.0"
    """Server host address."""
    
    port: int = 8025
    """Server port number."""
    
    # OpenRouter API configuration
    openrouter_api_key: Optional[str] = None
    """OpenRouter API key (from environment variable)."""
    
    openrouter_model: str = "anthropic/claude-3.5-sonnet"
    """Default OpenRouter model to use."""
    
    # Application metadata
    app_name: str = "Burnout Prevention AI"
    """Application name."""
    
    app_version: str = "1.0.0"
    """Application version."""
    
    debug: bool = False
    """Debug mode flag."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra env vars
    )


# Singleton instance
_config_instance: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    Get application configuration (singleton pattern).
    
    Returns the same configuration instance on every call,
    ensuring consistency and efficient memory usage.
    
    Returns:
        AppConfig instance (singleton)
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance

