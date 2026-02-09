"""
Settings Configuration
======================

Configuración del sistema Artist Manager AI.
"""

import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Configuración de la aplicación."""
    
    # OpenRouter
    openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-haiku")
    openrouter_max_tokens: int = int(os.getenv("OPENROUTER_MAX_TOKENS", "2000"))
    openrouter_temperature: float = float(os.getenv("OPENROUTER_TEMPERATURE", "0.7"))
    
    # API
    api_prefix: str = "/artist-manager"
    api_version: str = "v1"
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Obtener configuración (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

