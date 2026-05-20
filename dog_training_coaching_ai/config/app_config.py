"""
Application Configuration
==========================
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    """Configuración de la aplicación."""
    
    # Server
    host: str = os.getenv("DOG_TRAINING_AI_HOST", "0.0.0.0")
    port: int = int(os.getenv("DOG_TRAINING_AI_PORT", "8030"))
    
    # OpenRouter
    openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
    
    # App
    app_name: str = "Dog Training Coaching AI"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Training Settings
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "60"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    max_training_sessions: int = int(os.getenv("MAX_TRAINING_SESSIONS", "100"))
    
    # CORS
    cors_origins: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


_config_instance: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    Obtener configuración de la aplicación (singleton).
    
    Returns:
        Instancia de configuración
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance

