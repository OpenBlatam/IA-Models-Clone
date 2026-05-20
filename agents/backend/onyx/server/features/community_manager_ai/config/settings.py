"""
Settings - Configuración del Sistema
====================================

Configuración centralizada para Community Manager AI.
"""

import os
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Configuración de la aplicación"""
    
    # General
    app_name: str = "Community Manager AI"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Storage
    meme_storage_path: str = Field(
        default="data/memes",
        env="MEME_STORAGE_PATH"
    )
    data_path: str = Field(
        default="data",
        env="DATA_PATH"
    )
    
    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # Scheduler
    scheduler_check_interval: int = Field(
        default=60,
        env="SCHEDULER_CHECK_INTERVAL"
    )
    max_posts_per_run: int = Field(
        default=10,
        env="MAX_POSTS_PER_RUN"
    )
    
    # Social Media APIs
    facebook_api_version: str = "v18.0"
    instagram_api_version: str = "v18.0"
    twitter_api_version: str = "2"
    linkedin_api_version: str = "v2"
    tiktok_api_version: str = "v1.3"
    youtube_api_version: str = "v3"
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_per_minute: int = Field(
        default=60,
        env="RATE_LIMIT_PER_MINUTE"
    )
    
    # Database (opcional)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # Email Configuration
    email_enabled: bool = Field(default=False, env="EMAIL_ENABLED")
    smtp_host: str = Field(default="localhost", env="SMTP_HOST")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_user: Optional[str] = Field(default=None, env="SMTP_USER")
    smtp_password: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    smtp_from: Optional[str] = Field(default=None, env="SMTP_FROM")
    
    # Webhook Configuration
    default_webhook_url: Optional[str] = Field(default=None, env="DEFAULT_WEBHOOK_URL")
    webhook_secret: Optional[str] = Field(default=None, env="WEBHOOK_SECRET")
    
    # AI Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    ai_provider: str = Field(default="openai", env="AI_PROVIDER")
    ai_model: str = Field(default="gpt-4", env="AI_MODEL")
    ai_enabled: bool = Field(default=True, env="AI_ENABLED")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Obtener instancia de configuración (singleton)
    
    Returns:
        Instancia de Settings
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

