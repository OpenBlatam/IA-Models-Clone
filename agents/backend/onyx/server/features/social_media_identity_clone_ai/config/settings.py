"""
Configuración del sistema Social Media Identity Clone AI
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración de la aplicación"""
    
    # API Keys
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    tiktok_api_key: Optional[str] = os.getenv("TIKTOK_API_KEY")
    instagram_api_key: Optional[str] = os.getenv("INSTAGRAM_API_KEY")
    youtube_api_key: Optional[str] = os.getenv("YOUTUBE_API_KEY")
    
    # Modelos de IA
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    transcription_model: str = os.getenv("TRANSCRIPTION_MODEL", "whisper-1")
    
    # Configuración de extracción
    max_videos_per_profile: int = int(os.getenv("MAX_VIDEOS_PER_PROFILE", "100"))
    max_posts_per_profile: int = int(os.getenv("MAX_POSTS_PER_PROFILE", "100"))
    max_comments_per_post: int = int(os.getenv("MAX_COMMENTS_PER_POST", "50"))
    
    # Configuración de análisis
    min_content_for_analysis: int = int(os.getenv("MIN_CONTENT_FOR_ANALYSIS", "10"))
    analysis_depth: str = os.getenv("ANALYSIS_DEPTH", "deep")  # quick, standard, deep
    
    # Configuración de generación
    content_temperature: float = float(os.getenv("CONTENT_TEMPERATURE", "0.7"))
    max_content_length: int = int(os.getenv("MAX_CONTENT_LENGTH", "2000"))
    
    # Base de datos
    database_url: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./social_media_identity_clone.db"
    )
    
    # Storage
    storage_path: str = os.getenv("STORAGE_PATH", "./storage")
    
    # API Server
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8030"))
    api_debug: bool = os.getenv("API_DEBUG", "False").lower() == "true"
    
    # Rate Limiting
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Obtener instancia singleton de configuración"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings




