"""
Configuración del sistema de análisis musical
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración de la aplicación"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True
    )
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8010
    
    # Spotify API
    SPOTIFY_CLIENT_ID: Optional[str] = ""
    SPOTIFY_CLIENT_SECRET: Optional[str] = ""
    SPOTIFY_REDIRECT_URI: Optional[str] = "http://localhost:8010/callback"
    
    # Audio Analysis
    AUDIO_SAMPLE_RATE: int = 44100
    AUDIO_CHUNK_SIZE: int = 1024
    
    # Analysis Settings
    MAX_AUDIO_DURATION: int = 300  # 5 minutos máximo
    ANALYSIS_DETAIL_LEVEL: str = "detailed"  # basic, detailed, expert
    
    # Database (opcional)
    DATABASE_URL: Optional[str] = "sqlite:///./music_analyzer.db"
    
    # Cache
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600  # 1 hora
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = "music_analyzer.log"


settings = Settings()

