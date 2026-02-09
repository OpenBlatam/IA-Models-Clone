"""
Configuration settings for Social Video Transcriber AI
"""

from pydantic_settings import BaseSettings
from typing import Optional, List
from pathlib import Path
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    api_title: str = "Social Video Transcriber AI API"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    environment: str = "development"
    
    # OpenRouter Configuration
    openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_default_model: str = "anthropic/claude-3.5-sonnet"
    openrouter_site_url: str = os.getenv("OPENROUTER_SITE_URL", "https://blatam-academy.com")
    openrouter_app_name: str = os.getenv("OPENROUTER_APP_NAME", "Social Video Transcriber AI")
    
    # Whisper Configuration (for transcription)
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large
    whisper_language: Optional[str] = None  # Auto-detect if None
    
    # Storage
    temp_dir: str = "/tmp/social_video_transcriber"
    downloads_dir: str = "/tmp/social_video_transcriber/downloads"
    audio_dir: str = "/tmp/social_video_transcriber/audio"
    transcriptions_dir: str = "/tmp/social_video_transcriber/transcriptions"
    
    # Video Download Settings
    max_video_duration: int = 3600  # 1 hour max
    max_video_size_mb: int = 500
    download_timeout: int = 300  # 5 minutes
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 30
    rate_limit_per_hour: int = 200
    
    # Security
    cors_origins: List[str] = ["*"]
    api_key_required: bool = False
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Platform-specific settings
    tiktok_enabled: bool = True
    instagram_enabled: bool = True
    youtube_enabled: bool = True
    
    # Transcription Settings
    include_timestamps_default: bool = True
    timestamp_format: str = "[{start} -> {end}]"  # Format for timestamps
    chunk_duration: int = 30  # Seconds per chunk for long videos
    
    # AI Analysis Settings
    analysis_max_tokens: int = 4000
    variant_max_tokens: int = 2000
    temperature: float = 0.7
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton)"""
    global _settings
    if _settings is None:
        _settings = Settings()
        # Create directories
        for dir_path in [
            _settings.temp_dir,
            _settings.downloads_dir,
            _settings.audio_dir,
            _settings.transcriptions_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    return _settings












