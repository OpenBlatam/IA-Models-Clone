from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
from typing import Dict, List, Optional
from pydantic import BaseSettings, Field
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Configuration settings for HeyGen AI equivalent.
"""



class HeyGenAISettings(BaseSettings):
    """Settings for HeyGen AI system."""
    
    # API Settings
    api_host: str = Field(default="0.0.0.0", env="HEYGEN_API_HOST")
    api_port: int = Field(default=8000, env="HEYGEN_API_PORT")
    api_workers: int = Field(default=4, env="HEYGEN_API_WORKERS")
    debug: bool = Field(default=False, env="HEYGEN_DEBUG")
    
    # LangChain and OpenRouter Settings
    openrouter_api_key: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")
    openrouter_api_base: str = Field(default="https://openrouter.ai/api/v1", env="OPENROUTER_API_BASE")
    langchain_enabled: bool = Field(default=True, env="LANGCHAIN_ENABLED")
    langchain_cache_dir: str = Field(default="./langchain_cache", env="LANGCHAIN_CACHE_DIR")
    
    # AI Model Settings
    default_llm_model: str = Field(default="openai/gpt-4", env="DEFAULT_LLM_MODEL")
    default_embedding_model: str = Field(default="openai/text-embedding-ada-002", env="DEFAULT_EMBEDDING_MODEL")
    max_tokens: int = Field(default=4000, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    
    # Model Settings
    model_cache_dir: str = Field(default="./models", env="HEYGEN_MODEL_CACHE_DIR")
    gpu_enabled: bool = Field(default=True, env="HEYGEN_GPU_ENABLED")
    max_batch_size: int = Field(default=4, env="HEYGEN_MAX_BATCH_SIZE")
    
    # Video Settings
    default_resolution: str = Field(default="1080p", env="HEYGEN_DEFAULT_RESOLUTION")
    default_format: str = Field(default="mp4", env="HEYGEN_DEFAULT_FORMAT")
    max_video_duration: int = Field(default=600, env="HEYGEN_MAX_VIDEO_DURATION")  # 10 minutes
    temp_dir: str = Field(default="./temp", env="HEYGEN_TEMP_DIR")
    
    # Audio Settings
    default_sample_rate: int = Field(default=22050, env="HEYGEN_DEFAULT_SAMPLE_RATE")
    audio_quality: str = Field(default="high", env="HEYGEN_AUDIO_QUALITY")
    
    # Storage Settings
    storage_type: str = Field(default="local", env="HEYGEN_STORAGE_TYPE")  # local, s3, gcs, azure
    storage_bucket: Optional[str] = Field(default=None, env="HEYGEN_STORAGE_BUCKET")
    storage_region: Optional[str] = Field(default=None, env="HEYGEN_STORAGE_REGION")
    
    # Database Settings
    database_url: str = Field(default="sqlite:///heygen_ai.db", env="HEYGEN_DATABASE_URL")
    
    # Authentication
    api_key_required: bool = Field(default=False, env="HEYGEN_API_KEY_REQUIRED")
    api_key_header: str = Field(default="X-API-Key", env="HEYGEN_API_KEY_HEADER")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, env="HEYGEN_RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="HEYGEN_RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="HEYGEN_RATE_LIMIT_WINDOW")  # 1 hour
    
    # Logging
    log_level: str = Field(default="INFO", env="HEYGEN_LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="HEYGEN_LOG_FILE")
    
    # Monitoring
    metrics_enabled: bool = Field(default=True, env="HEYGEN_METRICS_ENABLED")
    sentry_dsn: Optional[str] = Field(default=None, env="HEYGEN_SENTRY_DSN")
    
    # AI Model Settings (Legacy)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    huggingface_token: Optional[str] = Field(default=None, env="HUGGINGFACE_TOKEN")
    
    # Avatar Settings
    avatar_models: List[str] = Field(default=[
        "stable_diffusion_v1_5",
        "wav2lip",
        "face_expression_model"
    ], env="HEYGEN_AVATAR_MODELS")
    
    # Voice Settings
    voice_models: List[str] = Field(default=[
        "coqui_tts",
        "your_tts",
        "emotion_tts"
    ], env="HEYGEN_VOICE_MODELS")
    
    # Script Generation Settings
    script_models: List[str] = Field(default=[
        "openai/gpt-4",
        "anthropic/claude-3-sonnet",
        "meta-llama/llama-2-70b-chat",
        "google/gemini-pro"
    ], env="HEYGEN_SCRIPT_MODELS")
    
    # Supported Languages
    supported_languages: List[str] = Field(default=[
        "en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"
    ], env="HEYGEN_SUPPORTED_LANGUAGES")
    
    # Video Styles
    video_styles: List[str] = Field(default=[
        "professional", "casual", "educational", "marketing", "entertainment"
    ], env="HEYGEN_VIDEO_STYLES")
    
    # Quality Settings
    video_quality_presets: Dict[str, Dict] = Field(default={
        "low": {
            "resolution": "720p",
            "bitrate": "1000k",
            "fps": 24
        },
        "medium": {
            "resolution": "1080p", 
            "bitrate": "2000k",
            "fps": 30
        },
        "high": {
            "resolution": "1080p",
            "bitrate": "4000k", 
            "fps": 30
        },
        "ultra": {
            "resolution": "4k",
            "bitrate": "8000k",
            "fps": 30
        }
    }, env="HEYGEN_VIDEO_QUALITY_PRESETS")
    
    # LangChain Specific Settings
    langchain_models: Dict[str, Dict] = Field(default={
        "gpt-4": {
            "model": "openai/gpt-4",
            "temperature": 0.7,
            "max_tokens": 4000
        },
        "gpt-3.5-turbo": {
            "model": "openai/gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 4000
        },
        "claude-3": {
            "model": "anthropic/claude-3-sonnet",
            "temperature": 0.7,
            "max_tokens": 4000
        },
        "llama-2": {
            "model": "meta-llama/llama-2-70b-chat",
            "temperature": 0.7,
            "max_tokens": 4000
        },
        "gemini": {
            "model": "google/gemini-pro",
            "temperature": 0.7,
            "max_tokens": 4000
        }
    }, env="LANGCHAIN_MODELS")
    
    # Vector Store Settings
    vector_store_type: str = Field(default="faiss", env="VECTOR_STORE_TYPE")
    vector_store_path: str = Field(default="./vector_stores", env="VECTOR_STORE_PATH")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    @dataclass
class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = HeyGenAISettings()


def get_settings() -> HeyGenAISettings:
    """Get settings instance."""
    return settings


# Environment-specific configurations
class DevelopmentSettings(HeyGenAISettings):
    """Development environment settings."""
    debug: bool = True
    log_level: str = "DEBUG"
    api_workers: int = 1
    langchain_enabled: bool = True


class ProductionSettings(HeyGenAISettings):
    """Production environment settings."""
    debug: bool = False
    log_level: str = "WARNING"
    api_workers: int = 8
    rate_limit_enabled: bool = True
    metrics_enabled: bool = True
    langchain_enabled: bool = True


class TestingSettings(HeyGenAISettings):
    """Testing environment settings."""
    debug: bool = True
    log_level: str = "DEBUG"
    api_workers: int = 1
    database_url: str = "sqlite:///test_heygen_ai.db"
    temp_dir: str = "./test_temp"
    langchain_enabled: bool = False  # Disable for testing


# Environment detection
def get_environment_settings() -> HeyGenAISettings:
    """Get environment-specific settings."""
    env = os.getenv("HEYGEN_ENV", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings() 