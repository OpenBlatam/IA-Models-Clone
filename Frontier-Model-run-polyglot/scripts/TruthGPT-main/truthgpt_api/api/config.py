"""
API Configuration
================

Configuration settings for the TruthGPT API server.
"""

from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field

from .constants import (
    DEFAULT_MODEL_DIR, DEFAULT_CACHE_TTL, DEFAULT_RATE_LIMIT_PER_MINUTE,
    MAX_MODEL_SIZE_MB, MAX_TRAINING_SAMPLES, MAX_BATCH_SIZE
)


class Settings(BaseSettings):
    """Application settings."""
    
    app_name: str = "TruthGPT API"
    app_version: str = "1.0.0"
    app_description: str = "REST API for TruthGPT - TensorFlow-like interface for neural networks"
    
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = Field(default_factory=lambda: ["*"])
    cors_allow_headers: List[str] = Field(default_factory=lambda: ["*"])
    
    default_model_dir: str = DEFAULT_MODEL_DIR
    
    rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE
    cache_ttl: int = DEFAULT_CACHE_TTL
    
    max_model_size_mb: int = MAX_MODEL_SIZE_MB
    max_training_samples: int = MAX_TRAINING_SAMPLES
    max_batch_size: int = MAX_BATCH_SIZE
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()

