"""
Core Configuration
==================

Simple and clear configuration for the Document Workflow Chain system.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings - simple and clear"""
    
    # Application
    APP_NAME: str = "Document Workflow Chain v3.0+"
    VERSION: str = "3.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/workflow_chain"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 30
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_POOL_SIZE: int = 20
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # API
    API_V1_STR: str = "/api/v1"
    API_V3_STR: str = "/api/v3"
    
    # AI Services
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    
    # File Storage
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


