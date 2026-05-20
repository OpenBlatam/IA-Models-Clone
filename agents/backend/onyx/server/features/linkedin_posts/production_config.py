from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import os
from typing import List, Optional
from pydantic import BaseSettings, validator
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Production Configuration
========================

Complete production configuration for LinkedIn Posts system.
"""



class ProductionSettings(BaseSettings):
    """Production settings with validation."""
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    WORKER_CLASS: str = "uvicorn.workers.UvicornWorker"
    
    # Database Configuration
    DATABASE_URL: str = "postgresql://user:password@localhost/linkedin_posts"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 30
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_POOL_RECYCLE: int = 3600
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_POOL_SIZE: int = 10
    REDIS_MAX_CONNECTIONS: int = 20
    REDIS_RETRY_ON_TIMEOUT: bool = True
    
    # Cache Configuration
    CACHE_TTL: int = 3600  # 1 hour
    CACHE_MAX_SIZE: int = 10000
    CACHE_ENABLE_COMPRESSION: bool = True
    
    # NLP Configuration
    NLP_MODEL_PATH: str = "models/nlp/"
    NLP_BATCH_SIZE: int = 32
    NLP_MAX_LENGTH: int = 512
    NLP_USE_GPU: bool = False
    
    # API Configuration
    API_VERSION: str = "v1"
    API_PREFIX: str = "/api"
    API_TITLE: str = "LinkedIn Posts API"
    API_DESCRIPTION: str = "Ultra-optimized LinkedIn Posts management system"
    
    # Security Configuration
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "https://yourdomain.com",
        "https://app.yourdomain.com"
    ]
    ALLOWED_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    ALLOWED_HEADERS: List[str] = ["*"]
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    RATE_LIMIT_BURST: int = 200
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "logs/linkedin_posts.log"
    LOG_MAX_SIZE: int = 100 * 1024 * 1024  # 100MB
    LOG_BACKUP_COUNT: int = 5
    
    # Monitoring Configuration
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    HEALTH_CHECK_INTERVAL: int = 30
    HEALTH_CHECK_TIMEOUT: int = 5
    
    # Performance Configuration
    MAX_CONCURRENT_REQUESTS: int = 1000
    REQUEST_TIMEOUT: int = 30
    RESPONSE_TIMEOUT: int = 60
    ENABLE_COMPRESSION: bool = True
    COMPRESSION_MIN_SIZE: int = 1000
    
    # Background Tasks
    ENABLE_BACKGROUND_TASKS: bool = True
    BACKGROUND_WORKERS: int = 4
    TASK_QUEUE_SIZE: int = 1000
    TASK_TIMEOUT: int = 300
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = ["jpg", "jpeg", "png", "gif", "pdf"]
    UPLOAD_DIR: str = "uploads/"
    
    # Email Configuration
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = "your-email@gmail.com"
    SMTP_PASSWORD: str = "your-app-password"
    SMTP_USE_TLS: bool = True
    
    # External APIs
    OPENAI_API_KEY: Optional[str] = None
    LINKEDIN_API_KEY: Optional[str] = None
    LINKEDIN_API_SECRET: Optional[str] = None
    
    # Feature Flags
    ENABLE_NLP_OPTIMIZATION: bool = True
    ENABLE_AB_TESTING: bool = True
    ENABLE_ENGAGEMENT_ANALYSIS: bool = True
    ENABLE_AUTO_POSTING: bool = False
    
    # Development/Production
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    TESTING: bool = False
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v) -> bool:
        if not v or v == "postgresql://user:password@localhost/linkedin_posts":
            raise ValueError("DATABASE_URL must be properly configured for production")
        return v
    
    @validator("SECRET_KEY")
    def validate_secret_key(cls, v) -> bool:
        if v == "your-secret-key-here":
            raise ValueError("SECRET_KEY must be properly configured for production")
        return v
    
    @validator("REDIS_URL")
    def validate_redis_url(cls, v) -> bool:
        if not v or v == "redis://localhost:6379/0":
            raise ValueError("REDIS_URL must be properly configured for production")
        return v
    
    @dataclass
class Config:
        env_file = ".env"
        case_sensitive = True


class DevelopmentSettings(ProductionSettings):
    """Development settings with relaxed validation."""
    
    # Override production settings for development
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    
    # Use local database
    DATABASE_URL: str = "postgresql://dev:dev@localhost/linkedin_posts_dev"
    
    # Use local Redis
    REDIS_URL: str = "redis://localhost:6379/1"
    
    # Relaxed validation for development
    @validator("DATABASE_URL")
    def validate_database_url(cls, v) -> bool:
        return v
    
    @validator("SECRET_KEY")
    def validate_secret_key(cls, v) -> bool:
        return v
    
    @validator("REDIS_URL")
    def validate_redis_url(cls, v) -> bool:
        return v


class TestingSettings(ProductionSettings):
    """Testing settings."""
    
    # Override for testing
    TESTING: bool = True
    ENVIRONMENT: str = "testing"
    
    # Use test database
    DATABASE_URL: str = "postgresql://test:test@localhost/linkedin_posts_test"
    
    # Use test Redis
    REDIS_URL: str = "redis://localhost:6379/2"
    
    # Disable external APIs for testing
    ENABLE_NLP_OPTIMIZATION: bool = False
    ENABLE_AB_TESTING: bool = False
    ENABLE_ENGAGEMENT_ANALYSIS: bool = False
    ENABLE_AUTO_POSTING: bool = False
    
    # Relaxed validation for testing
    @validator("DATABASE_URL")
    def validate_database_url(cls, v) -> bool:
        return v
    
    @validator("SECRET_KEY")
    def validate_secret_key(cls, v) -> bool:
        return v
    
    @validator("REDIS_URL")
    def validate_redis_url(cls, v) -> bool:
        return v


def get_settings() -> ProductionSettings:
    """Get settings based on environment."""
    environment = os.getenv("ENVIRONMENT", "production").lower()
    
    if environment == "development":
        return DevelopmentSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return ProductionSettings()


# Global settings instance
settings = get_settings() 