"""
Configuration Management for Email Sequence System

This module provides centralized configuration management using Pydantic settings
with environment variable support and validation.
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field, validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = Field(default="Email Sequence AI", env="APP_NAME")
    app_version: str = Field(default="2.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    
    # Security
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")
    
    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost/email_sequences",
        env="DATABASE_URL"
    )
    db_pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")
    
    # OpenAI/LangChain
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=2000, env="OPENAI_MAX_TOKENS")
    
    # Email Delivery
    smtp_host: str = Field(default="localhost", env="SMTP_HOST")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_username: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    smtp_use_tls: bool = Field(default=True, env="SMTP_USE_TLS")
    from_email: str = Field(default="noreply@example.com", env="FROM_EMAIL")
    from_name: str = Field(default="Email Sequence System", env="FROM_NAME")
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(default=60, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    rate_limit_burst: int = Field(default=100, env="RATE_LIMIT_BURST")
    
    # Performance
    max_concurrent_sequences: int = Field(default=50, env="MAX_CONCURRENT_SEQUENCES")
    max_concurrent_emails: int = Field(default=100, env="MAX_CONCURRENT_EMAILS")
    email_batch_size: int = Field(default=100, env="EMAIL_BATCH_SIZE")
    email_delay_between_batches: float = Field(default=1.0, env="EMAIL_DELAY_BETWEEN_BATCHES")
    
    # Caching
    cache_ttl_seconds: int = Field(default=300, env="CACHE_TTL_SECONDS")
    cache_sequence_ttl: int = Field(default=3600, env="CACHE_SEQUENCE_TTL")
    cache_analytics_ttl: int = Field(default=1800, env="CACHE_ANALYTICS_TTL")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    enable_tracing: bool = Field(default=False, env="ENABLE_TRACING")
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        env="CORS_ALLOW_METHODS"
    )
    cors_allow_headers: List[str] = Field(
        default=["*"],
        env="CORS_ALLOW_HEADERS"
    )
    
    # File Upload
    max_file_size: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    allowed_file_types: List[str] = Field(
        default=["csv", "xlsx", "json"],
        env="ALLOWED_FILE_TYPES"
    )
    
    # Webhooks
    webhook_timeout: int = Field(default=30, env="WEBHOOK_TIMEOUT")
    webhook_max_retries: int = Field(default=3, env="WEBHOOK_MAX_RETRIES")
    webhook_retry_delay: float = Field(default=1.0, env="WEBHOOK_RETRY_DELAY")
    
    # Analytics
    analytics_batch_size: int = Field(default=1000, env="ANALYTICS_BATCH_SIZE")
    analytics_flush_interval: int = Field(default=60, env="ANALYTICS_FLUSH_INTERVAL")
    
    # A/B Testing
    ab_test_min_sample_size: int = Field(default=100, env="AB_TEST_MIN_SAMPLE_SIZE")
    ab_test_confidence_level: float = Field(default=0.95, env="AB_TEST_CONFIDENCE_LEVEL")
    
    @validator('environment')
    def validate_environment(cls, v):
        allowed_environments = ['development', 'staging', 'production']
        if v not in allowed_environments:
            raise ValueError(f'Environment must be one of {allowed_environments}')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed_levels:
            raise ValueError(f'Log level must be one of {allowed_levels}')
        return v.upper()
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('cors_allow_methods', pre=True)
    def parse_cors_methods(cls, v):
        if isinstance(v, str):
            return [method.strip() for method in v.split(',')]
        return v
    
    @validator('cors_allow_headers', pre=True)
    def parse_cors_headers(cls, v):
        if isinstance(v, str):
            return [header.strip() for header in v.split(',')]
        return v
    
    @validator('allowed_file_types', pre=True)
    def parse_file_types(cls, v):
        if isinstance(v, str):
            return [file_type.strip() for file_type in v.split(',')]
        return v
    
    @validator('database_url')
    def validate_database_url(cls, v):
        if not v.startswith(('postgresql://', 'postgresql+asyncpg://', 'sqlite://')):
            raise ValueError('Database URL must be a valid PostgreSQL or SQLite URL')
        return v
    
    @validator('redis_url')
    def validate_redis_url(cls, v):
        if not v.startswith('redis://'):
            raise ValueError('Redis URL must start with redis://')
        return v
    
    @validator('openai_api_key')
    def validate_openai_key(cls, v):
        if not v and os.getenv('ENVIRONMENT') == 'production':
            raise ValueError('OpenAI API key is required in production')
        return v
    
    @validator('secret_key')
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError('Secret key must be at least 32 characters long')
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings"""
    debug: bool = True
    environment: str = "development"
    log_level: str = "DEBUG"
    database_url: str = "postgresql+asyncpg://user:password@localhost/email_sequences_dev"
    redis_url: str = "redis://localhost:6379/1"


class StagingSettings(Settings):
    """Staging environment settings"""
    debug: bool = False
    environment: str = "staging"
    log_level: str = "INFO"
    database_url: str = "postgresql+asyncpg://user:password@staging-db/email_sequences_staging"
    redis_url: str = "redis://staging-redis:6379/0"


class ProductionSettings(Settings):
    """Production environment settings"""
    debug: bool = False
    environment: str = "production"
    log_level: str = "WARNING"
    database_url: str = "postgresql+asyncpg://user:password@prod-db/email_sequences_prod"
    redis_url: str = "redis://prod-redis:6379/0"
    rate_limit_requests_per_minute: int = 30
    max_concurrent_sequences: int = 100
    max_concurrent_emails: int = 200


def get_environment_settings() -> Settings:
    """Get settings based on environment"""
    environment = os.getenv('ENVIRONMENT', 'development').lower()
    
    if environment == 'production':
        return ProductionSettings()
    elif environment == 'staging':
        return StagingSettings()
    else:
        return DevelopmentSettings()






























