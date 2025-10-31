"""
Instagram Captions API v10.0 - Configuration Module

Centralized configuration management with environment variable support
and production-ready settings.
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field, validator
from pathlib import Path

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

class EnvironmentConfig(BaseSettings):
    """Environment-specific configuration."""
    
    # Environment
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # API Configuration
    API_VERSION: str = Field(default="10.0.0", env="API_VERSION")
    API_NAME: str = Field(default="Instagram Captions API v10.0", env="API_NAME")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8100, env="PORT")
    WORKERS: int = Field(default=1, env="WORKERS")
    
    # Security
    SECRET_KEY: str = Field(default_factory=lambda: os.urandom(32).hex(), env="SECRET_KEY")
    API_KEY_HEADER: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    CORS_ORIGINS: str = Field(default="*", env="CORS_ORIGINS")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    RATE_LIMIT_BURST: int = Field(default=20, env="RATE_LIMIT_BURST")
    
    # AI Configuration
    AI_MODEL_NAME: str = Field(default="gpt2", env="AI_MODEL_NAME")
    AI_PROVIDER: str = Field(default="local", env="AI_PROVIDER")
    MAX_TOKENS: int = Field(default=150, env="MAX_TOKENS")
    TEMPERATURE: float = Field(default=0.7, env="TEMPERATURE")
    
    # Performance
    CACHE_SIZE: int = Field(default=1000, env="CACHE_SIZE")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")
    MAX_BATCH_SIZE: int = Field(default=100, env="MAX_BATCH_SIZE")
    AI_WORKERS: int = Field(default=4, env="AI_WORKERS")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    
    # Database (if needed in future)
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # External Services
    REDIS_URL: Optional[str] = Field(default=None, env="REDIS_URL")
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=8101, env="METRICS_PORT")
    
    @validator('ENVIRONMENT')
    def validate_environment(cls, v):
        allowed = ['development', 'staging', 'production', 'testing']
        if v not in allowed:
            raise ValueError(f'Environment must be one of: {", ".join(allowed)}')
        return v
    
    @validator('CORS_ORIGINS')
    def parse_cors_origins(cls, v):
        if v == "*":
            return ["*"]
        return [origin.strip() for origin in v.split(",")]
    
    @validator('TEMPERATURE')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v
    
    @validator('MAX_TOKENS')
    def validate_max_tokens(cls, v):
        if v < 1 or v > 1000:
            raise ValueError('Max tokens must be between 1 and 1000')
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# =============================================================================
# PRODUCTION CONFIGURATION
# =============================================================================

class ProductionConfig(EnvironmentConfig):
    """Production-specific configuration."""
    
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "WARNING"
    
    # Production security
    CORS_ORIGINS: str = Field(default="https://yourdomain.com", env="CORS_ORIGINS")
    
    # Production performance
    WORKERS: int = Field(default=4, env="WORKERS")
    CACHE_SIZE: int = Field(default=10000, env="CACHE_SIZE")
    AI_WORKERS: int = Field(default=8, env="AI_WORKERS")
    
    # Production monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = Field(default=8101, env="METRICS_PORT")

# =============================================================================
# DEVELOPMENT CONFIGURATION
# =============================================================================

class DevelopmentConfig(EnvironmentConfig):
    """Development-specific configuration."""
    
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "DEBUG"
    
    # Development settings
    CORS_ORIGINS: str = "*"
    WORKERS: int = 1
    CACHE_SIZE: int = 100
    AI_WORKERS: int = 2

# =============================================================================
# TESTING CONFIGURATION
# =============================================================================

class TestingConfig(EnvironmentConfig):
    """Testing-specific configuration."""
    
    DEBUG: bool = True
    ENVIRONMENT: str = "testing"
    LOG_LEVEL: str = "DEBUG"
    
    # Testing settings
    CACHE_SIZE: int = 10
    AI_WORKERS: int = 1
    MAX_BATCH_SIZE: int = 5

# =============================================================================
# CONFIGURATION FACTORY
# =============================================================================

def get_config() -> EnvironmentConfig:
    """Get configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config(config: EnvironmentConfig) -> Dict[str, Any]:
    """Validate configuration and return validation results."""
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "recommendations": []
    }
    
    # Check required environment variables
    if config.ENVIRONMENT == "production":
        if not config.SECRET_KEY or config.SECRET_KEY == os.urandom(32).hex():
            validation_results["warnings"].append("SECRET_KEY should be set in production")
        
        if config.CORS_ORIGINS == ["*"]:
            validation_results["warnings"].append("CORS_ORIGINS should be restricted in production")
    
    # Check AI configuration
    if config.AI_PROVIDER == "openai" and not config.OPENAI_API_KEY:
        validation_results["errors"].append("OPENAI_API_KEY required for OpenAI provider")
    
    if config.AI_PROVIDER == "anthropic" and not config.ANTHROPIC_API_KEY:
        validation_results["errors"].append("ANTHROPIC_API_KEY required for Anthropic provider")
    
    # Performance recommendations
    if config.CACHE_SIZE < 100:
        validation_results["recommendations"].append("Consider increasing CACHE_SIZE for better performance")
    
    if config.AI_WORKERS < 2:
        validation_results["recommendations"].append("Consider increasing AI_WORKERS for better concurrency")
    
    # Update validation status
    if validation_results["errors"]:
        validation_results["valid"] = False
    
    return validation_results

# =============================================================================
# CONFIGURATION EXPORTS
# =============================================================================

# Default configuration instance
config = get_config()

# Export main classes and functions
__all__ = [
    'EnvironmentConfig',
    'ProductionConfig', 
    'DevelopmentConfig',
    'TestingConfig',
    'get_config',
    'validate_config',
    'config'
]






