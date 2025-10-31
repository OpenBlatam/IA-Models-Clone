"""
BUL System - Practical Configuration
Real, practical configuration management for the BUL system
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field
import logging

logger = logging.getLogger(__name__)

class DatabaseConfig(BaseSettings):
    """Real database configuration"""
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    name: str = Field(default="bul_db", env="DB_NAME")
    user: str = Field(default="bul_user", env="DB_USER")
    password: str = Field(default="bul_password", env="DB_PASSWORD")
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    
    class Config:
        env_file = ".env"

class RedisConfig(BaseSettings):
    """Real Redis configuration"""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")
    
    class Config:
        env_file = ".env"

class APIConfig(BaseSettings):
    """Real API configuration"""
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=4, env="API_WORKERS")
    max_requests: int = Field(default=1000, env="API_MAX_REQUESTS")
    timeout: int = Field(default=30, env="API_TIMEOUT")
    
    class Config:
        env_file = ".env"

class SecurityConfig(BaseSettings):
    """Real security configuration"""
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    
    class Config:
        env_file = ".env"

class LoggingConfig(BaseSettings):
    """Real logging configuration"""
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    file_path: Optional[str] = Field(default=None, env="LOG_FILE_PATH")
    max_file_size: int = Field(default=10485760, env="LOG_MAX_FILE_SIZE")  # 10MB
    backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    class Config:
        env_file = ".env"

class AIConfig(BaseSettings):
    """Real AI configuration"""
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    max_tokens: int = Field(default=2000, env="AI_MAX_TOKENS")
    temperature: float = Field(default=0.7, env="AI_TEMPERATURE")
    timeout: int = Field(default=30, env="AI_TIMEOUT")
    
    class Config:
        env_file = ".env"

class PracticalConfig:
    """Real, practical configuration manager"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.api = APIConfig()
        self.security = SecurityConfig()
        self.logging = LoggingConfig()
        self.ai = AIConfig()
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings"""
        try:
            # Check required settings
            if not self.security.secret_key or self.security.secret_key == "your-secret-key-here":
                logger.warning("Using default secret key. Please set SECRET_KEY environment variable.")
            
            if not self.ai.openai_api_key:
                logger.warning("OpenAI API key not set. AI features may not work.")
            
            # Validate numeric ranges
            if self.api.workers < 1 or self.api.workers > 32:
                raise ValueError("API workers must be between 1 and 32")
            
            if self.database.pool_size < 1 or self.database.pool_size > 100:
                raise ValueError("Database pool size must be between 1 and 100")
            
            logger.info("Configuration validated successfully")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        return f"postgresql://{self.database.user}:{self.database.password}@{self.database.host}:{self.database.port}/{self.database.name}"
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.redis.password:
            return f"redis://:{self.redis.password}@{self.redis.host}:{self.redis.port}/{self.redis.db}"
        return f"redis://{self.redis.host}:{self.redis.port}/{self.redis.db}"
    
    def get_ai_config(self) -> Dict[str, Any]:
        """Get AI configuration dictionary"""
        return {
            "api_key": self.ai.openai_api_key,
            "model": self.ai.openai_model,
            "max_tokens": self.ai.max_tokens,
            "temperature": self.ai.temperature,
            "timeout": self.ai.timeout
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration dictionary"""
        return {
            "secret_key": self.security.secret_key,
            "algorithm": self.security.algorithm,
            "access_token_expire_minutes": self.security.access_token_expire_minutes,
            "refresh_token_expire_days": self.security.refresh_token_expire_days
        }
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return os.getenv("ENVIRONMENT", "development").lower() == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    def get_environment(self) -> str:
        """Get current environment"""
        return os.getenv("ENVIRONMENT", "development").lower()

# Global configuration instance
config = PracticalConfig()

# Export commonly used configurations
DATABASE_URL = config.get_database_url()
REDIS_URL = config.get_redis_url()
AI_CONFIG = config.get_ai_config()
SECURITY_CONFIG = config.get_security_config()
ENVIRONMENT = config.get_environment()













