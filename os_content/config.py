#!/usr/bin/env python3
"""
Configuration management for OS Content System
Centralized configuration with environment variable support
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseSettings, Field, validator
from pydantic_settings import BaseSettings as PydanticBaseSettings
import yaml
import json

class DatabaseConfig(BaseSettings):
    """Database configuration settings"""
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    username: str = Field(default="postgres", env="DB_USERNAME")
    password: str = Field(default="", env="DB_PASSWORD")
    database: str = Field(default="os_content", env="DB_NAME")
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    
    @property
    def connection_string(self) -> str:
        """Generate database connection string"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class RedisConfig(BaseSettings):
    """Redis configuration settings"""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")
    pool_size: int = Field(default=10, env="REDIS_POOL_SIZE")
    
    @property
    def connection_string(self) -> str:
        """Generate Redis connection string"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

class APIConfig(BaseSettings):
    """API configuration settings"""
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="API_DEBUG")
    workers: int = Field(default=1, env="API_WORKERS")
    reload: bool = Field(default=False, env="API_RELOAD")
    cors_origins: List[str] = Field(default=["*"], env="API_CORS_ORIGINS")
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v

class SecurityConfig(BaseSettings):
    """Security configuration settings"""
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    
    @validator('secret_key')
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError('Secret key must be at least 32 characters long')
        return v

class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration"""
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()

class MLConfig(BaseSettings):
    """Machine Learning configuration"""
    model_cache_dir: str = Field(default="./models", env="MODEL_CACHE_DIR")
    gpu_enabled: bool = Field(default=False, env="GPU_ENABLED")
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    model_timeout: int = Field(default=300, env="MODEL_TIMEOUT")
    
    @property
    def model_path(self) -> Path:
        """Get model cache directory path"""
        return Path(self.model_cache_dir)

class CacheConfig(BaseSettings):
    """Cache configuration settings"""
    l1_cache_size: int = Field(default=1000, env="L1_CACHE_SIZE")
    l2_cache_ttl: int = Field(default=3600, env="L2_CACHE_TTL")
    l3_cache_ttl: int = Field(default=86400, env="L3_CACHE_TTL")
    compression_enabled: bool = Field(default=True, env="COMPRESSION_ENABLED")
    compression_algorithm: str = Field(default="zstandard", env="COMPRESSION_ALGORITHM")

class OSContentConfig(BaseSettings):
    """Main configuration class for OS Content System"""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    app_name: str = Field(default="OS Content System", env="APP_NAME")
    version: str = Field(default="1.0.0", env="APP_VERSION")
    
    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    api: APIConfig = APIConfig()
    security: SecurityConfig = SecurityConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    ml: MLConfig = MLConfig()
    cache: CacheConfig = CacheConfig()
    
    # File paths
    base_dir: Path = Field(default=Path(__file__).parent.parent)
    logs_dir: Path = Field(default=Path("logs"))
    data_dir: Path = Field(default=Path("data"))
    temp_dir: Path = Field(default=Path("temp"))
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        for directory in [self.logs_dir, self.data_dir, self.temp_dir, self.ml.model_path]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "environment": self.environment,
            "app_name": self.app_name,
            "version": self.version,
            "database": self.database.dict(),
            "redis": self.redis.dict(),
            "api": self.api.dict(),
            "security": self.security.dict(),
            "monitoring": self.monitoring.dict(),
            "ml": self.ml.dict(),
            "cache": self.cache.dict(),
            "paths": {
                "base_dir": str(self.base_dir),
                "logs_dir": str(self.logs_dir),
                "data_dir": str(self.data_dir),
                "temp_dir": str(self.temp_dir),
                "model_path": str(self.ml.model_path)
            }
        }
    
    def save_to_file(self, file_path: Union[str, Path], format: str = "yaml"):
        """Save configuration to file"""
        file_path = Path(file_path)
        config_dict = self.to_dict()
        
        if format.lower() == "yaml":
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'OSContentConfig':
        """Load configuration from file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        if file_path.suffix.lower() == ".yaml" or file_path.suffix.lower() == ".yml":
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        elif file_path.suffix.lower() == ".json":
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return cls(**config_dict)

# Global configuration instance
config = OSContentConfig()

def get_config() -> OSContentConfig:
    """Get global configuration instance"""
    return config

def reload_config() -> OSContentConfig:
    """Reload configuration from environment variables"""
    global config
    config = OSContentConfig()
    return config
