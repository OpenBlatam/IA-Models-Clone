"""
Settings Management
==================

Centralized settings management for the Bulk TruthGPT system.
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
from pydantic import BaseSettings, Field, validator

class Environment(str, Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str
    pool_size: int = 20
    max_overflow: int = 30
    echo: bool = False
    pool_timeout: int = 30
    pool_recycle: int = 3600

@dataclass
class RedisConfig:
    """Redis configuration."""
    url: str
    pool_size: int = 10
    decode_responses: bool = True
    socket_timeout: int = 5
    socket_connect_timeout: int = 5

@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "info"

@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str
    access_token_expire_minutes: int = 30
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600

@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enabled: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    prometheus_enabled: bool = True
    grafana_enabled: bool = True

@dataclass
class TruthGPTConfig:
    """TruthGPT configuration."""
    model_path: str = "./models/truthgpt"
    device: str = "cuda"
    batch_size: int = 4
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

@dataclass
class GenerationConfig:
    """Generation configuration."""
    max_documents_per_task: int = 1000
    max_concurrent_tasks: int = 10
    timeout: int = 300
    min_quality_score: float = 0.7
    auto_optimization: bool = True

@dataclass
class StorageConfig:
    """Storage configuration."""
    storage_path: str = "./storage"
    templates_path: str = "./templates"
    models_path: str = "./models"
    knowledge_path: str = "./knowledge_base"
    logs_path: str = "./logs"

class Settings(BaseSettings):
    """
    Application settings.
    
    Uses Pydantic for validation and environment variable loading.
    """
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    
    # Database Configuration
    database_url: str = Field(default="postgresql://user:password@localhost/bulk_truthgpt", env="DATABASE_URL")
    database_pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_pool_size: int = Field(default=10, env="REDIS_POOL_SIZE")
    
    # Security Configuration
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    
    # TruthGPT Configuration
    truthgpt_model_path: str = Field(default="./models/truthgpt", env="TRUTHGPT_MODEL_PATH")
    truthgpt_device: str = Field(default="cuda", env="TRUTHGPT_DEVICE")
    truthgpt_batch_size: int = Field(default=4, env="TRUTHGPT_BATCH_SIZE")
    truthgpt_max_length: int = Field(default=2048, env="TRUTHGPT_MAX_LENGTH")
    
    # Generation Configuration
    max_documents_per_task: int = Field(default=1000, env="MAX_DOCUMENTS_PER_TASK")
    max_concurrent_tasks: int = Field(default=10, env="MAX_CONCURRENT_TASKS")
    document_generation_timeout: int = Field(default=300, env="DOCUMENT_GENERATION_TIMEOUT")
    min_quality_score: float = Field(default=0.7, env="MIN_QUALITY_SCORE")
    
    # Monitoring Configuration
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    # Logging Configuration
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: str = Field(default="./logs/bulk_truthgpt.log", env="LOG_FILE")
    log_max_size: str = Field(default="100MB", env="LOG_MAX_SIZE")
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # Storage Configuration
    storage_path: str = Field(default="./storage", env="STORAGE_PATH")
    templates_path: str = Field(default="./templates", env="TEMPLATES_PATH")
    models_path: str = Field(default="./models", env="MODELS_PATH")
    knowledge_path: str = Field(default="./knowledge_base", env="KNOWLEDGE_PATH")
    logs_path: str = Field(default="./logs", env="LOGS_PATH")
    
    # Performance Configuration
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    cache_max_size: int = Field(default=1000, env="CACHE_MAX_SIZE")
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")
    
    # Feature Flags
    learning_enabled: bool = Field(default=True, env="LEARNING_ENABLED")
    optimization_enabled: bool = Field(default=True, env="OPTIMIZATION_ENABLED")
    analytics_enabled: bool = Field(default=True, env="ANALYTICS_ENABLED")
    notification_enabled: bool = Field(default=True, env="NOTIFICATION_ENABLED")
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('log_level', pre=True)
    def parse_log_level(cls, v):
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v
    
    @validator('environment', pre=True)
    def parse_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return DatabaseConfig(
            url=self.database_url,
            pool_size=self.database_pool_size,
            max_overflow=self.database_max_overflow,
            echo=self.debug
        )
    
    def get_redis_config(self) -> RedisConfig:
        """Get Redis configuration."""
        return RedisConfig(
            url=self.redis_url,
            pool_size=self.redis_pool_size
        )
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration."""
        return APIConfig(
            host=self.api_host,
            port=self.api_port,
            workers=self.api_workers,
            reload=self.debug
        )
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        return SecurityConfig(
            secret_key=self.secret_key,
            access_token_expire_minutes=self.access_token_expire_minutes,
            cors_origins=self.cors_origins,
            rate_limit_requests=self.rate_limit_requests,
            rate_limit_window=self.rate_limit_window
        )
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        return MonitoringConfig(
            enabled=self.metrics_enabled,
            metrics_port=self.metrics_port,
            health_check_interval=self.health_check_interval
        )
    
    def get_truthgpt_config(self) -> TruthGPTConfig:
        """Get TruthGPT configuration."""
        return TruthGPTConfig(
            model_path=self.truthgpt_model_path,
            device=self.truthgpt_device,
            batch_size=self.truthgpt_batch_size,
            max_length=self.truthgpt_max_length
        )
    
    def get_generation_config(self) -> GenerationConfig:
        """Get generation configuration."""
        return GenerationConfig(
            max_documents_per_task=self.max_documents_per_task,
            max_concurrent_tasks=self.max_concurrent_tasks,
            timeout=self.document_generation_timeout,
            min_quality_score=self.min_quality_score,
            auto_optimization=self.optimization_enabled
        )
    
    def get_storage_config(self) -> StorageConfig:
        """Get storage configuration."""
        return StorageConfig(
            storage_path=self.storage_path,
            templates_path=self.templates_path,
            models_path=self.models_path,
            knowledge_path=self.knowledge_path,
            logs_path=self.logs_path
        )
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        try:
            # Check required paths
            required_paths = [
                self.storage_path,
                self.templates_path,
                self.models_path,
                self.knowledge_path,
                self.logs_path
            ]
            
            for path in required_paths:
                Path(path).mkdir(parents=True, exist_ok=True)
            
            # Check required environment variables
            if not self.secret_key or self.secret_key == "your-secret-key-here":
                raise ValueError("SECRET_KEY must be set to a secure value")
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {str(e)}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return self.dict()
    
    def save_to_file(self, file_path: str) -> None:
        """Save settings to file."""
        with open(file_path, 'w') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            else:
                json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'Settings':
        """Load settings from file."""
        with open(file_path, 'r') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls(**data)

# Global settings instance
settings = Settings()