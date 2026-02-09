"""
Configuration System v2.0 - Optimized & Refactored
=================================================

Advanced configuration management with:
- Environment-based configuration
- Validation and type safety
- Hot reloading support
- Performance optimization settings
- Security configurations
- Monitoring and logging settings
"""

import os
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import yaml
from pydantic import BaseSettings, Field, validator
import redis
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AIClientType(str, Enum):
    """AI client types"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    LOCAL = "local"


class DatabaseType(str, Enum):
    """Database types"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"


class CacheType(str, Enum):
    """Cache types"""
    REDIS = "redis"
    MEMORY = "memory"
    FILE = "file"


@dataclass
class AIConfig:
    """AI configuration"""
    client_type: AIClientType = AIClientType.OPENAI
    api_key: str = ""
    base_url: Optional[str] = None
    model: str = "gpt-4"
    max_tokens: int = 4000
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000


@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: DatabaseType = DatabaseType.SQLITE
    url: str = "sqlite:///./workflow_chain.db"
    host: str = "localhost"
    port: int = 5432
    username: str = ""
    password: str = ""
    database: str = "workflow_chain"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    ssl_mode: str = "prefer"


@dataclass
class CacheConfig:
    """Cache configuration"""
    type: CacheType = CacheType.MEMORY
    redis_url: str = "redis://localhost:6379/0"
    max_size: int = 10000
    default_ttl: int = 3600
    compression: bool = True
    serialization: str = "json"  # json, pickle, msgpack


@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_min_length: int = 8
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 100
    rate_limit_per_hour: int = 1000


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enabled: bool = True
    metrics_endpoint: str = "/metrics"
    health_check_interval: int = 30
    performance_tracking: bool = True
    error_tracking: bool = True
    usage_analytics: bool = True
    log_requests: bool = True
    log_responses: bool = False
    slow_query_threshold: float = 1.0  # seconds


@dataclass
class PerformanceConfig:
    """Performance configuration"""
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    keep_alive_timeout: int = 5
    max_keep_alive_requests: int = 100
    worker_processes: int = 1
    worker_threads: int = 4
    memory_limit_mb: int = 512
    cpu_limit_percent: float = 80.0
    enable_compression: bool = True
    compression_level: int = 6
    enable_caching: bool = True
    cache_ttl: int = 300


class Settings(BaseSettings):
    """Main settings class with Pydantic validation"""
    
    # Application settings
    app_name: str = Field("Document Workflow Chain", description="Application name")
    app_version: str = Field("2.0.0", description="Application version")
    app_description: str = Field("High-performance document workflow chain system", description="Application description")
    debug: bool = Field(False, description="Debug mode")
    environment: str = Field("development", description="Environment (development, staging, production)")
    
    # Server settings
    host: str = Field("0.0.0.0", description="Server host")
    port: int = Field(8000, description="Server port")
    workers: int = Field(1, description="Number of worker processes")
    reload: bool = Field(False, description="Auto-reload on code changes")
    
    # Logging settings
    log_level: LogLevel = Field(LogLevel.INFO, description="Logging level")
    log_format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    log_file: Optional[str] = Field(None, description="Log file path")
    log_rotation: str = Field("daily", description="Log rotation (daily, weekly, monthly)")
    log_retention: int = Field(30, description="Log retention days")
    
    # AI settings
    ai_client_type: AIClientType = Field(AIClientType.OPENAI, description="AI client type")
    ai_api_key: str = Field("", description="AI API key")
    ai_base_url: Optional[str] = Field(None, description="AI base URL")
    ai_model: str = Field("gpt-4", description="AI model")
    ai_max_tokens: int = Field(4000, description="AI max tokens")
    ai_temperature: float = Field(0.7, description="AI temperature")
    ai_timeout: int = Field(30, description="AI timeout")
    ai_max_retries: int = Field(3, description="AI max retries")
    ai_rate_limit_per_minute: int = Field(60, description="AI rate limit per minute")
    
    # Database settings
    database_type: DatabaseType = Field(DatabaseType.SQLITE, description="Database type")
    database_url: str = Field("sqlite:///./workflow_chain.db", description="Database URL")
    database_host: str = Field("localhost", description="Database host")
    database_port: int = Field(5432, description="Database port")
    database_username: str = Field("", description="Database username")
    database_password: str = Field("", description="Database password")
    database_name: str = Field("workflow_chain", description="Database name")
    database_pool_size: int = Field(10, description="Database pool size")
    database_echo: bool = Field(False, description="Database echo SQL")
    
    # Cache settings
    cache_type: CacheType = Field(CacheType.MEMORY, description="Cache type")
    cache_redis_url: str = Field("redis://localhost:6379/0", description="Redis URL")
    cache_max_size: int = Field(10000, description="Cache max size")
    cache_default_ttl: int = Field(3600, description="Cache default TTL")
    cache_compression: bool = Field(True, description="Cache compression")
    
    # Security settings
    secret_key: str = Field("your-secret-key-change-in-production", description="Secret key")
    cors_origins: List[str] = Field(["*"], description="CORS origins")
    rate_limit_per_minute: int = Field(100, description="Rate limit per minute")
    rate_limit_per_hour: int = Field(1000, description="Rate limit per hour")
    
    # Performance settings
    max_concurrent_requests: int = Field(100, description="Max concurrent requests")
    request_timeout: int = Field(30, description="Request timeout")
    worker_threads: int = Field(4, description="Worker threads")
    memory_limit_mb: int = Field(512, description="Memory limit MB")
    enable_compression: bool = Field(True, description="Enable compression")
    enable_caching: bool = Field(True, description="Enable caching")
    
    # Monitoring settings
    monitoring_enabled: bool = Field(True, description="Enable monitoring")
    metrics_endpoint: str = Field("/metrics", description="Metrics endpoint")
    health_check_interval: int = Field(30, description="Health check interval")
    performance_tracking: bool = Field(True, description="Performance tracking")
    
    # Workflow settings
    max_chain_length: int = Field(100, description="Max chain length")
    max_nodes_per_chain: int = Field(1000, description="Max nodes per chain")
    default_workflow_timeout: int = Field(300, description="Default workflow timeout")
    enable_workflow_validation: bool = Field(True, description="Enable workflow validation")
    enable_content_analysis: bool = Field(True, description="Enable content analysis")
    enable_quality_control: bool = Field(True, description="Enable quality control")
    
    # File storage settings
    storage_type: str = Field("local", description="Storage type (local, s3, gcs)")
    storage_path: str = Field("./storage", description="Storage path")
    max_file_size_mb: int = Field(100, description="Max file size MB")
    allowed_file_types: List[str] = Field(["txt", "md", "json", "pdf"], description="Allowed file types")
    
    # Background tasks settings
    enable_background_tasks: bool = Field(True, description="Enable background tasks")
    task_queue_size: int = Field(1000, description="Task queue size")
    task_timeout: int = Field(300, description="Task timeout")
    max_concurrent_tasks: int = Field(10, description="Max concurrent tasks")
    
    # Email settings
    email_enabled: bool = Field(False, description="Enable email notifications")
    email_smtp_host: str = Field("", description="SMTP host")
    email_smtp_port: int = Field(587, description="SMTP port")
    email_username: str = Field("", description="Email username")
    email_password: str = Field("", description="Email password")
    email_from: str = Field("", description="Email from address")
    
    # Development settings
    enable_hot_reload: bool = Field(False, description="Enable hot reload")
    enable_profiling: bool = Field(False, description="Enable profiling")
    enable_debug_toolbar: bool = Field(False, description="Enable debug toolbar")
    mock_ai_responses: bool = Field(False, description="Mock AI responses")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Environment variable mapping
        fields = {
            "ai_api_key": {"env": "AI_API_KEY"},
            "ai_base_url": {"env": "AI_BASE_URL"},
            "database_url": {"env": "DATABASE_URL"},
            "secret_key": {"env": "SECRET_KEY"},
            "cache_redis_url": {"env": "REDIS_URL"},
        }
    
    @validator("ai_temperature")
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v
    
    @validator("ai_max_tokens")
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("Max tokens must be positive")
        return v
    
    @validator("port")
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @validator("workers")
    def validate_workers(cls, v):
        if v <= 0:
            raise ValueError("Workers must be positive")
        return v
    
    @validator("cors_origins")
    def validate_cors_origins(cls, v):
        if not isinstance(v, list):
            return [v] if v else ["*"]
        return v


class ConfigManager:
    """Configuration manager with hot reloading and validation"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.settings = Settings()
        self._last_modified = None
        self._watchers = []
        
        # Load configuration
        self._load_config()
        
        # Setup logging
        self._setup_logging()
    
    def _load_config(self):
        """Load configuration from file and environment"""
        if self.config_file and Path(self.config_file).exists():
            self._load_from_file()
        
        # Override with environment variables
        self._load_from_env()
        
        # Validate configuration
        self._validate_config()
    
    def _load_from_file(self):
        """Load configuration from file"""
        config_path = Path(self.config_file)
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Update settings with file data
        for key, value in config_data.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # This is handled automatically by Pydantic BaseSettings
        pass
    
    def _validate_config(self):
        """Validate configuration"""
        # Check required settings
        if not self.settings.secret_key or self.settings.secret_key == "your-secret-key-change-in-production":
            if self.settings.environment == "production":
                raise ValueError("Secret key must be set in production")
        
        # Check AI configuration
        if self.settings.ai_client_type != AIClientType.LOCAL and not self.settings.ai_api_key:
            logger.warning("AI API key not set - AI features may not work")
        
        # Check database configuration
        if self.settings.database_type == DatabaseType.SQLITE:
            # Ensure directory exists
            db_path = Path(self.settings.database_url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.settings.log_level.value)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=self.settings.log_format,
            handlers=[
                logging.StreamHandler(),
                *([logging.FileHandler(self.settings.log_file)] if self.settings.log_file else [])
            ]
        )
        
        # Set specific loggers
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("fastapi").setLevel(logging.INFO)
        logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    
    def get_ai_config(self) -> AIConfig:
        """Get AI configuration"""
        return AIConfig(
            client_type=self.settings.ai_client_type,
            api_key=self.settings.ai_api_key,
            base_url=self.settings.ai_base_url,
            model=self.settings.ai_model,
            max_tokens=self.settings.ai_max_tokens,
            temperature=self.settings.ai_temperature,
            timeout=self.settings.ai_timeout,
            max_retries=self.settings.ai_max_retries,
            rate_limit_per_minute=self.settings.ai_rate_limit_per_minute
        )
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        return DatabaseConfig(
            type=self.settings.database_type,
            url=self.settings.database_url,
            host=self.settings.database_host,
            port=self.settings.database_port,
            username=self.settings.database_username,
            password=self.settings.database_password,
            database=self.settings.database_name,
            pool_size=self.settings.database_pool_size,
            echo=self.settings.database_echo
        )
    
    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration"""
        return CacheConfig(
            type=self.settings.cache_type,
            redis_url=self.settings.cache_redis_url,
            max_size=self.settings.cache_max_size,
            default_ttl=self.settings.cache_default_ttl,
            compression=self.settings.cache_compression
        )
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        return SecurityConfig(
            secret_key=self.settings.secret_key,
            cors_origins=self.settings.cors_origins,
            rate_limit_per_minute=self.settings.rate_limit_per_minute,
            rate_limit_per_hour=self.settings.rate_limit_per_hour
        )
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration"""
        return PerformanceConfig(
            max_concurrent_requests=self.settings.max_concurrent_requests,
            request_timeout=self.settings.request_timeout,
            worker_threads=self.settings.worker_threads,
            memory_limit_mb=self.settings.memory_limit_mb,
            enable_compression=self.settings.enable_compression,
            enable_caching=self.settings.enable_caching
        )
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        return MonitoringConfig(
            enabled=self.settings.monitoring_enabled,
            metrics_endpoint=self.settings.metrics_endpoint,
            health_check_interval=self.settings.health_check_interval,
            performance_tracking=self.settings.performance_tracking
        )
    
    def reload(self):
        """Reload configuration"""
        self._load_config()
        logger.info("Configuration reloaded")
    
    def add_watcher(self, callback):
        """Add configuration change watcher"""
        self._watchers.append(callback)
    
    def remove_watcher(self, callback):
        """Remove configuration change watcher"""
        if callback in self._watchers:
            self._watchers.remove(callback)
    
    def notify_watchers(self):
        """Notify all watchers of configuration changes"""
        for callback in self._watchers:
            try:
                callback(self.settings)
            except Exception as e:
                logger.error(f"Error in config watcher: {e}")


# Global configuration instance
config_manager = ConfigManager()

# Convenience functions
def get_settings() -> Settings:
    """Get current settings"""
    return config_manager.settings

def get_ai_config() -> AIConfig:
    """Get AI configuration"""
    return config_manager.get_ai_config()

def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return config_manager.get_database_config()

def get_cache_config() -> CacheConfig:
    """Get cache configuration"""
    return config_manager.get_cache_config()

def get_security_config() -> SecurityConfig:
    """Get security configuration"""
    return config_manager.get_security_config()

def get_performance_config() -> PerformanceConfig:
    """Get performance configuration"""
    return config_manager.get_performance_config()

def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration"""
    return config_manager.get_monitoring_config()

def reload_config():
    """Reload configuration"""
    config_manager.reload()

def validate_config() -> bool:
    """Validate configuration"""
    try:
        config_manager._validate_config()
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


# Example usage
if __name__ == "__main__":
    # Test configuration
    settings = get_settings()
    print(f"App: {settings.app_name} v{settings.app_version}")
    print(f"Environment: {settings.environment}")
    print(f"Debug: {settings.debug}")
    print(f"AI Client: {settings.ai_client_type}")
    print(f"Database: {settings.database_type}")
    print(f"Cache: {settings.cache_type}")
    
    # Test validation
    if validate_config():
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration validation failed")




