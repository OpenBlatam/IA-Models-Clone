"""
Configuration Manager for Refactored Opus Clip

Centralized configuration management with:
- Environment-based configuration
- Validation and type checking
- Hot reloading capabilities
- Secret management
- Performance tuning
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Union, List
import os
import yaml
import json
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import structlog
from pydantic import BaseModel, Field, validator
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = structlog.get_logger("config_manager")

class Environment(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "opus_clip"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False

@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    max_connections: int = 100
    socket_timeout: int = 5

@dataclass
class AIConfig:
    """AI models configuration."""
    whisper_model: str = "base"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    device: str = "auto"  # auto, cpu, cuda
    batch_size: int = 1
    max_length: int = 512
    temperature: float = 0.7

@dataclass
class VideoConfig:
    """Video processing configuration."""
    max_duration: float = 300.0  # 5 minutes
    min_duration: float = 1.0    # 1 second
    max_clips: int = 50
    segment_duration: float = 5.0
    engagement_threshold: float = 0.3
    quality_settings: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "low": {"bitrate": "800k", "resolution": "480p"},
        "medium": {"bitrate": "1500k", "resolution": "720p"},
        "high": {"bitrate": "3000k", "resolution": "1080p"},
        "ultra": {"bitrate": "5000k", "resolution": "4k"}
    })

@dataclass
class PerformanceConfig:
    """Performance configuration."""
    max_workers: int = 4
    max_memory_mb: int = 8192
    enable_gpu: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_monitoring: bool = True
    log_level: str = "INFO"

@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = ""
    jwt_secret: str = ""
    jwt_expiry_hours: int = 24
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour

class ConfigManager:
    """
    Centralized configuration manager for Opus Clip.
    
    Features:
    - Environment-based configuration
    - Hot reloading
    - Validation
    - Secret management
    - Performance tuning
    """
    
    def __init__(self, config_path: Optional[str] = None, 
                 environment: Optional[Environment] = None):
        """Initialize configuration manager."""
        self.config_path = config_path or "config.yaml"
        self.environment = environment or self._detect_environment()
        self.logger = structlog.get_logger("config_manager")
        
        # Configuration sections
        self.database: Optional[DatabaseConfig] = None
        self.redis: Optional[RedisConfig] = None
        self.ai: Optional[AIConfig] = None
        self.video: Optional[VideoConfig] = None
        self.performance: Optional[PerformanceConfig] = None
        self.security: Optional[SecurityConfig] = None
        
        # Hot reloading
        self.observer: Optional[Observer] = None
        self.reload_callbacks: List[callable] = []
        
        # Load initial configuration
        self._load_config()
        
        self.logger.info(f"Initialized ConfigManager for {self.environment.value}")
    
    def _detect_environment(self) -> Environment:
        """Detect environment from environment variables."""
        env = os.getenv("OPUS_CLIP_ENV", "development").lower()
        
        if env == "production":
            return Environment.PRODUCTION
        elif env == "staging":
            return Environment.STAGING
        elif env == "testing":
            return Environment.TESTING
        else:
            return Environment.DEVELOPMENT
    
    def _load_config(self):
        """Load configuration from file and environment variables."""
        try:
            # Load from YAML file if exists
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            else:
                config_data = {}
            
            # Override with environment variables
            config_data = self._apply_env_overrides(config_data)
            
            # Load configuration sections
            self.database = self._load_database_config(config_data.get("database", {}))
            self.redis = self._load_redis_config(config_data.get("redis", {}))
            self.ai = self._load_ai_config(config_data.get("ai", {}))
            self.video = self._load_video_config(config_data.get("video", {}))
            self.performance = self._load_performance_config(config_data.get("performance", {}))
            self.security = self._load_security_config(config_data.get("security", {}))
            
            self.logger.info("Configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            # Load default configuration
            self._load_default_config()
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        env_mappings = {
            "OPUS_CLIP_DB_HOST": "database.host",
            "OPUS_CLIP_DB_PORT": "database.port",
            "OPUS_CLIP_DB_NAME": "database.database",
            "OPUS_CLIP_DB_USER": "database.username",
            "OPUS_CLIP_DB_PASSWORD": "database.password",
            "OPUS_CLIP_REDIS_HOST": "redis.host",
            "OPUS_CLIP_REDIS_PORT": "redis.port",
            "OPUS_CLIP_REDIS_PASSWORD": "redis.password",
            "OPUS_CLIP_WHISPER_MODEL": "ai.whisper_model",
            "OPUS_CLIP_DEVICE": "ai.device",
            "OPUS_CLIP_MAX_WORKERS": "performance.max_workers",
            "OPUS_CLIP_LOG_LEVEL": "performance.log_level",
            "OPUS_CLIP_SECRET_KEY": "security.secret_key",
            "OPUS_CLIP_JWT_SECRET": "security.jwt_secret"
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Set nested config value
                keys = config_path.split('.')
                current = config_data
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = self._convert_env_value(value)
        
        return config_data
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable value to appropriate type."""
        # Try to convert to number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Try to convert to boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Return as string
        return value
    
    def _load_database_config(self, config: Dict[str, Any]) -> DatabaseConfig:
        """Load database configuration."""
        return DatabaseConfig(
            host=config.get("host", "localhost"),
            port=config.get("port", 5432),
            database=config.get("database", "opus_clip"),
            username=config.get("username", "postgres"),
            password=config.get("password", ""),
            pool_size=config.get("pool_size", 10),
            max_overflow=config.get("max_overflow", 20),
            echo=config.get("echo", False)
        )
    
    def _load_redis_config(self, config: Dict[str, Any]) -> RedisConfig:
        """Load Redis configuration."""
        return RedisConfig(
            host=config.get("host", "localhost"),
            port=config.get("port", 6379),
            password=config.get("password", ""),
            db=config.get("db", 0),
            max_connections=config.get("max_connections", 100),
            socket_timeout=config.get("socket_timeout", 5)
        )
    
    def _load_ai_config(self, config: Dict[str, Any]) -> AIConfig:
        """Load AI configuration."""
        return AIConfig(
            whisper_model=config.get("whisper_model", "base"),
            sentiment_model=config.get("sentiment_model", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
            device=config.get("device", "auto"),
            batch_size=config.get("batch_size", 1),
            max_length=config.get("max_length", 512),
            temperature=config.get("temperature", 0.7)
        )
    
    def _load_video_config(self, config: Dict[str, Any]) -> VideoConfig:
        """Load video configuration."""
        return VideoConfig(
            max_duration=config.get("max_duration", 300.0),
            min_duration=config.get("min_duration", 1.0),
            max_clips=config.get("max_clips", 50),
            segment_duration=config.get("segment_duration", 5.0),
            engagement_threshold=config.get("engagement_threshold", 0.3),
            quality_settings=config.get("quality_settings", {
                "low": {"bitrate": "800k", "resolution": "480p"},
                "medium": {"bitrate": "1500k", "resolution": "720p"},
                "high": {"bitrate": "3000k", "resolution": "1080p"},
                "ultra": {"bitrate": "5000k", "resolution": "4k"}
            })
        )
    
    def _load_performance_config(self, config: Dict[str, Any]) -> PerformanceConfig:
        """Load performance configuration."""
        return PerformanceConfig(
            max_workers=config.get("max_workers", 4),
            max_memory_mb=config.get("max_memory_mb", 8192),
            enable_gpu=config.get("enable_gpu", True),
            enable_caching=config.get("enable_caching", True),
            cache_ttl_seconds=config.get("cache_ttl_seconds", 3600),
            enable_monitoring=config.get("enable_monitoring", True),
            log_level=config.get("log_level", "INFO")
        )
    
    def _load_security_config(self, config: Dict[str, Any]) -> SecurityConfig:
        """Load security configuration."""
        return SecurityConfig(
            secret_key=config.get("secret_key", ""),
            jwt_secret=config.get("jwt_secret", ""),
            jwt_expiry_hours=config.get("jwt_expiry_hours", 24),
            cors_origins=config.get("cors_origins", ["*"]),
            rate_limit_requests=config.get("rate_limit_requests", 100),
            rate_limit_window=config.get("rate_limit_window", 3600)
        )
    
    def _load_default_config(self):
        """Load default configuration."""
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.ai = AIConfig()
        self.video = VideoConfig()
        self.performance = PerformanceConfig()
        self.security = SecurityConfig()
        
        self.logger.warning("Loaded default configuration")
    
    def get_database_url(self) -> str:
        """Get database URL."""
        return (f"postgresql://{self.database.username}:{self.database.password}"
                f"@{self.database.host}:{self.database.port}/{self.database.database}")
    
    def get_redis_url(self) -> str:
        """Get Redis URL."""
        if self.redis.password:
            return f"redis://:{self.redis.password}@{self.redis.host}:{self.redis.port}/{self.redis.db}"
        else:
            return f"redis://{self.redis.host}:{self.redis.port}/{self.redis.db}"
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT
    
    def is_testing(self) -> bool:
        """Check if running in testing."""
        return self.environment == Environment.TESTING
    
    def add_reload_callback(self, callback: callable):
        """Add callback for configuration reload."""
        self.reload_callbacks.append(callback)
    
    async def reload_config(self):
        """Reload configuration and notify callbacks."""
        self.logger.info("Reloading configuration...")
        self._load_config()
        
        # Notify callbacks
        for callback in self.reload_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                self.logger.error(f"Error in reload callback: {e}")
        
        self.logger.info("Configuration reloaded successfully")
    
    def start_hot_reload(self):
        """Start hot reloading for configuration file changes."""
        if self.observer is not None:
            return
        
        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(self, config_manager):
                self.config_manager = config_manager
            
            def on_modified(self, event):
                if event.src_path == self.config_manager.config_path:
                    asyncio.create_task(self.config_manager.reload_config())
        
        self.observer = Observer()
        self.observer.schedule(
            ConfigFileHandler(self),
            path=str(Path(self.config_path).parent),
            recursive=False
        )
        self.observer.start()
        
        self.logger.info("Hot reload started")
    
    def stop_hot_reload(self):
        """Stop hot reloading."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            self.logger.info("Hot reload stopped")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for debugging."""
        return {
            "environment": self.environment.value,
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "pool_size": self.database.pool_size
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db
            },
            "ai": {
                "whisper_model": self.ai.whisper_model,
                "device": self.ai.device,
                "batch_size": self.ai.batch_size
            },
            "video": {
                "max_duration": self.video.max_duration,
                "max_clips": self.video.max_clips,
                "segment_duration": self.video.segment_duration
            },
            "performance": {
                "max_workers": self.performance.max_workers,
                "enable_gpu": self.performance.enable_gpu,
                "enable_caching": self.performance.enable_caching
            }
        }
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop_hot_reload()


