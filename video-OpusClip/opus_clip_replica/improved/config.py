"""
Advanced Configuration Management for OpusClip Improved
=====================================================

Comprehensive configuration system with environment support, validation, and hot reloading.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union, Type
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml
import json
from pydantic import BaseSettings, Field, validator, root_validator
from pydantic.env_settings import SettingsSourceCallable

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseType(str, Enum):
    """Database types"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"


class CacheType(str, Enum):
    """Cache types"""
    REDIS = "redis"
    MEMORY = "memory"
    FILE = "file"


class StorageType(str, Enum):
    """Storage types"""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: DatabaseType = DatabaseType.POSTGRESQL
    host: str = "localhost"
    port: int = 5432
    database: str = "opusclip"
    username: str = "opusclip"
    password: str = "opusclip"
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    ssl_mode: str = "prefer"


@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 20
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30


@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_special: bool = True
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    enable_2fa: bool = False
    session_timeout_minutes: int = 60


@dataclass
class AIConfig:
    """AI configuration"""
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    openai_max_tokens: int = 4000
    openai_temperature: float = 0.7
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-sonnet-20240229"
    anthropic_max_tokens: int = 4000
    google_api_key: Optional[str] = None
    google_model: str = "gemini-pro"
    huggingface_api_key: Optional[str] = None
    huggingface_model: str = "microsoft/DialoGPT-medium"
    default_provider: str = "openai"
    fallback_providers: List[str] = field(default_factory=lambda: ["anthropic", "google"])


@dataclass
class VideoProcessingConfig:
    """Video processing configuration"""
    max_file_size_mb: int = 100
    allowed_formats: List[str] = field(default_factory=lambda: ["mp4", "avi", "mov", "mkv", "webm"])
    max_duration_seconds: int = 3600
    default_quality: str = "high"
    gpu_acceleration: bool = True
    cuda_device: int = 0
    max_concurrent_processes: int = 4
    temp_directory: str = "/tmp/opusclip"
    output_directory: str = "/app/outputs"
    watermark_enabled: bool = False
    watermark_path: Optional[str] = None
    subtitle_enabled: bool = True
    subtitle_language: str = "en"


@dataclass
class StorageConfig:
    """Storage configuration"""
    type: StorageType = StorageType.LOCAL
    local_path: str = "/app/storage"
    s3_bucket: Optional[str] = None
    s3_region: str = "us-east-1"
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    gcs_bucket: Optional[str] = None
    gcs_credentials_path: Optional[str] = None
    azure_account_name: Optional[str] = None
    azure_account_key: Optional[str] = None
    azure_container: Optional[str] = None
    cdn_url: Optional[str] = None
    signed_url_expiry: int = 3600


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    grafana_enabled: bool = True
    grafana_port: int = 3000
    health_check_interval: int = 30
    metrics_retention_days: int = 30
    alert_manager_enabled: bool = True
    alert_webhook_url: Optional[str] = None
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    log_file: Optional[str] = None
    log_rotation: bool = True
    log_max_size_mb: int = 100
    log_backup_count: int = 5


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    enabled: bool = True
    default_limit: int = 100
    default_window: int = 60
    burst_limit: int = 200
    storage_backend: str = "redis"
    key_prefix: str = "rate_limit"
    skip_successful_requests: bool = False
    skip_failed_requests: bool = False


@dataclass
class WebhookConfig:
    """Webhook configuration"""
    enabled: bool = True
    max_retries: int = 3
    retry_delay: int = 60
    timeout: int = 30
    signature_header: str = "X-Webhook-Signature"
    secret_key: str = "webhook-secret-key"
    queue_size: int = 1000
    worker_count: int = 5


class Settings(BaseSettings):
    """Main application settings"""
    
    # Application
    app_name: str = "OpusClip Improved"
    app_version: str = "2.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # API
    api_prefix: str = "/api/v2/opus-clip"
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    cors_methods: List[str] = Field(default_factory=lambda: ["*"])
    cors_headers: List[str] = Field(default_factory=lambda: ["*"])
    
    # Database
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    # Redis
    redis: RedisConfig = Field(default_factory=RedisConfig)
    
    # Security
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # AI
    ai: AIConfig = Field(default_factory=AIConfig)
    
    # Video Processing
    video_processing: VideoProcessingConfig = Field(default_factory=VideoProcessingConfig)
    
    # Storage
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    # Monitoring
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # Rate Limiting
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    
    # Webhooks
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True
        
        # Environment variable mapping
        fields = {
            "database.host": {"env": "DB_HOST"},
            "database.port": {"env": "DB_PORT"},
            "database.database": {"env": "DB_NAME"},
            "database.username": {"env": "DB_USER"},
            "database.password": {"env": "DB_PASSWORD"},
            "redis.host": {"env": "REDIS_HOST"},
            "redis.port": {"env": "REDIS_PORT"},
            "redis.password": {"env": "REDIS_PASSWORD"},
            "security.secret_key": {"env": "SECRET_KEY"},
            "ai.openai_api_key": {"env": "OPENAI_API_KEY"},
            "ai.anthropic_api_key": {"env": "ANTHROPIC_API_KEY"},
            "ai.google_api_key": {"env": "GOOGLE_API_KEY"},
            "ai.huggingface_api_key": {"env": "HUGGINGFACE_API_KEY"},
            "storage.s3_bucket": {"env": "S3_BUCKET"},
            "storage.s3_access_key": {"env": "S3_ACCESS_KEY"},
            "storage.s3_secret_key": {"env": "S3_SECRET_KEY"},
        }
    
    @validator("environment", pre=True)
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator("debug", pre=True)
    def validate_debug(cls, v, values):
        if "environment" in values and values["environment"] == Environment.PRODUCTION:
            return False
        return v
    
    @validator("cors_origins", pre=True)
    def validate_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @root_validator
    def validate_configuration(cls, values):
        """Validate overall configuration"""
        environment = values.get("environment", Environment.DEVELOPMENT)
        
        # Production-specific validations
        if environment == Environment.PRODUCTION:
            if not values.get("security", {}).get("secret_key") or values["security"]["secret_key"] == "your-secret-key-here":
                raise ValueError("Secret key must be set in production")
            
            if values.get("debug", False):
                raise ValueError("Debug mode must be disabled in production")
            
            if "*" in values.get("cors_origins", []):
                raise ValueError("CORS origins must be specific in production")
        
        # Validate AI configuration
        ai_config = values.get("ai", {})
        if not any([
            ai_config.get("openai_api_key"),
            ai_config.get("anthropic_api_key"),
            ai_config.get("google_api_key"),
            ai_config.get("huggingface_api_key")
        ]):
            logger.warning("No AI API keys configured")
        
        # Validate storage configuration
        storage_config = values.get("storage", {})
        storage_type = storage_config.get("type", StorageType.LOCAL)
        
        if storage_type == StorageType.S3:
            if not all([storage_config.get("s3_bucket"), storage_config.get("s3_access_key"), storage_config.get("s3_secret_key")]):
                raise ValueError("S3 configuration incomplete")
        
        elif storage_type == StorageType.GCS:
            if not storage_config.get("gcs_bucket"):
                raise ValueError("GCS bucket not configured")
        
        elif storage_type == StorageType.AZURE:
            if not all([storage_config.get("azure_account_name"), storage_config.get("azure_account_key"), storage_config.get("azure_container")]):
                raise ValueError("Azure configuration incomplete")
        
        return values
    
    def get_database_url(self) -> str:
        """Get database URL"""
        db = self.database
        return f"{db.type.value}://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}"
    
    def get_redis_url(self) -> str:
        """Get Redis URL"""
        redis = self.redis
        if redis.password:
            return f"redis://:{redis.password}@{redis.host}:{redis.port}/{redis.db}"
        return f"redis://{redis.host}:{redis.port}/{redis.db}"
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT
    
    def is_testing(self) -> bool:
        """Check if running in testing"""
        return self.environment == Environment.TESTING
    
    def get_log_level(self) -> int:
        """Get log level as integer"""
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL
        }
        return level_map.get(self.monitoring.log_level, logging.INFO)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return self.dict()
    
    def save_to_file(self, file_path: str, format: str = "yaml"):
        """Save settings to file"""
        try:
            data = self.to_dict()
            
            if format.lower() == "yaml":
                with open(file_path, "w") as f:
                    yaml.dump(data, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Settings saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            raise
    
    @classmethod
    def load_from_file(cls, file_path: str) -> "Settings":
        """Load settings from file"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"Settings file not found: {file_path}")
            
            if path.suffix.lower() == ".yaml" or path.suffix.lower() == ".yml":
                with open(file_path, "r") as f:
                    data = yaml.safe_load(f)
            elif path.suffix.lower() == ".json":
                with open(file_path, "r") as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
            
            return cls(**data)
            
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            raise


class ConfigurationManager:
    """Configuration management with hot reloading"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.settings: Optional[Settings] = None
        self.last_modified: Optional[datetime] = None
        self._load_settings()
    
    def _load_settings(self):
        """Load settings from file or environment"""
        try:
            if self.config_file and Path(self.config_file).exists():
                self.settings = Settings.load_from_file(self.config_file)
                self.last_modified = datetime.fromtimestamp(Path(self.config_file).stat().st_mtime)
            else:
                self.settings = Settings()
                self.last_modified = datetime.utcnow()
            
            logger.info(f"Configuration loaded for environment: {self.settings.environment}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Fallback to default settings
            self.settings = Settings()
    
    def get_settings(self) -> Settings:
        """Get current settings"""
        if self.config_file and Path(self.config_file).exists():
            # Check if file was modified
            current_mtime = datetime.fromtimestamp(Path(self.config_file).stat().st_mtime)
            if self.last_modified and current_mtime > self.last_modified:
                logger.info("Configuration file modified, reloading...")
                self._load_settings()
        
        return self.settings
    
    def reload_settings(self):
        """Manually reload settings"""
        self._load_settings()
        logger.info("Settings reloaded")
    
    def update_setting(self, key: str, value: Any):
        """Update a setting value"""
        try:
            # Parse nested key (e.g., "database.host")
            keys = key.split(".")
            current = self.settings.dict()
            
            # Navigate to the nested key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set the value
            current[keys[-1]] = value
            
            # Create new settings instance
            self.settings = Settings(**self.settings.dict())
            
            logger.info(f"Setting updated: {key} = {value}")
            
        except Exception as e:
            logger.error(f"Failed to update setting: {e}")
            raise
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value"""
        try:
            # Parse nested key
            keys = key.split(".")
            current = self.settings.dict()
            
            # Navigate to the nested key
            for k in keys:
                if k not in current:
                    return default
                current = current[k]
            
            return current
            
        except Exception as e:
            logger.error(f"Failed to get setting: {e}")
            return default
    
    def export_settings(self, file_path: str, format: str = "yaml", include_secrets: bool = False):
        """Export settings to file"""
        try:
            settings_dict = self.settings.to_dict()
            
            # Remove secrets if requested
            if not include_secrets:
                self._remove_secrets(settings_dict)
            
            if format.lower() == "yaml":
                with open(file_path, "w") as f:
                    yaml.dump(settings_dict, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                with open(file_path, "w") as f:
                    json.dump(settings_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Settings exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export settings: {e}")
            raise
    
    def _remove_secrets(self, settings_dict: Dict[str, Any]):
        """Remove secret values from settings dictionary"""
        secret_keys = [
            "secret_key", "password", "api_key", "access_key", "secret_key",
            "token", "credentials", "private_key"
        ]
        
        def remove_secrets_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if any(secret in key.lower() for secret in secret_keys):
                        obj[key] = "***REDACTED***"
                    else:
                        remove_secrets_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    remove_secrets_recursive(item)
        
        remove_secrets_recursive(settings_dict)
    
    def validate_settings(self) -> Dict[str, Any]:
        """Validate current settings"""
        try:
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            settings = self.settings
            
            # Check required fields
            if not settings.security.secret_key or settings.security.secret_key == "your-secret-key-here":
                validation_result["errors"].append("Secret key not configured")
                validation_result["valid"] = False
            
            # Check AI configuration
            ai_configured = any([
                settings.ai.openai_api_key,
                settings.ai.anthropic_api_key,
                settings.ai.google_api_key,
                settings.ai.huggingface_api_key
            ])
            
            if not ai_configured:
                validation_result["warnings"].append("No AI providers configured")
            
            # Check storage configuration
            if settings.storage.type == StorageType.S3:
                if not all([settings.storage.s3_bucket, settings.storage.s3_access_key, settings.storage.s3_secret_key]):
                    validation_result["errors"].append("S3 storage configuration incomplete")
                    validation_result["valid"] = False
            
            # Check database configuration
            if not settings.database.host:
                validation_result["errors"].append("Database host not configured")
                validation_result["valid"] = False
            
            # Check Redis configuration
            if not settings.redis.host:
                validation_result["errors"].append("Redis host not configured")
                validation_result["valid"] = False
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Settings validation failed: {e}")
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": [],
                "timestamp": datetime.utcnow().isoformat()
            }


# Global configuration manager
config_manager = ConfigurationManager()

def get_settings() -> Settings:
    """Get current application settings"""
    return config_manager.get_settings()

def reload_settings():
    """Reload application settings"""
    config_manager.reload_settings()

def update_setting(key: str, value: Any):
    """Update a setting value"""
    config_manager.update_setting(key, value)

def get_setting(key: str, default: Any = None) -> Any:
    """Get a setting value"""
    return config_manager.get_setting(key, default)





























