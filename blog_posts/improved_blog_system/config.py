"""
Advanced Blog Posts System Configuration
=======================================

Configuration management for blog posts system with environment support and validation.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseSettings, Field, validator, root_validator
from pydantic.env_settings import SettingsSourceCallable
import yaml
import json
from pathlib import Path


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


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="blog_posts", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="password", description="Database password")
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Max overflow connections")
    pool_timeout: int = Field(default=30, ge=1, le=300, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, ge=300, le=86400, description="Pool recycle time in seconds")
    echo: bool = Field(default=False, description="Echo SQL queries")
    
    @property
    def url(self) -> str:
        """Get database URL"""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    class Config:
        env_prefix = "DB_"


class RedisConfig(BaseSettings):
    """Redis configuration"""
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    password: Optional[str] = Field(default=None, description="Redis password")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    max_connections: int = Field(default=10, ge=1, le=100, description="Max connections")
    socket_timeout: int = Field(default=5, ge=1, le=60, description="Socket timeout in seconds")
    socket_connect_timeout: int = Field(default=5, ge=1, le=60, description="Socket connect timeout")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    decode_responses: bool = Field(default=True, description="Decode responses")
    
    class Config:
        env_prefix = "REDIS_"


class AIConfig(BaseSettings):
    """AI services configuration"""
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4", description="OpenAI model")
    openai_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="OpenAI temperature")
    openai_max_tokens: int = Field(default=2000, ge=1, le=8000, description="OpenAI max tokens")
    openai_timeout: int = Field(default=30, ge=1, le=300, description="OpenAI timeout in seconds")
    
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", description="Anthropic model")
    anthropic_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Anthropic temperature")
    anthropic_max_tokens: int = Field(default=2000, ge=1, le=8000, description="Anthropic max tokens")
    anthropic_timeout: int = Field(default=30, ge=1, le=300, description="Anthropic timeout in seconds")
    
    huggingface_api_key: Optional[str] = Field(default=None, description="Hugging Face API key")
    huggingface_model: str = Field(default="cardiffnlp/twitter-roberta-base-sentiment-latest", description="Hugging Face model")
    
    google_api_key: Optional[str] = Field(default=None, description="Google API key")
    google_model: str = Field(default="gemini-pro", description="Google model")
    
    class Config:
        env_prefix = "AI_"


class SecurityConfig(BaseSettings):
    """Security configuration"""
    secret_key: str = Field(default="your-secret-key-here", description="Secret key for JWT")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, ge=1, le=1440, description="Access token expire minutes")
    refresh_token_expire_days: int = Field(default=7, ge=1, le=30, description="Refresh token expire days")
    
    # Password hashing
    password_hash_algorithm: str = Field(default="bcrypt", description="Password hash algorithm")
    password_hash_rounds: int = Field(default=12, ge=4, le=20, description="Password hash rounds")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, ge=1, le=10000, description="Rate limit requests per minute")
    rate_limit_window: int = Field(default=60, ge=1, le=3600, description="Rate limit window in seconds")
    
    # CORS
    cors_origins: List[str] = Field(default=["*"], description="CORS origins")
    cors_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE"], description="CORS methods")
    cors_headers: List[str] = Field(default=["*"], description="CORS headers")
    
    # Security headers
    enable_security_headers: bool = Field(default=True, description="Enable security headers")
    content_security_policy: str = Field(default="default-src 'self'", description="Content Security Policy")
    
    class Config:
        env_prefix = "SECURITY_"


class MonitoringConfig(BaseSettings):
    """Monitoring configuration"""
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=8000, ge=1000, le=65535, description="Metrics port")
    metrics_path: str = Field(default="/metrics", description="Metrics path")
    
    # Health checks
    enable_health_checks: bool = Field(default=True, description="Enable health checks")
    health_check_interval: int = Field(default=30, ge=5, le=300, description="Health check interval in seconds")
    
    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    log_rotation: str = Field(default="daily", description="Log rotation")
    log_retention: int = Field(default=30, ge=1, le=365, description="Log retention in days")
    
    # Performance monitoring
    enable_performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    slow_query_threshold: float = Field(default=1.0, ge=0.1, le=10.0, description="Slow query threshold in seconds")
    
    class Config:
        env_prefix = "MONITORING_"


class BlogPostConfig(BaseSettings):
    """Blog post specific configuration"""
    max_posts_per_user: int = Field(default=1000, ge=1, le=10000, description="Max posts per user")
    max_content_length: int = Field(default=100000, ge=1000, le=1000000, description="Max content length")
    max_title_length: int = Field(default=200, ge=10, le=500, description="Max title length")
    max_excerpt_length: int = Field(default=500, ge=50, le=1000, description="Max excerpt length")
    max_tags_per_post: int = Field(default=10, ge=1, le=50, description="Max tags per post")
    max_categories_per_post: int = Field(default=5, ge=1, le=20, description="Max categories per post")
    
    # Auto-save
    enable_auto_save: bool = Field(default=True, description="Enable auto save")
    auto_save_interval: int = Field(default=30, ge=5, le=300, description="Auto save interval in seconds")
    
    # Content analysis
    enable_content_analysis: bool = Field(default=True, description="Enable content analysis")
    content_analysis_timeout: int = Field(default=30, ge=5, le=300, description="Content analysis timeout")
    
    # SEO optimization
    enable_seo_optimization: bool = Field(default=True, description="Enable SEO optimization")
    seo_optimization_timeout: int = Field(default=30, ge=5, le=300, description="SEO optimization timeout")
    
    # ML generation
    enable_ml_generation: bool = Field(default=True, description="Enable ML content generation")
    ml_generation_timeout: int = Field(default=60, ge=10, le=600, description="ML generation timeout")
    
    # Analytics
    enable_analytics: bool = Field(default=True, description="Enable analytics")
    analytics_retention_days: int = Field(default=365, ge=30, le=3650, description="Analytics retention days")
    
    # Collaboration
    enable_collaboration: bool = Field(default=True, description="Enable collaboration")
    max_collaborators_per_post: int = Field(default=10, ge=1, le=50, description="Max collaborators per post")
    
    # Workflows
    enable_workflows: bool = Field(default=True, description="Enable workflows")
    workflow_timeout: int = Field(default=300, ge=30, le=3600, description="Workflow timeout in seconds")
    
    # Templates
    enable_templates: bool = Field(default=True, description="Enable templates")
    max_templates_per_user: int = Field(default=50, ge=1, le=500, description="Max templates per user")
    
    class Config:
        env_prefix = "BLOG_"


class CacheConfig(BaseSettings):
    """Cache configuration"""
    enable_caching: bool = Field(default=True, description="Enable caching")
    default_ttl: int = Field(default=3600, ge=60, le=86400, description="Default TTL in seconds")
    
    # Cache strategies
    enable_lru_cache: bool = Field(default=True, description="Enable LRU cache")
    lru_cache_size: int = Field(default=1000, ge=100, le=10000, description="LRU cache size")
    
    enable_lfu_cache: bool = Field(default=True, description="Enable LFU cache")
    lfu_cache_size: int = Field(default=1000, ge=100, le=10000, description="LFU cache size")
    
    # Cache invalidation
    enable_intelligent_invalidation: bool = Field(default=True, description="Enable intelligent invalidation")
    invalidation_batch_size: int = Field(default=100, ge=10, le=1000, description="Invalidation batch size")
    
    class Config:
        env_prefix = "CACHE_"


class StorageConfig(BaseSettings):
    """Storage configuration"""
    # Local storage
    local_storage_path: str = Field(default="./storage", description="Local storage path")
    max_file_size: int = Field(default=10485760, ge=1048576, le=1073741824, description="Max file size in bytes")
    allowed_file_types: List[str] = Field(default=["jpg", "jpeg", "png", "gif", "webp", "mp4", "mov", "avi"], description="Allowed file types")
    
    # S3 storage
    s3_enabled: bool = Field(default=False, description="Enable S3 storage")
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket name")
    s3_region: Optional[str] = Field(default=None, description="S3 region")
    s3_access_key: Optional[str] = Field(default=None, description="S3 access key")
    s3_secret_key: Optional[str] = Field(default=None, description="S3 secret key")
    s3_endpoint_url: Optional[str] = Field(default=None, description="S3 endpoint URL")
    
    # Google Cloud Storage
    gcs_enabled: bool = Field(default=False, description="Enable Google Cloud Storage")
    gcs_bucket: Optional[str] = Field(default=None, description="GCS bucket name")
    gcs_project: Optional[str] = Field(default=None, description="GCS project ID")
    gcs_credentials_path: Optional[str] = Field(default=None, description="GCS credentials path")
    
    # Azure Blob Storage
    azure_enabled: bool = Field(default=False, description="Enable Azure Blob Storage")
    azure_account_name: Optional[str] = Field(default=None, description="Azure account name")
    azure_account_key: Optional[str] = Field(default=None, description="Azure account key")
    azure_container: Optional[str] = Field(default=None, description="Azure container name")
    
    class Config:
        env_prefix = "STORAGE_"


class NotificationConfig(BaseSettings):
    """Notification configuration"""
    enable_notifications: bool = Field(default=True, description="Enable notifications")
    
    # Email notifications
    email_enabled: bool = Field(default=True, description="Enable email notifications")
    smtp_host: Optional[str] = Field(default=None, description="SMTP host")
    smtp_port: int = Field(default=587, ge=1, le=65535, description="SMTP port")
    smtp_username: Optional[str] = Field(default=None, description="SMTP username")
    smtp_password: Optional[str] = Field(default=None, description="SMTP password")
    smtp_use_tls: bool = Field(default=True, description="Use TLS for SMTP")
    
    # Slack notifications
    slack_enabled: bool = Field(default=False, description="Enable Slack notifications")
    slack_webhook_url: Optional[str] = Field(default=None, description="Slack webhook URL")
    slack_channel: Optional[str] = Field(default=None, description="Slack channel")
    
    # Discord notifications
    discord_enabled: bool = Field(default=False, description="Enable Discord notifications")
    discord_webhook_url: Optional[str] = Field(default=None, description="Discord webhook URL")
    
    class Config:
        env_prefix = "NOTIFICATION_"


class BlogPostSystemConfig(BaseSettings):
    """Main blog post system configuration"""
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Application
    app_name: str = Field(default="Blog Posts System", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    app_description: str = Field(default="Advanced blog posts system with AI and ML", description="Application description")
    
    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1000, le=65535, description="Server port")
    workers: int = Field(default=1, ge=1, le=32, description="Number of workers")
    
    # API
    api_prefix: str = Field(default="/api/v1", description="API prefix")
    api_title: str = Field(default="Blog Posts API", description="API title")
    api_description: str = Field(default="Advanced blog posts API with AI and ML", description="API description")
    api_version: str = Field(default="1.0.0", description="API version")
    
    # Configuration components
    database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Database configuration")
    redis: RedisConfig = Field(default_factory=RedisConfig, description="Redis configuration")
    ai: AIConfig = Field(default_factory=AIConfig, description="AI configuration")
    security: SecurityConfig = Field(default_factory=SecurityConfig, description="Security configuration")
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig, description="Monitoring configuration")
    blog_post: BlogPostConfig = Field(default_factory=BlogPostConfig, description="Blog post configuration")
    cache: CacheConfig = Field(default_factory=CacheConfig, description="Cache configuration")
    storage: StorageConfig = Field(default_factory=StorageConfig, description="Storage configuration")
    notification: NotificationConfig = Field(default_factory=NotificationConfig, description="Notification configuration")
    
    # Feature flags
    enable_ml_pipeline: bool = Field(default=True, description="Enable ML pipeline")
    enable_content_analysis: bool = Field(default=True, description="Enable content analysis")
    enable_content_generation: bool = Field(default=True, description="Enable content generation")
    enable_seo_optimization: bool = Field(default=True, description="Enable SEO optimization")
    enable_analytics: bool = Field(default=True, description="Enable analytics")
    enable_collaboration: bool = Field(default=True, description="Enable collaboration")
    enable_workflows: bool = Field(default=True, description="Enable workflows")
    enable_templates: bool = Field(default=True, description="Enable templates")
    enable_webhooks: bool = Field(default=True, description="Enable webhooks")
    enable_automation: bool = Field(default=True, description="Enable automation")
    
    # Performance
    enable_compression: bool = Field(default=True, description="Enable response compression")
    enable_cors: bool = Field(default=True, description="Enable CORS")
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    enable_caching: bool = Field(default=True, description="Enable caching")
    
    # Development
    enable_hot_reload: bool = Field(default=False, description="Enable hot reload")
    enable_debug_toolbar: bool = Field(default=False, description="Enable debug toolbar")
    enable_profiling: bool = Field(default=False, description="Enable profiling")
    
    class Config:
        env_prefix = "BLOG_SYSTEM_"
        case_sensitive = False
        
        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> tuple[SettingsSourceCallable, ...]:
            return (
                init_settings,
                env_settings,
                file_secret_settings,
                cls._yaml_config_settings_source,
                cls._json_config_settings_source,
            )
    
    @classmethod
    def _yaml_config_settings_source(cls, settings: BaseSettings) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_file = Path("config.yaml")
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logging.warning(f"Failed to load YAML config: {e}")
        return {}
    
    @classmethod
    def _json_config_settings_source(cls, settings: BaseSettings) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        config_file = Path("config.json")
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    return json.load(f) or {}
            except Exception as e:
                logging.warning(f"Failed to load JSON config: {e}")
        return {}
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment"""
        if v not in [env.value for env in Environment]:
            raise ValueError(f"Invalid environment: {v}")
        return v
    
    @validator('debug')
    def validate_debug(cls, v, values):
        """Validate debug mode"""
        if values.get('environment') == Environment.PRODUCTION and v:
            raise ValueError("Debug mode cannot be enabled in production")
        return v
    
    @root_validator
    def validate_configuration(cls, values):
        """Validate overall configuration"""
        environment = values.get('environment')
        
        # Production-specific validations
        if environment == Environment.PRODUCTION:
            if not values.get('security', {}).get('secret_key') or values.get('security', {}).get('secret_key') == "your-secret-key-here":
                raise ValueError("Secret key must be set in production")
            
            if values.get('debug'):
                raise ValueError("Debug mode must be disabled in production")
            
            if not values.get('database', {}).get('password') or values.get('database', {}).get('password') == "password":
                raise ValueError("Database password must be set in production")
        
        return values
    
    def get_database_url(self) -> str:
        """Get database URL"""
        return self.database.url
    
    def get_redis_url(self) -> str:
        """Get Redis URL"""
        if self.redis.password:
            return f"redis://:{self.redis.password}@{self.redis.host}:{self.redis.port}/{self.redis.db}"
        return f"redis://{self.redis.host}:{self.redis.port}/{self.redis.db}"
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT
    
    def is_testing(self) -> bool:
        """Check if running in testing"""
        return self.environment == Environment.TESTING
    
    def get_log_level(self) -> str:
        """Get log level"""
        if self.is_production():
            return "INFO"
        elif self.is_development():
            return "DEBUG"
        else:
            return self.monitoring.log_level.value
    
    def export_config(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return self.dict()
    
    def export_config_to_file(self, file_path: str, format: str = "yaml") -> None:
        """Export configuration to file"""
        config_data = self.export_config()
        
        if format.lower() == "yaml":
            with open(file_path, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False)
        elif format.lower() == "json":
            with open(file_path, "w") as f:
                json.dump(config_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def validate_ai_configuration(self) -> List[str]:
        """Validate AI configuration and return warnings"""
        warnings = []
        
        if not self.ai.openai_api_key and not self.ai.anthropic_api_key:
            warnings.append("No AI API keys configured. Content generation will be disabled.")
        
        if self.ai.openai_api_key and not self.ai.openai_api_key.startswith("sk-"):
            warnings.append("OpenAI API key format appears to be invalid.")
        
        if self.ai.anthropic_api_key and not self.ai.anthropic_api_key.startswith("sk-ant-"):
            warnings.append("Anthropic API key format appears to be invalid.")
        
        return warnings
    
    def validate_storage_configuration(self) -> List[str]:
        """Validate storage configuration and return warnings"""
        warnings = []
        
        if not any([
            self.storage.s3_enabled,
            self.storage.gcs_enabled,
            self.storage.azure_enabled
        ]):
            warnings.append("No cloud storage configured. Only local storage will be available.")
        
        if self.storage.s3_enabled and not all([
            self.storage.s3_bucket,
            self.storage.s3_access_key,
            self.storage.s3_secret_key
        ]):
            warnings.append("S3 storage enabled but configuration incomplete.")
        
        return warnings
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get feature flags"""
        return {
            "ml_pipeline": self.enable_ml_pipeline,
            "content_analysis": self.enable_content_analysis,
            "content_generation": self.enable_content_generation,
            "seo_optimization": self.enable_seo_optimization,
            "analytics": self.enable_analytics,
            "collaboration": self.enable_collaboration,
            "workflows": self.enable_workflows,
            "templates": self.enable_templates,
            "webhooks": self.enable_webhooks,
            "automation": self.enable_automation
        }


# Global configuration instance
_config: Optional[BlogPostSystemConfig] = None


def get_settings() -> BlogPostSystemConfig:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = BlogPostSystemConfig()
    return _config


def reload_settings() -> BlogPostSystemConfig:
    """Reload configuration from environment and files"""
    global _config
    _config = BlogPostSystemConfig()
    return _config


def validate_settings() -> Dict[str, Any]:
    """Validate current settings and return validation results"""
    settings = get_settings()
    
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "feature_flags": settings.get_feature_flags()
    }
    
    # Validate AI configuration
    ai_warnings = settings.validate_ai_configuration()
    validation_results["warnings"].extend(ai_warnings)
    
    # Validate storage configuration
    storage_warnings = settings.validate_storage_configuration()
    validation_results["warnings"].extend(storage_warnings)
    
    # Check for critical errors
    if settings.is_production():
        if not settings.security.secret_key or settings.security.secret_key == "your-secret-key-here":
            validation_results["errors"].append("Secret key must be set in production")
            validation_results["valid"] = False
        
        if settings.debug:
            validation_results["errors"].append("Debug mode must be disabled in production")
            validation_results["valid"] = False
    
    return validation_results


def export_settings(file_path: str, format: str = "yaml") -> None:
    """Export current settings to file"""
    settings = get_settings()
    settings.export_config_to_file(file_path, format)


def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return get_settings().database


def get_redis_config() -> RedisConfig:
    """Get Redis configuration"""
    return get_settings().redis


def get_ai_config() -> AIConfig:
    """Get AI configuration"""
    return get_settings().ai


def get_security_config() -> SecurityConfig:
    """Get security configuration"""
    return get_settings().security


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration"""
    return get_settings().monitoring


def get_blog_post_config() -> BlogPostConfig:
    """Get blog post configuration"""
    return get_settings().blog_post


def get_cache_config() -> CacheConfig:
    """Get cache configuration"""
    return get_settings().cache


def get_storage_config() -> StorageConfig:
    """Get storage configuration"""
    return get_settings().storage


def get_notification_config() -> NotificationConfig:
    """Get notification configuration"""
    return get_settings().notification





























