"""
Gamma App - Final Configuration
Complete system configuration and initialization
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    ssl_mode: str = "prefer"

@dataclass
class RedisConfig:
    """Redis configuration"""
    url: str
    max_connections: int = 20
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    decode_responses: bool = True

@dataclass
class AIConfig:
    """AI configuration"""
    openai_api_key: str
    anthropic_api_key: str
    default_model: str = "gpt-4"
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 30
    retry_attempts: int = 3

@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_special_chars: bool = True

@dataclass
class CacheConfig:
    """Cache configuration"""
    local_enabled: bool = True
    local_max_size: str = "256MB"
    local_ttl: int = 3600
    redis_enabled: bool = True
    redis_ttl: int = 7200
    redis_key_prefix: str = "gamma:"
    cdn_enabled: bool = True
    cdn_ttl: int = 86400

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    prometheus_path: str = "/metrics"
    logging_level: str = "INFO"
    logging_format: str = "json"
    logging_file: str = "/app/logs/gamma.log"
    logging_max_size: str = "100MB"
    logging_backup_count: int = 5
    health_checks_enabled: bool = True
    health_checks_interval: int = 30
    health_checks_timeout: int = 10

@dataclass
class ExportConfig:
    """Export configuration"""
    max_file_size: str = "100MB"
    supported_formats: list = None
    default_quality: str = "high"
    quality_options: list = None
    templates_enabled: bool = True
    templates_directory: str = "/app/templates"

@dataclass
class CollaborationConfig:
    """Collaboration configuration"""
    websockets_enabled: bool = True
    websockets_max_connections: int = 1000
    websockets_ping_interval: int = 30
    websockets_ping_timeout: int = 10
    real_time_enabled: bool = True
    real_time_sync_interval: int = 100
    real_time_conflict_resolution: str = "last_write_wins"

@dataclass
class PerformanceConfig:
    """Performance configuration"""
    profiling_enabled: bool = False
    profiling_sample_rate: float = 0.1
    optimization_enabled: bool = True
    auto_gc_enabled: bool = True
    gc_threshold: float = 0.8
    max_content_size: str = "50MB"
    max_concurrent_requests: int = 100
    request_timeout: int = 300

@dataclass
class EmailConfig:
    """Email configuration"""
    smtp_host: str
    smtp_port: int = 587
    smtp_username: str
    smtp_password: str
    smtp_use_tls: bool = True
    templates_directory: str = "/app/email_templates"
    notifications_enabled: bool = True
    from_email: str = "noreply@gamma.app"

@dataclass
class StorageConfig:
    """Storage configuration"""
    local_enabled: bool = True
    local_upload_dir: str = "/app/uploads"
    local_max_file_size: str = "100MB"
    s3_enabled: bool = False
    s3_bucket: str = ""
    s3_region: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""

@dataclass
class FeatureFlags:
    """Feature flags"""
    ai_generation: bool = True
    real_time_collaboration: bool = True
    advanced_export: bool = True
    analytics: bool = True
    user_management: bool = True
    content_templates: bool = True
    multimedia_support: bool = True
    custom_themes: bool = True
    api_access: bool = True
    webhooks: bool = True

@dataclass
class GammaAppConfig:
    """Complete Gamma App configuration"""
    environment: Environment
    app_name: str = "Gamma App"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    
    # Sub-configurations
    database: DatabaseConfig
    redis: RedisConfig
    ai: AIConfig
    security: SecurityConfig
    cache: CacheConfig
    monitoring: MonitoringConfig
    export: ExportConfig
    collaboration: CollaborationConfig
    performance: PerformanceConfig
    email: EmailConfig
    storage: StorageConfig
    features: FeatureFlags

class ConfigManager:
    """Configuration manager"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config: Optional[GammaAppConfig] = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from files and environment"""
        try:
            # Determine environment
            env = Environment(os.getenv("ENVIRONMENT", "development"))
            
            # Load base configuration
            base_config = self._load_yaml_config("config.yaml")
            
            # Load environment-specific configuration
            env_config = self._load_yaml_config(f"{env.value}.yaml")
            
            # Load environment variables
            env_vars = self._load_environment_variables()
            
            # Merge configurations
            config = self._merge_configs(base_config, env_config, env_vars)
            
            # Create configuration objects
            self.config = self._create_config_objects(config, env)
            
            logger.info(f"Configuration loaded for environment: {env.value}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _load_yaml_config(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        config_file = self.config_dir / filename
        
        if not config_file.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return {}
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading YAML config {filename}: {e}")
            return {}
    
    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_vars = {}
        
        # Database
        if os.getenv("DATABASE_URL"):
            env_vars["database"] = {"url": os.getenv("DATABASE_URL")}
        
        # Redis
        if os.getenv("REDIS_URL"):
            env_vars["redis"] = {"url": os.getenv("REDIS_URL")}
        
        # AI
        if os.getenv("OPENAI_API_KEY"):
            env_vars["ai"] = {"openai_api_key": os.getenv("OPENAI_API_KEY")}
        if os.getenv("ANTHROPIC_API_KEY"):
            env_vars.setdefault("ai", {})["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")
        
        # Security
        if os.getenv("SECRET_KEY"):
            env_vars["security"] = {"secret_key": os.getenv("SECRET_KEY")}
        
        # Email
        if os.getenv("SMTP_HOST"):
            env_vars["email"] = {
                "smtp_host": os.getenv("SMTP_HOST"),
                "smtp_username": os.getenv("SMTP_USERNAME", ""),
                "smtp_password": os.getenv("SMTP_PASSWORD", "")
            }
        
        # Storage
        if os.getenv("S3_BUCKET"):
            env_vars["storage"] = {
                "s3_enabled": True,
                "s3_bucket": os.getenv("S3_BUCKET"),
                "s3_region": os.getenv("S3_REGION", ""),
                "s3_access_key": os.getenv("S3_ACCESS_KEY", ""),
                "s3_secret_key": os.getenv("S3_SECRET_KEY", "")
            }
        
        return env_vars
    
    def _merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries"""
        merged = {}
        
        for config in configs:
            for key, value in config.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = self._merge_configs(merged[key], value)
                else:
                    merged[key] = value
        
        return merged
    
    def _create_config_objects(self, config: Dict[str, Any], env: Environment) -> GammaAppConfig:
        """Create configuration objects from dictionary"""
        # Set default values
        defaults = {
            "app_name": "Gamma App",
            "app_version": "1.0.0",
            "debug": env == Environment.DEVELOPMENT,
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 4,
            "reload": env == Environment.DEVELOPMENT,
        }
        
        # Merge with config
        app_config = {**defaults, **config}
        
        # Create sub-configurations
        database_config = DatabaseConfig(**app_config.get("database", {}))
        redis_config = RedisConfig(**app_config.get("redis", {}))
        ai_config = AIConfig(**app_config.get("ai", {}))
        security_config = SecurityConfig(**app_config.get("security", {}))
        cache_config = CacheConfig(**app_config.get("cache", {}))
        monitoring_config = MonitoringConfig(**app_config.get("monitoring", {}))
        export_config = ExportConfig(**app_config.get("export", {}))
        collaboration_config = CollaborationConfig(**app_config.get("collaboration", {}))
        performance_config = PerformanceConfig(**app_config.get("performance", {}))
        email_config = EmailConfig(**app_config.get("email", {}))
        storage_config = StorageConfig(**app_config.get("storage", {}))
        features_config = FeatureFlags(**app_config.get("features", {}))
        
        return GammaAppConfig(
            environment=env,
            database=database_config,
            redis=redis_config,
            ai=ai_config,
            security=security_config,
            cache=cache_config,
            monitoring=monitoring_config,
            export=export_config,
            collaboration=collaboration_config,
            performance=performance_config,
            email=email_config,
            storage=storage_config,
            features=features_config,
            **{k: v for k, v in app_config.items() if k not in [
                "database", "redis", "ai", "security", "cache", "monitoring",
                "export", "collaboration", "performance", "email", "storage", "features"
            ]}
        )
    
    def get_config(self) -> GammaAppConfig:
        """Get current configuration"""
        if self.config is None:
            raise RuntimeError("Configuration not loaded")
        return self.config
    
    def reload_config(self):
        """Reload configuration"""
        self._load_config()
    
    def save_config(self, config: GammaAppConfig, filename: str = "config.yaml"):
        """Save configuration to file"""
        try:
            config_file = self.config_dir / filename
            
            # Convert to dictionary
            config_dict = asdict(config)
            
            # Remove environment enum
            config_dict["environment"] = config.environment.value
            
            # Save to YAML
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def validate_config(self, config: GammaAppConfig) -> Dict[str, Any]:
        """Validate configuration"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Validate required fields
            if not config.database.url:
                validation_results["errors"].append("Database URL is required")
            
            if not config.redis.url:
                validation_results["errors"].append("Redis URL is required")
            
            if not config.ai.openai_api_key and not config.ai.anthropic_api_key:
                validation_results["warnings"].append("No AI API keys configured")
            
            if not config.security.secret_key:
                validation_results["errors"].append("Secret key is required")
            
            # Validate URLs
            if config.database.url and not config.database.url.startswith(("postgresql://", "sqlite://")):
                validation_results["errors"].append("Invalid database URL format")
            
            if config.redis.url and not config.redis.url.startswith("redis://"):
                validation_results["errors"].append("Invalid Redis URL format")
            
            # Validate ports
            if not (1 <= config.port <= 65535):
                validation_results["errors"].append("Invalid port number")
            
            if not (1 <= config.monitoring.prometheus_port <= 65535):
                validation_results["errors"].append("Invalid Prometheus port number")
            
            # Check for production security
            if config.environment == Environment.PRODUCTION:
                if config.debug:
                    validation_results["warnings"].append("Debug mode enabled in production")
                
                if config.security.secret_key == "dev-secret-key-change-in-production":
                    validation_results["errors"].append("Default secret key used in production")
                
                if not config.email.smtp_host:
                    validation_results["warnings"].append("Email not configured in production")
            
            # Set overall validity
            validation_results["valid"] = len(validation_results["errors"]) == 0
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation error: {e}")
        
        return validation_results
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        if self.config is None:
            return {"error": "Configuration not loaded"}
        
        return {
            "environment": self.config.environment.value,
            "app_name": self.config.app_name,
            "app_version": self.config.app_version,
            "debug": self.config.debug,
            "host": self.config.host,
            "port": self.config.port,
            "workers": self.config.workers,
            "features_enabled": {
                "ai_generation": self.config.features.ai_generation,
                "real_time_collaboration": self.config.features.real_time_collaboration,
                "advanced_export": self.config.features.advanced_export,
                "analytics": self.config.features.analytics,
                "user_management": self.config.features.user_management,
                "content_templates": self.config.features.content_templates,
                "multimedia_support": self.config.features.multimedia_support,
                "custom_themes": self.config.features.custom_themes,
                "api_access": self.config.features.api_access,
                "webhooks": self.config.features.webhooks
            },
            "storage": {
                "local_enabled": self.config.storage.local_enabled,
                "s3_enabled": self.config.storage.s3_enabled
            },
            "monitoring": {
                "prometheus_enabled": self.config.monitoring.prometheus_enabled,
                "logging_level": self.config.monitoring.logging_level
            }
        }

# Global configuration instance
config_manager = ConfigManager()
config = config_manager.get_config()

def get_config() -> GammaAppConfig:
    """Get current configuration"""
    return config_manager.get_config()

def reload_config():
    """Reload configuration"""
    config_manager.reload_config()
    global config
    config = config_manager.get_config()

def validate_config() -> Dict[str, Any]:
    """Validate current configuration"""
    return config_manager.validate_config(config)

def get_config_summary() -> Dict[str, Any]:
    """Get configuration summary"""
    return config_manager.get_config_summary()

























