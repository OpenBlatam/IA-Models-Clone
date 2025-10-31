"""
Gamma App - Advanced Configuration
Comprehensive configuration management with environment-specific settings
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import yaml
import toml

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    echo_pool: bool = False

@dataclass
class RedisConfig:
    """Redis configuration"""
    url: str
    max_connections: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30

@dataclass
class AIConfig:
    """AI configuration"""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    default_model: str = "gpt-3.5-turbo"
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 30
    retry_attempts: int = 3
    enable_local_models: bool = True
    local_models_dir: str = "models"
    quantization: bool = False

@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_min_length: int = 8
    require_strong_password: bool = True
    enable_2fa: bool = False
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600
    max_login_attempts: int = 5
    lockout_duration: int = 900
    allowed_ips: Optional[List[str]] = None
    blocked_ips: Optional[List[str]] = None
    enable_csrf: bool = True
    enable_cors: bool = True
    cors_origins: List[str] = None

@dataclass
class CacheConfig:
    """Cache configuration"""
    default_ttl: int = 3600
    max_memory: str = "100mb"
    compression: bool = True
    serialization: str = "json"
    key_prefix: str = "gamma_app"
    enable_local_cache: bool = True
    local_cache_size: int = 1000

@dataclass
class PerformanceConfig:
    """Performance configuration"""
    enable_memory_profiling: bool = True
    enable_cpu_profiling: bool = True
    enable_request_tracking: bool = True
    enable_database_monitoring: bool = True
    enable_cache_monitoring: bool = True
    sampling_rate: float = 1.0
    alert_thresholds: Dict[str, float] = None
    retention_days: int = 7
    max_metrics_per_minute: int = 1000

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_json_logging: bool = False
    enable_structured_logging: bool = True

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    enable_health_checks: bool = True
    health_check_interval: int = 30
    enable_metrics_endpoint: bool = True
    metrics_path: str = "/metrics"
    enable_tracing: bool = False
    tracing_endpoint: Optional[str] = None

@dataclass
class ExportConfig:
    """Export configuration"""
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    supported_formats: List[str] = None
    temp_dir: str = "temp"
    output_dir: str = "exports"
    enable_compression: bool = True
    compression_level: int = 6
    max_concurrent_exports: int = 5

@dataclass
class CollaborationConfig:
    """Collaboration configuration"""
    max_sessions: int = 100
    max_users_per_session: int = 10
    session_timeout: int = 3600
    enable_real_time: bool = True
    websocket_timeout: int = 30
    enable_typing_indicators: bool = True
    enable_cursor_tracking: bool = True
    enable_comment_system: bool = True

@dataclass
class AdvancedConfig:
    """Advanced configuration container"""
    environment: Environment
    debug: bool = False
    database: DatabaseConfig
    redis: RedisConfig
    ai: AIConfig
    security: SecurityConfig
    cache: CacheConfig
    performance: PerformanceConfig
    logging: LoggingConfig
    monitoring: MonitoringConfig
    export: ExportConfig
    collaboration: CollaborationConfig

class AdvancedConfigManager:
    """
    Advanced configuration manager with environment-specific settings
    """
    
    def __init__(self, config_dir: str = "config"):
        """Initialize configuration manager"""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Load environment
        self.environment = Environment(os.getenv("ENVIRONMENT", "development"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Load configuration
        self.config = self._load_configuration()
        
        logger.info(f"Configuration loaded for environment: {self.environment.value}")

    def _load_configuration(self) -> AdvancedConfig:
        """Load configuration from multiple sources"""
        try:
            # Load base configuration
            base_config = self._load_base_config()
            
            # Load environment-specific configuration
            env_config = self._load_environment_config()
            
            # Load secrets from environment variables
            secrets_config = self._load_secrets_config()
            
            # Merge configurations
            merged_config = self._merge_configurations(base_config, env_config, secrets_config)
            
            # Validate configuration
            self._validate_configuration(merged_config)
            
            return merged_config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration from files"""
        config = {}
        
        # Try different configuration file formats
        config_files = [
            "config.yaml",
            "config.yml", 
            "config.json",
            "config.toml"
        ]
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                try:
                    if config_file.endswith(('.yaml', '.yml')):
                        with open(config_path, 'r') as f:
                            file_config = yaml.safe_load(f)
                    elif config_file.endswith('.json'):
                        with open(config_path, 'r') as f:
                            file_config = json.load(f)
                    elif config_file.endswith('.toml'):
                        with open(config_path, 'r') as f:
                            file_config = toml.load(f)
                    
                    config.update(file_config)
                    logger.info(f"Loaded configuration from {config_file}")
                    break
                    
                except Exception as e:
                    logger.warning(f"Error loading {config_file}: {e}")
                    continue
        
        return config

    def _load_environment_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration"""
        env_config_file = f"config_{self.environment.value}.yaml"
        env_config_path = self.config_dir / env_config_file
        
        if env_config_path.exists():
            try:
                with open(env_config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Error loading environment config: {e}")
        
        return {}

    def _load_secrets_config(self) -> Dict[str, Any]:
        """Load secrets from environment variables"""
        secrets = {}
        
        # Database secrets
        if os.getenv("DATABASE_URL"):
            secrets["database"] = {"url": os.getenv("DATABASE_URL")}
        
        # Redis secrets
        if os.getenv("REDIS_URL"):
            secrets["redis"] = {"url": os.getenv("REDIS_URL")}
        
        # AI API keys
        if os.getenv("OPENAI_API_KEY"):
            secrets["ai"] = {"openai_api_key": os.getenv("OPENAI_API_KEY")}
        
        if os.getenv("ANTHROPIC_API_KEY"):
            if "ai" not in secrets:
                secrets["ai"] = {}
            secrets["ai"]["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")
        
        # Security secrets
        if os.getenv("SECRET_KEY"):
            secrets["security"] = {"secret_key": os.getenv("SECRET_KEY")}
        
        return secrets

    def _merge_configurations(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries"""
        merged = {}
        
        for config in configs:
            merged = self._deep_merge(merged, config)
        
        return merged

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

    def _validate_configuration(self, config: Dict[str, Any]):
        """Validate configuration"""
        required_sections = ["database", "redis", "ai", "security"]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate database URL
        if not config["database"].get("url"):
            raise ValueError("Database URL is required")
        
        # Validate Redis URL
        if not config["redis"].get("url"):
            raise ValueError("Redis URL is required")
        
        # Validate security secret key
        if not config["security"].get("secret_key"):
            raise ValueError("Security secret key is required")

    def get_config(self) -> AdvancedConfig:
        """Get the complete configuration"""
        return self.config

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        return DatabaseConfig(**self.config["database"])

    def get_redis_config(self) -> RedisConfig:
        """Get Redis configuration"""
        return RedisConfig(**self.config["redis"])

    def get_ai_config(self) -> AIConfig:
        """Get AI configuration"""
        return AIConfig(**self.config["ai"])

    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        return SecurityConfig(**self.config["security"])

    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration"""
        return CacheConfig(**self.config.get("cache", {}))

    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration"""
        return PerformanceConfig(**self.config.get("performance", {}))

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration"""
        return LoggingConfig(**self.config.get("logging", {}))

    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        return MonitoringConfig(**self.config.get("monitoring", {}))

    def get_export_config(self) -> ExportConfig:
        """Get export configuration"""
        return ExportConfig(**self.config.get("export", {}))

    def get_collaboration_config(self) -> CollaborationConfig:
        """Get collaboration configuration"""
        return CollaborationConfig(**self.config.get("collaboration", {}))

    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == Environment.DEVELOPMENT

    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == Environment.PRODUCTION

    def is_testing(self) -> bool:
        """Check if running in testing mode"""
        return self.environment == Environment.TESTING

    def save_config(self, config: AdvancedConfig, filename: str = "config.yaml"):
        """Save configuration to file"""
        try:
            config_path = self.config_dir / filename
            
            # Convert to dictionary
            config_dict = asdict(config)
            
            # Save as YAML
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise

    def reload_config(self):
        """Reload configuration"""
        try:
            self.config = self._load_configuration()
            logger.info("Configuration reloaded successfully")
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
            raise

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary (without secrets)"""
        try:
            config_dict = asdict(self.config)
            
            # Remove sensitive information
            if "ai" in config_dict:
                if "openai_api_key" in config_dict["ai"]:
                    config_dict["ai"]["openai_api_key"] = "***"
                if "anthropic_api_key" in config_dict["ai"]:
                    config_dict["ai"]["anthropic_api_key"] = "***"
            
            if "security" in config_dict:
                if "secret_key" in config_dict["security"]:
                    config_dict["security"]["secret_key"] = "***"
            
            if "database" in config_dict:
                if "url" in config_dict["database"]:
                    # Mask password in database URL
                    url = config_dict["database"]["url"]
                    if "@" in url and "://" in url:
                        parts = url.split("://")
                        if len(parts) == 2:
                            auth_part = parts[1].split("@")[0]
                            if ":" in auth_part:
                                user_pass = auth_part.split(":")
                                if len(user_pass) == 2:
                                    config_dict["database"]["url"] = f"{parts[0]}://{user_pass[0]}:***@{parts[1].split('@', 1)[1]}"
            
            return config_dict
            
        except Exception as e:
            logger.error(f"Error getting config summary: {e}")
            return {}

# Global configuration manager instance
config_manager = AdvancedConfigManager()

# Convenience functions
def get_config() -> AdvancedConfig:
    """Get the complete configuration"""
    return config_manager.get_config()

def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return config_manager.get_database_config()

def get_redis_config() -> RedisConfig:
    """Get Redis configuration"""
    return config_manager.get_redis_config()

def get_ai_config() -> AIConfig:
    """Get AI configuration"""
    return config_manager.get_ai_config()

def get_security_config() -> SecurityConfig:
    """Get security configuration"""
    return config_manager.get_security_config()

def get_cache_config() -> CacheConfig:
    """Get cache configuration"""
    return config_manager.get_cache_config()

def get_performance_config() -> PerformanceConfig:
    """Get performance configuration"""
    return config_manager.get_performance_config()

def get_logging_config() -> LoggingConfig:
    """Get logging configuration"""
    return config_manager.get_logging_config()

def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration"""
    return config_manager.get_monitoring_config()

def get_export_config() -> ExportConfig:
    """Get export configuration"""
    return config_manager.get_export_config()

def get_collaboration_config() -> CollaborationConfig:
    """Get collaboration configuration"""
    return config_manager.get_collaboration_config()

def is_development() -> bool:
    """Check if running in development mode"""
    return config_manager.is_development()

def is_production() -> bool:
    """Check if running in production mode"""
    return config_manager.is_production()

def is_testing() -> bool:
    """Check if running in testing mode"""
    return config_manager.is_testing()



























