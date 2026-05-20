"""
Configuration management for Enhanced Blaze AI.

This module provides centralized configuration management with validation,
environment variable support, and type safety.
"""

import os
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from pydantic.settings import BaseSettings


class ServerConfig(BaseModel):
    """Server configuration settings."""
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=4, description="Number of worker processes")
    reload: bool = Field(default=False, description="Enable auto-reload for development")


class CORSConfig(BaseModel):
    """CORS configuration settings."""
    allow_origins: List[str] = Field(default=["*"], description="Allowed origins")
    allow_credentials: bool = Field(default=True, description="Allow credentials")
    allow_methods: List[str] = Field(default=["*"], description="Allowed HTTP methods")
    allow_headers: List[str] = Field(default=["*"], description="Allowed headers")


class SecurityConfig(BaseModel):
    """Security configuration settings."""
    enable_authentication: bool = Field(default=True, description="Enable JWT authentication")
    enable_authorization: bool = Field(default=True, description="Enable role-based access control")
    enable_threat_detection: bool = Field(default=True, description="Enable threat detection")
    jwt_secret_key: str = Field(default="your-super-secret-key-change-in-production", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration: int = Field(default=3600, description="JWT expiration time in seconds")
    api_key_required: bool = Field(default=False, description="Require API key for requests")
    max_request_size: int = Field(default=10 * 1024 * 1024, description="Maximum request size in bytes")


class RateLimitingConfig(BaseModel):
    """Rate limiting configuration settings."""
    algorithm: str = Field(default="adaptive", description="Rate limiting algorithm")
    requests_per_minute: int = Field(default=100, description="Requests per minute limit")
    requests_per_hour: int = Field(default=1000, description="Requests per hour limit")
    requests_per_day: int = Field(default=10000, description="Requests per day limit")
    burst_limit: int = Field(default=50, description="Burst limit")
    enable_distributed: bool = Field(default=False, description="Enable distributed rate limiting")
    redis_host: str = Field(default="localhost", description="Redis host for distributed rate limiting")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")


class MonitoringConfig(BaseModel):
    """Performance monitoring configuration settings."""
    enable_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    enable_profiling: bool = Field(default=True, description="Enable function profiling")
    enable_system_metrics: bool = Field(default=True, description="Enable system metrics collection")
    enable_custom_metrics: bool = Field(default=True, description="Enable custom metrics")
    metrics_interval: float = Field(default=1.0, description="Metrics collection interval in seconds")
    max_metrics_history: int = Field(default=1000, description="Maximum metrics history size")
    enable_alerting: bool = Field(default=True, description="Enable performance alerts")
    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus metrics export")
    prometheus_port: int = Field(default=9090, description="Prometheus metrics port")


class ErrorHandlingConfig(BaseModel):
    """Error handling configuration settings."""
    enable_circuit_breaker: bool = Field(default=True, description="Enable circuit breaker pattern")
    enable_retry_logic: bool = Field(default=True, description="Enable retry logic")
    enable_error_recovery: bool = Field(default=True, description="Enable error recovery strategies")
    max_retry_attempts: int = Field(default=3, description="Maximum retry attempts")
    retry_backoff_factor: float = Field(default=2.0, description="Retry backoff factor")
    circuit_breaker_threshold: int = Field(default=5, description="Circuit breaker failure threshold")
    circuit_breaker_timeout: int = Field(default=60, description="Circuit breaker timeout in seconds")


class AIModelsConfig(BaseModel):
    """AI models configuration settings."""
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    stability_api_key: Optional[str] = Field(default=None, description="Stability AI API key")
    default_model: str = Field(default="gpt-3.5-turbo", description="Default AI model")
    max_tokens: int = Field(default=2048, description="Maximum tokens for AI responses")
    temperature: float = Field(default=0.7, description="AI model temperature")
    enable_fallback: bool = Field(default=True, description="Enable model fallback")


class DatabaseConfig(BaseModel):
    """Database configuration settings."""
    enable_database: bool = Field(default=False, description="Enable database support")
    database_url: Optional[str] = Field(default=None, description="Database connection URL")
    pool_size: int = Field(default=20, description="Database connection pool size")
    max_overflow: int = Field(default=30, description="Maximum database connections overflow")


class CacheConfig(BaseModel):
    """Cache configuration settings."""
    enable_cache: bool = Field(default=True, description="Enable caching")
    cache_type: str = Field(default="memory", description="Cache type (memory, redis)")
    redis_host: str = Field(default="localhost", description="Redis host for caching")
    redis_port: int = Field(default=6379, description="Redis port for caching")
    redis_db: int = Field(default=1, description="Redis database for caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")


class LoggingConfig(BaseModel):
    """Logging configuration settings."""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    file: Optional[str] = Field(default=None, description="Log file path")
    max_size: int = Field(default=10 * 1024 * 1024, description="Maximum log file size in bytes")
    backup_count: int = Field(default=5, description="Number of log file backups")
    enable_structured_logging: bool = Field(default=True, description="Enable structured logging")


class ExternalServicesConfig(BaseModel):
    """External services configuration settings."""
    enable_webhooks: bool = Field(default=False, description="Enable webhook support")
    webhook_urls: List[str] = Field(default=[], description="Webhook URLs")
    enable_third_party_integrations: bool = Field(default=False, description="Enable third-party integrations")
    integration_keys: Dict[str, str] = Field(default={}, description="Integration API keys")


class DeploymentConfig(BaseModel):
    """Deployment configuration settings."""
    environment: str = Field(default="development", description="Deployment environment")
    enable_health_checks: bool = Field(default=True, description="Enable health check endpoints")
    enable_metrics: bool = Field(default=True, description="Enable metrics endpoints")
    enable_documentation: bool = Field(default=True, description="Enable API documentation")
    trusted_hosts: List[str] = Field(default=["*"], description="Trusted host headers")


class FeatureFlagsConfig(BaseModel):
    """Feature flags configuration settings."""
    enable_advanced_security: bool = Field(default=True, description="Enable advanced security features")
    enable_performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    enable_circuit_breaker: bool = Field(default=True, description="Enable circuit breaker pattern")
    enable_error_recovery: bool = Field(default=True, description="Enable error recovery strategies")
    enable_audit_logging: bool = Field(default=True, description="Enable audit logging")
    enable_metrics_export: bool = Field(default=True, description="Enable metrics export")


class AppConfig(BaseModel):
    """Main application configuration."""
    app: Dict[str, Any] = Field(default_factory=dict, description="Application metadata")
    server: ServerConfig = Field(default_factory=ServerConfig, description="Server configuration")
    cors: CORSConfig = Field(default_factory=CORSConfig, description="CORS configuration")
    security: SecurityConfig = Field(default_factory=SecurityConfig, description="Security configuration")
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig, description="Rate limiting configuration")
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig, description="Monitoring configuration")
    error_handling: ErrorHandlingConfig = Field(default_factory=ErrorHandlingConfig, description="Error handling configuration")
    ai_models: AIModelsConfig = Field(default_factory=AIModelsConfig, description="AI models configuration")
    database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Database configuration")
    cache: CacheConfig = Field(default_factory=CacheConfig, description="Cache configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    external_services: ExternalServicesConfig = Field(default_factory=ExternalServicesConfig, description="External services configuration")
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig, description="Deployment configuration")
    features: FeatureFlagsConfig = Field(default_factory=FeatureFlagsConfig, description="Feature flags configuration")
    
    @validator('server')
    def validate_server_config(cls, v):
        """Validate server configuration."""
        if v.port < 1 or v.port > 65535:
            raise ValueError("Port must be between 1 and 65535")
        if v.workers < 1:
            raise ValueError("Workers must be at least 1")
        return v
    
    @validator('security')
    def validate_security_config(cls, v):
        """Validate security configuration."""
        if v.enable_authentication and not v.jwt_secret_key:
            raise ValueError("JWT secret key is required when authentication is enabled")
        if v.jwt_expiration < 60:
            raise ValueError("JWT expiration must be at least 60 seconds")
        return v
    
    @validator('rate_limiting')
    def validate_rate_limiting_config(cls, v):
        """Validate rate limiting configuration."""
        if v.requests_per_minute < 1:
            raise ValueError("Requests per minute must be at least 1")
        if v.requests_per_hour < v.requests_per_minute:
            raise ValueError("Requests per hour must be greater than or equal to requests per minute")
        if v.requests_per_day < v.requests_per_hour:
            raise ValueError("Requests per day must be greater than or equal to requests per hour")
        return v
    
    @validator('monitoring')
    def validate_monitoring_config(cls, v):
        """Validate monitoring configuration."""
        if v.metrics_interval < 0.1:
            raise ValueError("Metrics interval must be at least 0.1 seconds")
        if v.max_metrics_history < 100:
            raise ValueError("Maximum metrics history must be at least 100")
        return v
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from file and environment variables.
    
    Args:
        config_path: Path to configuration file. If None, uses default paths.
    
    Returns:
        Loaded configuration object.
    
    Raises:
        FileNotFoundError: If configuration file is not found.
        yaml.YAMLError: If configuration file is invalid YAML.
        ValidationError: If configuration validation fails.
    """
    if config_path is None:
        # Try to find configuration file in common locations
        possible_paths = [
            "config-enhanced.yaml",
            "config.yaml",
            "config.yml",
            "config/config.yaml",
            "config/config.yml"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                config_path = path
                break
        else:
            # Use default configuration
            return AppConfig()
    
    # Load configuration from file
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Override with environment variables
        config_data = _override_from_env(config_data)
        
        # Create and validate configuration
        config = AppConfig(**config_data)
        
        return config
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")


def _override_from_env(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override configuration values with environment variables.
    
    Args:
        config_data: Configuration data from file.
    
    Returns:
        Configuration data with environment variable overrides.
    """
    # Security overrides
    if os.getenv("JWT_SECRET_KEY"):
        config_data.setdefault("security", {})["jwt_secret_key"] = os.getenv("JWT_SECRET_KEY")
    
    if os.getenv("API_KEY_REQUIRED"):
        config_data.setdefault("security", {})["api_key_required"] = os.getenv("API_KEY_REQUIRED").lower() == "true"
    
    # AI Models overrides
    if os.getenv("OPENAI_API_KEY"):
        config_data.setdefault("ai_models", {})["openai_api_key"] = os.getenv("OPENAI_API_KEY")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        config_data.setdefault("ai_models", {})["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")
    
    if os.getenv("STABILITY_API_KEY"):
        config_data.setdefault("ai_models", {})["stability_api_key"] = os.getenv("STABILITY_API_KEY")
    
    # Server overrides
    if os.getenv("API_HOST"):
        config_data.setdefault("server", {})["host"] = os.getenv("API_HOST")
    
    if os.getenv("API_PORT"):
        try:
            config_data.setdefault("server", {})["port"] = int(os.getenv("API_PORT"))
        except ValueError:
            pass
    
    if os.getenv("API_WORKERS"):
        try:
            config_data.setdefault("server", {})["workers"] = int(os.getenv("API_WORKERS"))
        except ValueError:
            pass
    
    # Redis overrides
    if os.getenv("REDIS_HOST"):
        config_data.setdefault("cache", {})["redis_host"] = os.getenv("REDIS_HOST")
        config_data.setdefault("rate_limiting", {})["redis_host"] = os.getenv("REDIS_HOST")
    
    if os.getenv("REDIS_PORT"):
        try:
            port = int(os.getenv("REDIS_PORT"))
            config_data.setdefault("cache", {})["redis_port"] = port
            config_data.setdefault("rate_limiting", {})["redis_port"] = port
        except ValueError:
            pass
    
    # Monitoring overrides
    if os.getenv("PROMETHEUS_ENABLED"):
        config_data.setdefault("monitoring", {})["prometheus_enabled"] = os.getenv("PROMETHEUS_ENABLED").lower() == "true"
    
    if os.getenv("PROMETHEUS_PORT"):
        try:
            config_data.setdefault("monitoring", {})["prometheus_port"] = int(os.getenv("PROMETHEUS_PORT"))
        except ValueError:
            pass
    
    # Environment override
    if os.getenv("APP_ENVIRONMENT"):
        config_data.setdefault("deployment", {})["environment"] = os.getenv("APP_ENVIRONMENT")
    
    return config_data


def get_config_summary(config: AppConfig) -> Dict[str, Any]:
    """
    Get a summary of the configuration for logging/debugging.
    
    Args:
        config: Application configuration.
    
    Returns:
        Configuration summary dictionary.
    """
    return {
        "server": {
            "host": config.server.host,
            "port": config.server.port,
            "workers": config.server.workers
        },
        "security": {
            "authentication_enabled": config.security.enable_authentication,
            "authorization_enabled": config.security.enable_authorization,
            "threat_detection_enabled": config.security.enable_threat_detection
        },
        "monitoring": {
            "enabled": config.monitoring.enable_monitoring,
            "prometheus_enabled": config.monitoring.prometheus_enabled,
            "prometheus_port": config.monitoring.prometheus_port
        },
        "features": {
            "rate_limiting": config.features.enable_rate_limiting,
            "circuit_breaker": config.features.enable_circuit_breaker,
            "error_recovery": config.features.enable_error_recovery
        },
        "environment": config.deployment.environment
    }
