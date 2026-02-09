"""
Sistema de configuración avanzado para Robot Movement AI v2.0
Soporte para múltiples entornos, validación, y hot reload
"""

import os
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseSettings, Field, validator


class Environment(str, Enum):
    """Entornos disponibles"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DatabaseConfig:
    """Configuración de base de datos"""
    url: str
    pool_size: int = 20
    max_overflow: int = 40
    pool_timeout: int = 30
    echo: bool = False


@dataclass
class CacheConfig:
    """Configuración de cache"""
    ttl: int = 300
    max_size: int = 1000
    enabled: bool = True


@dataclass
class SecurityConfig:
    """Configuración de seguridad"""
    secret_key: str
    enable_csrf: bool = False
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    rate_limit_per_day: int = 10000


@dataclass
class LoggingConfig:
    """Configuración de logging"""
    level: str = "INFO"
    file: Optional[str] = None
    dir: str = "logs"
    max_size: str = "10MB"
    backup_count: int = 10
    enable_json: bool = False
    enable_colors: bool = True


@dataclass
class MonitoringConfig:
    """Configuración de monitoreo"""
    prometheus_enabled: bool = True
    metrics_port: int = 9090
    grafana_enabled: bool = False
    grafana_port: int = 3000


class AppConfig(BaseSettings):
    """Configuración principal de la aplicación"""
    
    # Entorno
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8010, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    
    # Robot
    robot_ip: str = Field(default="192.168.1.100", env="ROBOT_IP")
    robot_port: int = Field(default=30001, env="ROBOT_PORT")
    robot_brand: str = Field(default="kuka", env="ROBOT_BRAND")
    ros_enabled: bool = Field(default=False, env="ROS_ENABLED")
    feedback_frequency: int = Field(default=1000, env="FEEDBACK_FREQUENCY")
    
    # LLM
    llm_provider: str = Field(default="openai", env="LLM_PROVIDER")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Database
    database_url: str = Field(default="sqlite:///./db/robots.db", env="DATABASE_URL")
    repository_type: str = Field(default="in_memory", env="REPOSITORY_TYPE")
    
    # Cache
    cache_ttl: int = Field(default=300, env="CACHE_TTL")
    cache_max_size: int = Field(default=1000, env="CACHE_MAX_SIZE")
    cache_enabled: bool = Field(default=True, env="ENABLE_CACHE")
    
    # Security
    secret_key: str = Field(default="change-this-in-production", env="SECRET_KEY")
    enable_csrf: bool = Field(default=False, env="ENABLE_CSRF")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    log_dir: str = Field(default="logs", env="LOG_DIR")
    log_enable_json: bool = Field(default=False, env="LOG_ENABLE_JSON")
    log_enable_colors: bool = Field(default=True, env="LOG_ENABLE_COLORS")
    
    # Monitoring
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Performance
    max_connections: int = Field(default=100, env="MAX_CONNECTIONS")
    worker_threads: int = Field(default=4, env="WORKER_THREADS")
    connection_timeout: int = Field(default=30, env="CONNECTION_TIMEOUT")
    request_timeout: int = Field(default=60, env="REQUEST_TIMEOUT")
    
    # Circuit Breaker
    circuit_breaker_failure_threshold: int = Field(default=5, env="CIRCUIT_BREAKER_FAILURE_THRESHOLD")
    circuit_breaker_timeout: int = Field(default=60, env="CIRCUIT_BREAKER_TIMEOUT")
    
    @validator("environment", pre=True)
    def parse_environment(cls, v):
        """Parsear entorno"""
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator("debug", pre=True)
    def parse_debug(cls, v):
        """Parsear debug"""
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return bool(v)
    
    @validator("secret_key")
    def validate_secret_key(cls, v, values):
        """Validar secret key en producción"""
        if values.get("environment") == Environment.PRODUCTION:
            if v == "change-this-in-production":
                raise ValueError("SECRET_KEY must be changed in production")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Instancia global de configuración
_config: Optional[AppConfig] = None


def load_config(env_file: Optional[str] = None) -> AppConfig:
    """
    Cargar configuración
    
    Args:
        env_file: Ruta al archivo .env (opcional)
        
    Returns:
        Configuración cargada
    """
    global _config
    
    if _config is None:
        if env_file:
            os.environ.setdefault("ENV_FILE", env_file)
        _config = AppConfig()
    
    return _config


def get_config() -> AppConfig:
    """Obtener configuración actual"""
    if _config is None:
        return load_config()
    return _config


def reload_config():
    """Recargar configuración desde archivo"""
    global _config
    _config = None
    return load_config()


def get_database_config() -> DatabaseConfig:
    """Obtener configuración de base de datos"""
    config = get_config()
    return DatabaseConfig(
        url=config.database_url,
        pool_size=20,
        max_overflow=40,
        pool_timeout=30,
        echo=config.debug
    )


def get_cache_config() -> CacheConfig:
    """Obtener configuración de cache"""
    config = get_config()
    return CacheConfig(
        ttl=config.cache_ttl,
        max_size=config.cache_max_size,
        enabled=config.cache_enabled
    )


def get_security_config() -> SecurityConfig:
    """Obtener configuración de seguridad"""
    config = get_config()
    return SecurityConfig(
        secret_key=config.secret_key,
        enable_csrf=config.enable_csrf,
        rate_limit_per_minute=config.rate_limit_per_minute,
        rate_limit_per_hour=1000,
        rate_limit_per_day=10000
    )


def get_logging_config() -> LoggingConfig:
    """Obtener configuración de logging"""
    config = get_config()
    return LoggingConfig(
        level=config.log_level,
        file=config.log_file,
        dir=config.log_dir,
        enable_json=config.log_enable_json,
        enable_colors=config.log_enable_colors
    )


def get_monitoring_config() -> MonitoringConfig:
    """Obtener configuración de monitoreo"""
    config = get_config()
    return MonitoringConfig(
        prometheus_enabled=config.prometheus_enabled,
        metrics_port=config.metrics_port
    )


def validate_config() -> List[str]:
    """
    Validar configuración y retornar lista de errores
    
    Returns:
        Lista de errores (vacía si todo está bien)
    """
    errors = []
    config = get_config()
    
    # Validar entorno
    if config.environment == Environment.PRODUCTION:
        if config.debug:
            errors.append("DEBUG should be False in production")
        
        if config.secret_key == "change-this-in-production":
            errors.append("SECRET_KEY must be changed in production")
    
    # Validar puertos
    if config.api_port < 1024 and config.environment == Environment.PRODUCTION:
        errors.append("API_PORT should be >= 1024 in production")
    
    # Validar URLs de base de datos
    if not config.database_url:
        errors.append("DATABASE_URL is required")
    
    return errors




