"""
Core Configuration Module
Módulo de configuración centralizada del sistema
"""

import os
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field, validator
from enum import Enum

class Environment(str, Enum):
    """Entornos del sistema"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(str, Enum):
    """Niveles de logging"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class DatabaseConfig(BaseSettings):
    """Configuración de base de datos"""
    url: str = Field(..., env="DATABASE_URL")
    pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DATABASE_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, env="DATABASE_POOL_RECYCLE")
    echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    class Config:
        env_prefix = "DATABASE_"

class CacheConfig(BaseSettings):
    """Configuración de caché"""
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT")
    socket_connect_timeout: int = Field(default=5, env="REDIS_SOCKET_CONNECT_TIMEOUT")
    default_ttl: int = Field(default=3600, env="CACHE_DEFAULT_TTL")
    max_size: int = Field(default=10000, env="CACHE_MAX_SIZE")
    
    class Config:
        env_prefix = "CACHE_"

class LLMConfig(BaseSettings):
    """Configuración de servicios LLM"""
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    default_model: str = Field(default="gpt-3.5-turbo", env="LLM_DEFAULT_MODEL")
    max_tokens: int = Field(default=4000, env="LLM_MAX_TOKENS")
    temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    timeout: int = Field(default=30, env="LLM_TIMEOUT")
    retry_attempts: int = Field(default=3, env="LLM_RETRY_ATTEMPTS")
    
    class Config:
        env_prefix = "LLM_"

class SecurityConfig(BaseSettings):
    """Configuración de seguridad"""
    secret_key: str = Field(..., env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    algorithm: str = Field(default="HS256", env="SECURITY_ALGORITHM")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    class Config:
        env_prefix = "SECURITY_"

class MonitoringConfig(BaseSettings):
    """Configuración de monitoreo"""
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    class Config:
        env_prefix = "MONITORING_"

class PerformanceConfig(BaseSettings):
    """Configuración de rendimiento"""
    max_concurrent_requests: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    workers: int = Field(default=4, env="WORKERS")
    enable_compression: bool = Field(default=True, env="ENABLE_COMPRESSION")
    compression_min_size: int = Field(default=1000, env="COMPRESSION_MIN_SIZE")
    
    class Config:
        env_prefix = "PERFORMANCE_"

class Settings(BaseSettings):
    """Configuración principal del sistema"""
    
    # Configuración general
    app_name: str = Field(default="AI History Comparison System", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Configuraciones específicas
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    
    # Configuración de servidor
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator('debug', pre=True)
    def set_debug_from_environment(cls, v, values):
        """Configurar debug basado en el entorno"""
        if 'environment' in values:
            return values['environment'] == Environment.DEVELOPMENT
        return v
    
    def is_development(self) -> bool:
        """Verificar si está en desarrollo"""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Verificar si está en producción"""
        return self.environment == Environment.PRODUCTION
    
    def is_testing(self) -> bool:
        """Verificar si está en testing"""
        return self.environment == Environment.TESTING
    
    def get_database_url(self) -> str:
        """Obtener URL de base de datos"""
        return self.database.url
    
    def get_redis_url(self) -> str:
        """Obtener URL de Redis"""
        return self.cache.redis_url
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Obtener configuración de LLM"""
        return {
            "openai_api_key": self.llm.openai_api_key,
            "anthropic_api_key": self.llm.anthropic_api_key,
            "google_api_key": self.llm.google_api_key,
            "default_model": self.llm.default_model,
            "max_tokens": self.llm.max_tokens,
            "temperature": self.llm.temperature,
            "timeout": self.llm.timeout,
            "retry_attempts": self.llm.retry_attempts
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Obtener configuración de seguridad"""
        return {
            "secret_key": self.security.secret_key,
            "access_token_expire_minutes": self.security.access_token_expire_minutes,
            "algorithm": self.security.algorithm,
            "cors_origins": self.security.cors_origins,
            "cors_allow_credentials": self.security.cors_allow_credentials,
            "rate_limit_requests": self.security.rate_limit_requests,
            "rate_limit_window": self.security.rate_limit_window
        }

# Instancia global de configuración
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Obtener instancia de configuración (Singleton)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def reload_settings() -> Settings:
    """Recargar configuración"""
    global _settings
    _settings = Settings()
    return _settings

# Configuración por entorno
def get_environment_config(env: Environment) -> Dict[str, Any]:
    """Obtener configuración específica por entorno"""
    configs = {
        Environment.DEVELOPMENT: {
            "debug": True,
            "log_level": LogLevel.DEBUG,
            "database_echo": True,
            "enable_metrics": True
        },
        Environment.STAGING: {
            "debug": False,
            "log_level": LogLevel.INFO,
            "database_echo": False,
            "enable_metrics": True
        },
        Environment.PRODUCTION: {
            "debug": False,
            "log_level": LogLevel.WARNING,
            "database_echo": False,
            "enable_metrics": True,
            "workers": 8
        },
        Environment.TESTING: {
            "debug": True,
            "log_level": LogLevel.DEBUG,
            "database_echo": False,
            "enable_metrics": False
        }
    }
    return configs.get(env, {})