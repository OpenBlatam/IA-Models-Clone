"""
Configuration Loader
===================

Cargador de configuración desde YAML con validación.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OpenRouterConfig(BaseModel):
    """Configuración de OpenRouter."""
    api_key: str
    model: str = "anthropic/claude-3-haiku"
    max_tokens: int = 2000
    temperature: float = 0.7
    timeout: int = 60
    retry_count: int = 3
    retry_delay: float = 1.0
    exponential_base: float = 2.0


class DatabaseConfig(BaseModel):
    """Configuración de base de datos."""
    path: str = "artist_manager.db"
    pool_size: int = 10
    max_overflow: int = 20
    pool_recycle: int = 3600
    echo: bool = False


class CacheConfig(BaseModel):
    """Configuración de cache."""
    default_ttl_seconds: int = 1800
    max_size: int = 10000
    cleanup_interval: int = 3600


class RateLimitConfig(BaseModel):
    """Configuración de rate limiting."""
    enabled: bool = True
    max_requests: int = 100
    window_seconds: int = 60


class CircuitBreakerConfig(BaseModel):
    """Configuración de circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 60.0


class ServicesConfig(BaseModel):
    """Configuración de servicios."""
    persistence: bool = True
    notifications: bool = True
    analytics: bool = True
    webhooks: bool = True
    sync: bool = True
    default_interval_seconds: int = 3600


class APIConfig(BaseModel):
    """Configuración de API."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_enabled: bool = True
    cors_origins: list = ["*"]
    docs_enabled: bool = True


class LoggingConfig(BaseModel):
    """Configuración de logging."""
    format: str = "json"
    file: Optional[str] = "logs/artist_manager.log"
    max_bytes: int = 10485760
    backup_count: int = 5


class MetricsConfig(BaseModel):
    """Configuración de métricas."""
    enabled: bool = True
    retention_hours: int = 24
    export_interval: int = 3600


class HealthConfig(BaseModel):
    """Configuración de health checks."""
    enabled: bool = True
    interval_seconds: int = 30
    timeout_seconds: int = 10


class AppConfig(BaseModel):
    """Configuración completa de la aplicación."""
    app: Dict[str, Any] = Field(default_factory=dict)
    openrouter: OpenRouterConfig
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    services: ServicesConfig = Field(default_factory=ServicesConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)


class ConfigLoader:
    """Cargador de configuración."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializar cargador.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config_path = config_path or "config/config.yaml"
        self._config: Optional[AppConfig] = None
        self._logger = logger
    
    def load(self) -> AppConfig:
        """
        Cargar configuración.
        
        Returns:
            Configuración cargada
        """
        if self._config:
            return self._config
        
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            self._logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._load_defaults()
        
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            
            # Expandir variables de entorno
            config_data = self._expand_env_vars(config_data)
            
            # Validar y crear configuración
            self._config = AppConfig(**config_data)
            self._logger.info(f"Configuration loaded from {self.config_path}")
            
            return self._config
        
        except Exception as e:
            self._logger.error(f"Error loading config: {str(e)}, using defaults")
            return self._load_defaults()
    
    def _expand_env_vars(self, data: Any) -> Any:
        """
        Expandir variables de entorno en configuración.
        
        Args:
            data: Datos de configuración
        
        Returns:
            Datos con variables expandidas
        """
        if isinstance(data, dict):
            return {k: self._expand_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._expand_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            var_name = data[2:-1]
            return os.getenv(var_name, data)
        return data
    
    def _load_defaults(self) -> AppConfig:
        """Cargar configuración por defecto."""
        return AppConfig(
            openrouter=OpenRouterConfig(
                api_key=os.getenv("OPENROUTER_API_KEY", "")
            )
        )
    
    def reload(self) -> AppConfig:
        """Recargar configuración."""
        self._config = None
        return self.load()


# Singleton instance
_config_loader: Optional[ConfigLoader] = None


def get_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Obtener configuración (singleton).
    
    Args:
        config_path: Ruta al archivo de configuración
    
    Returns:
        Configuración
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader.load()

