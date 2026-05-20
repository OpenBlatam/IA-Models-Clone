"""
Sistema de configuración avanzado
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from enum import Enum


class Environment(str, Enum):
    """Entornos de ejecución"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Configuración de base de datos"""
    path: str = "dermatology_history.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_connections: int = 10


@dataclass
class CacheConfig:
    """Configuración de caché"""
    enabled: bool = True
    memory_cache_size: int = 100  # MB
    disk_cache_enabled: bool = True
    disk_cache_path: str = "cache"
    ttl_seconds: int = 3600  # 1 hora


@dataclass
class RateLimitConfig:
    """Configuración de rate limiting"""
    enabled: bool = True
    max_requests: int = 100
    window_seconds: int = 60
    per_user: bool = True


@dataclass
class MLConfig:
    """Configuración de ML"""
    enabled: bool = False
    model_path: Optional[str] = None
    confidence_threshold: float = 0.7
    batch_size: int = 1


@dataclass
class NotificationConfig:
    """Configuración de notificaciones"""
    email_enabled: bool = False
    push_enabled: bool = False
    sms_enabled: bool = False
    in_app_enabled: bool = True


@dataclass
class WebhookConfig:
    """Configuración de webhooks"""
    enabled: bool = True
    timeout_seconds: int = 10
    retry_attempts: int = 3
    signature_required: bool = True


@dataclass
class SecurityConfig:
    """Configuración de seguridad"""
    jwt_secret: Optional[str] = None
    jwt_expiration_hours: int = 24
    require_auth: bool = False
    allowed_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class Settings:
    """Configuración principal del sistema"""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: str = "INFO"
    api_version: str = "1.6.0"
    
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    webhooks: WebhookConfig = field(default_factory=WebhookConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Paths
    data_dir: str = "data"
    logs_dir: str = "logs"
    backups_dir: str = "backups"
    
    def __post_init__(self):
        """Inicializa directorios"""
        Path(self.data_dir).mkdir(exist_ok=True)
        Path(self.logs_dir).mkdir(exist_ok=True)
        Path(self.backups_dir).mkdir(exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'Settings':
        """
        Carga configuración desde variables de entorno
        
        Returns:
            Instancia de Settings
        """
        env = os.getenv("ENVIRONMENT", "development")
        
        settings = cls(
            environment=Environment(env),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            api_version=os.getenv("API_VERSION", "1.6.0")
        )
        
        # Database
        settings.database.path = os.getenv("DB_PATH", settings.database.path)
        settings.database.backup_enabled = os.getenv(
            "DB_BACKUP_ENABLED", "true"
        ).lower() == "true"
        
        # Cache
        settings.cache.enabled = os.getenv(
            "CACHE_ENABLED", "true"
        ).lower() == "true"
        settings.cache.memory_cache_size = int(
            os.getenv("CACHE_MEMORY_SIZE", "100")
        )
        
        # Rate Limit
        settings.rate_limit.enabled = os.getenv(
            "RATE_LIMIT_ENABLED", "true"
        ).lower() == "true"
        settings.rate_limit.max_requests = int(
            os.getenv("RATE_LIMIT_MAX", "100")
        )
        
        # ML
        settings.ml.enabled = os.getenv(
            "ML_ENABLED", "false"
        ).lower() == "true"
        settings.ml.model_path = os.getenv("ML_MODEL_PATH")
        
        # Security
        settings.security.jwt_secret = os.getenv("JWT_SECRET")
        settings.security.require_auth = os.getenv(
            "REQUIRE_AUTH", "false"
        ).lower() == "true"
        
        return settings
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Settings':
        """
        Carga configuración desde archivo JSON
        
        Args:
            config_path: Path al archivo de configuración
            
        Returns:
            Instancia de Settings
        """
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        settings = cls()
        
        # Actualizar desde archivo
        for key, value in config_data.items():
            if hasattr(settings, key):
                if isinstance(value, dict):
                    # Para sub-configuraciones
                    sub_config = getattr(settings, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_config, sub_key):
                            setattr(sub_config, sub_key, sub_value)
                else:
                    setattr(settings, key, value)
        
        return settings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte configuración a diccionario"""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "log_level": self.log_level,
            "api_version": self.api_version,
            "database": {
                "path": self.database.path,
                "backup_enabled": self.database.backup_enabled,
                "backup_interval_hours": self.database.backup_interval_hours
            },
            "cache": {
                "enabled": self.cache.enabled,
                "memory_cache_size": self.cache.memory_cache_size,
                "disk_cache_enabled": self.cache.disk_cache_enabled,
                "ttl_seconds": self.cache.ttl_seconds
            },
            "rate_limit": {
                "enabled": self.rate_limit.enabled,
                "max_requests": self.rate_limit.max_requests,
                "window_seconds": self.rate_limit.window_seconds
            }
        }
    
    def save_to_file(self, config_path: str):
        """Guarda configuración en archivo JSON"""
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Instancia global de configuración
settings = Settings.from_env()
