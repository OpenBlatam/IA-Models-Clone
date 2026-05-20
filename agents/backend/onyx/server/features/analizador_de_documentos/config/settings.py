"""
Configuración Centralizada del Sistema
=======================================

Sistema mejorado de gestión de configuración con validación
y soporte para variables de entorno.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseSettings, Field, validator
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuración del servidor"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    log_level: str = "info"
    workers: int = 1
    timeout_keep_alive: int = 5
    max_requests: int = 1000
    max_requests_jitter: int = 100


@dataclass
class ModelConfig:
    """Configuración de modelos"""
    model_name: str = "bert-base-multilingual-cased"
    device: str = "auto"  # auto, cuda, cpu
    max_length: int = 512
    batch_size: int = 16
    cache_dir: Optional[str] = None
    fine_tuned_model_path: Optional[str] = None


@dataclass
class CacheConfig:
    """Configuración de caché"""
    backend: str = "auto"  # auto, memory, disk, redis
    ttl: int = 3600  # Time to live en segundos
    max_size: int = 1000  # Tamaño máximo de caché
    redis_host: Optional[str] = None
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None


@dataclass
class SecurityConfig:
    """Configuración de seguridad"""
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key_enabled: bool = False
    api_key: Optional[str] = None
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 50
    rate_limit_per_hour: int = 1000


@dataclass
class PerformanceConfig:
    """Configuración de rendimiento"""
    enable_batch_processing: bool = True
    max_workers: int = 10
    batch_size: int = 100
    enable_async: bool = True
    memory_limit_mb: Optional[int] = None
    enable_profiling: bool = False


@dataclass
class MonitoringConfig:
    """Configuración de monitoreo"""
    enable_metrics: bool = True
    enable_health_checks: bool = True
    metrics_interval: int = 60  # segundos
    enable_telemetry: bool = False
    log_requests: bool = True


class Settings(BaseSettings):
    """Configuración principal del sistema"""
    
    # Información de la aplicación
    app_name: str = "Analizador de Documentos Inteligente"
    app_version: str = "3.8.0"
    app_description: str = "Sistema avanzado de análisis de documentos con fine-tuning"
    
    # Servidor
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    reload: bool = Field(default=True, env="RELOAD")
    log_level: str = Field(default="info", env="LOG_LEVEL")
    
    # Modelo
    model_name: str = Field(
        default="bert-base-multilingual-cased",
        env="MODEL_NAME"
    )
    device: str = Field(default="auto", env="DEVICE")
    max_length: int = Field(default=512, env="MAX_LENGTH")
    batch_size: int = Field(default=16, env="BATCH_SIZE")
    
    # Caché
    cache_backend: str = Field(default="auto", env="CACHE_BACKEND")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    redis_host: Optional[str] = Field(default=None, env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    
    # Seguridad
    cors_origins: str = Field(default="*", env="CORS_ORIGINS")
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_per_minute: int = Field(
        default=50,
        env="RATE_LIMIT_PER_MINUTE"
    )
    
    # Rendimiento
    max_workers: int = Field(default=10, env="MAX_WORKERS")
    enable_profiling: bool = Field(default=False, env="ENABLE_PROFILING")
    
    # Monitoreo
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    enable_health_checks: bool = Field(default=True, env="ENABLE_HEALTH_CHECKS")
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parsear CORS origins desde string a lista"""
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("device")
    def validate_device(cls, v):
        """Validar dispositivo"""
        valid_devices = ["auto", "cuda", "cpu"]
        if v not in valid_devices:
            logger.warning(f"Dispositivo '{v}' no válido, usando 'auto'")
            return "auto"
        return v
    
    @validator("cache_backend")
    def validate_cache_backend(cls, v):
        """Validar backend de caché"""
        valid_backends = ["auto", "memory", "disk", "redis"]
        if v not in valid_backends:
            logger.warning(f"Backend de caché '{v}' no válido, usando 'auto'")
            return "auto"
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Obtener configuración (singleton)"""
    return Settings()


def get_server_config() -> ServerConfig:
    """Obtener configuración del servidor"""
    settings = get_settings()
    return ServerConfig(
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level
    )


def get_model_config() -> ModelConfig:
    """Obtener configuración de modelos"""
    settings = get_settings()
    return ModelConfig(
        model_name=settings.model_name,
        device=settings.device,
        max_length=settings.max_length,
        batch_size=settings.batch_size
    )


def get_cache_config() -> CacheConfig:
    """Obtener configuración de caché"""
    settings = get_settings()
    return CacheConfig(
        backend=settings.cache_backend,
        ttl=settings.cache_ttl,
        redis_host=settings.redis_host,
        redis_port=settings.redis_port
    )


def get_security_config() -> SecurityConfig:
    """Obtener configuración de seguridad"""
    settings = get_settings()
    return SecurityConfig(
        cors_origins=settings.cors_origins,
        rate_limit_enabled=settings.rate_limit_enabled,
        rate_limit_per_minute=settings.rate_limit_per_minute
    )


def get_performance_config() -> PerformanceConfig:
    """Obtener configuración de rendimiento"""
    settings = get_settings()
    return PerformanceConfig(
        max_workers=settings.max_workers,
        enable_profiling=settings.enable_profiling
    )


def get_monitoring_config() -> MonitoringConfig:
    """Obtener configuración de monitoreo"""
    settings = get_settings()
    return MonitoringConfig(
        enable_metrics=settings.enable_metrics,
        enable_health_checks=settings.enable_health_checks
    )











