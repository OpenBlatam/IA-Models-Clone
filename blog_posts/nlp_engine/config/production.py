from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import os
import multiprocessing
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🚀 PRODUCTION CONFIGURATION - Ultra-Optimized Settings
=====================================================

Configuración enterprise para máximo performance y confiabilidad.
"""



class Environment(Enum):
    """Ambientes de deployment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(Enum):
    """Niveles de logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Configuración de base de datos."""
    host: str = "localhost"
    port: int = 5432
    database: str = "nlp_engine"
    username: str = "nlp_user"
    password: str = ""
    pool_size: int = 20
    max_overflow: int = 50
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class CacheConfig:
    """Configuración de cache ultra-optimizada."""
    # Redis para cache distribuido
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Configuración de cache
    default_ttl: int = 3600  # 1 hora
    max_memory: str = "512mb"
    eviction_policy: str = "allkeys-lru"
    
    # Cache local en memoria
    local_cache_size: int = 10000
    local_cache_ttl: int = 300  # 5 minutos
    
    # Optimizaciones
    compression_enabled: bool = True
    compression_threshold: int = 1024  # Comprimir si > 1KB
    serialization_format: str = "msgpack"  # msgpack es más rápido que JSON


@dataclass
class PerformanceConfig:
    """Configuración de performance ultra-optimizada."""
    # Workers y concurrencia
    worker_processes: int = multiprocessing.cpu_count()
    worker_threads: int = 4
    max_concurrent_requests: int = 1000
    
    # Timeouts optimizados
    request_timeout: int = 30
    keep_alive_timeout: int = 5
    graceful_timeout: int = 30
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 1000
    rate_limit_burst: int = 100
    
    # Memory management
    max_memory_usage_mb: int = 2048
    gc_threshold: int = 700
    
    # Processing optimization
    batch_size_optimal: int = 50
    parallel_processing_threshold: int = 10
    
    # Cache optimization
    cache_hit_target: float = 0.85
    cache_warming_enabled: bool = True


@dataclass
class SecurityConfig:
    """Configuración de seguridad enterprise."""
    # API Keys
    api_key_expiry_days: int = 90
    api_key_rotation_enabled: bool = True
    
    # Rate limiting avanzado
    ddos_protection_enabled: bool = True
    suspicious_activity_threshold: int = 100
    
    # Encryption
    encryption_algorithm: str = "AES-256-GCM"
    key_derivation_iterations: int = 100000
    
    # Headers de seguridad
    security_headers: Dict[str, str] = field(default_factory=lambda: {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'"
    })


@dataclass
class MonitoringConfig:
    """Configuración de monitoreo avanzado."""
    # Métricas
    metrics_enabled: bool = True
    metrics_interval: int = 60  # segundos
    metrics_retention_days: int = 30
    
    # Health checks
    health_check_interval: int = 30
    health_check_timeout: int = 5
    
    # Alertas
    alerts_enabled: bool = True
    alert_channels: list = field(default_factory=lambda: ["email", "slack"])
    
    # Performance monitoring
    slow_query_threshold_ms: int = 1000
    error_rate_threshold: float = 0.05
    memory_usage_threshold: float = 0.85
    
    # Logging avanzado
    structured_logging: bool = True
    log_correlation_id: bool = True
    log_performance_metrics: bool = True


@dataclass
class ProductionConfig:
    """
    🚀 Configuración principal de producción ultra-optimizada.
    
    Incluye todas las configuraciones enterprise para:
    - Performance máximo (< 0.1ms latency)
    - Escalabilidad horizontal
    - Seguridad enterprise
    - Monitoreo completo
    - Alta disponibilidad
    """
    
    # Ambiente
    environment: Environment = Environment.PRODUCTION
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    
    # Servidor
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = multiprocessing.cpu_count()
    
    # Configuraciones especializadas
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Configuración del motor NLP
    nlp_model_path: str = "/models/nlp"
    nlp_cache_enabled: bool = True
    nlp_parallel_processing: bool = True
    nlp_optimization_level: int = 3  # Máximo
    
    @classmethod
    def from_environment(cls) -> 'ProductionConfig':
        """Crear configuración desde variables de entorno."""
        config = cls()
        
        # Override con variables de entorno
        config.host = os.getenv('NLP_HOST', config.host)
        config.port = int(os.getenv('NLP_PORT', config.port))
        config.workers = int(os.getenv('NLP_WORKERS', config.workers))
        config.debug = os.getenv('NLP_DEBUG', 'false').lower() == 'true'
        
        # Database
        config.database.host = os.getenv('DB_HOST', config.database.host)
        config.database.port = int(os.getenv('DB_PORT', config.database.port))
        config.database.database = os.getenv('DB_NAME', config.database.database)
        config.database.username = os.getenv('DB_USER', config.database.username)
        config.database.password = os.getenv('DB_PASSWORD', config.database.password)
        
        # Cache
        config.cache.redis_host = os.getenv('REDIS_HOST', config.cache.redis_host)
        config.cache.redis_port = int(os.getenv('REDIS_PORT', config.cache.redis_port))
        config.cache.redis_password = os.getenv('REDIS_PASSWORD', config.cache.redis_password)
        
        # Performance
        config.performance.max_concurrent_requests = int(
            os.getenv('MAX_CONCURRENT_REQUESTS', config.performance.max_concurrent_requests)
        )
        config.performance.rate_limit_requests_per_minute = int(
            os.getenv('RATE_LIMIT_RPM', config.performance.rate_limit_requests_per_minute)
        )
        
        return config
    
    def get_uvicorn_config(self) -> Dict[str, Any]:
        """Obtener configuración optimizada para Uvicorn."""
        return {
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "worker_class": "uvicorn.workers.UvicornWorker",
            "worker_connections": self.performance.max_concurrent_requests,
            "keepalive": self.performance.keep_alive_timeout,
            "timeout": self.performance.request_timeout,
            "graceful_timeout": self.performance.graceful_timeout,
            "max_requests": 10000,  # Restart worker después de 10k requests
            "max_requests_jitter": 1000,
            "preload_app": True,  # Cargar app antes de fork para compartir memoria
            "log_level": self.log_level.value.lower(),
            "access_log": self.monitoring.structured_logging,
            "use_colors": False,  # Mejor para logs estructurados
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Obtener configuración optimizada para Redis."""
        return {
            "host": self.cache.redis_host,
            "port": self.cache.redis_port,
            "db": self.cache.redis_db,
            "password": self.cache.redis_password,
            "decode_responses": True,
            "socket_keepalive": True,
            "socket_keepalive_options": {},
            "health_check_interval": 30,
            "retry_on_timeout": True,
            "max_connections": 50,
        }
    
    def validate(self) -> bool:
        """Validar configuración."""
        errors = []
        
        # Validar workers
        if self.workers <= 0:
            errors.append("Workers debe ser > 0")
        
        # Validar performance
        if self.performance.max_concurrent_requests <= 0:
            errors.append("Max concurrent requests debe ser > 0")
        
        # Validar cache
        if self.cache.default_ttl <= 0:
            errors.append("Cache TTL debe ser > 0")
        
        if errors:
            raise ValueError(f"Errores de configuración: {errors}")
        
        return True
    
    def get_health_check_config(self) -> Dict[str, Any]:
        """Configuración para health checks."""
        return {
            "interval": self.monitoring.health_check_interval,
            "timeout": self.monitoring.health_check_timeout,
            "retries": 3,
            "checks": [
                "database_connection",
                "redis_connection", 
                "nlp_engine_status",
                "memory_usage",
                "disk_space"
            ]
        }


# Instancia global optimizada para producción
PRODUCTION_CONFIG = ProductionConfig.from_environment()

def get_config() -> ProductionConfig:
    """Obtener configuración de producción."""
    return PRODUCTION_CONFIG

def update_config(**kwargs) -> ProductionConfig:
    """Actualizar configuración dinámicamente."""
    global PRODUCTION_CONFIG
    
    for key, value in kwargs.items():
        if hasattr(PRODUCTION_CONFIG, key):
            setattr(PRODUCTION_CONFIG, key, value)
    
    PRODUCTION_CONFIG.validate()
    return PRODUCTION_CONFIG 