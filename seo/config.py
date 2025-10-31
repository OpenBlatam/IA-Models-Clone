from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Configuración modular y refactorizada para el Servicio SEO Ultra-Optimizado.
"""



class Environment(Enum):
    """Entornos de ejecución."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Niveles de logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ServerConfig:
    """Configuración del servidor."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = True
    log_level: LogLevel = LogLevel.INFO


@dataclass
class CacheConfig:
    """Configuración del cache."""
    ttl: int = 3600  # 1 hora
    max_size: int = 2000
    redis_url: Optional[str] = None
    enable_redis: bool = False


@dataclass
class SeleniumConfig:
    """Configuración de Selenium."""
    headless: bool = True
    timeout: int = 30
    chrome_driver_path: Optional[str] = None
    window_size: str = "1920,1080"
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


@dataclass
class PerformanceConfig:
    """Configuración de rendimiento."""
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    batch_size: int = 20
    enable_tracemalloc: bool = True
    enable_metrics: bool = True


@dataclass
class MonitoringConfig:
    """Configuración de monitoreo."""
    metrics_port: int = 9090
    health_check_interval: int = 60
    log_format: str = "json"
    enable_prometheus: bool = False


@dataclass
class AIConfig:
    """Configuración de IA."""
    openai_api_key: Optional[str] = None
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_retries: int = 3
    timeout: int = 30


@dataclass
class OptimizationConfig:
    """Configuración de optimizaciones."""
    httpx_timeout: float = 30.0
    httpx_max_keepalive: int = 20
    httpx_max_connections: int = 100
    lxml_encoding: str = "utf-8"
    lxml_remove_blank_text: bool = True
    orjson_option: int = 0
    tenacity_max_attempts: int = 3
    tenacity_multiplier: int = 1
    tenacity_max_wait: int = 10


class Config:
    """Configuración principal del servicio SEO."""
    
    def __init__(self) -> Any:
        self.environment = Environment(os.getenv("ENVIRONMENT", "development"))
        self.base_dir = Path(__file__).parent
        
        # Configuraciones específicas
        self.server = ServerConfig(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", 8000)),
            workers=int(os.getenv("WORKERS", 4)),
            reload=os.getenv("RELOAD", "true").lower() == "true",
            log_level=LogLevel(os.getenv("LOG_LEVEL", "INFO"))
        )
        
        self.cache = CacheConfig(
            ttl=int(os.getenv("CACHE_TTL", 3600)),
            max_size=int(os.getenv("CACHE_MAX_SIZE", 2000)),
            redis_url=os.getenv("REDIS_URL"),
            enable_redis=bool(os.getenv("REDIS_URL"))
        )
        
        self.selenium = SeleniumConfig(
            headless=os.getenv("SELENIUM_HEADLESS", "true").lower() == "true",
            timeout=int(os.getenv("SELENIUM_TIMEOUT", 30)),
            chrome_driver_path=os.getenv("CHROME_DRIVER_PATH"),
            window_size=os.getenv("SELENIUM_WINDOW_SIZE", "1920,1080"),
            user_agent=os.getenv("SELENIUM_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        )
        
        self.performance = PerformanceConfig(
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", 100)),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", 30)),
            batch_size=int(os.getenv("BATCH_SIZE", 20)),
            enable_tracemalloc=os.getenv("ENABLE_TRACEMALLOC", "true").lower() == "true",
            enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true"
        )
        
        self.monitoring = MonitoringConfig(
            metrics_port=int(os.getenv("METRICS_PORT", 9090)),
            health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", 60)),
            log_format=os.getenv("LOG_FORMAT", "json"),
            enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "false").lower() == "true"
        )
        
        self.ai = AIConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("AI_MODEL", "gpt-3.5-turbo"),
            temperature=float(os.getenv("AI_TEMPERATURE", 0.1)),
            max_retries=int(os.getenv("AI_MAX_RETRIES", 3)),
            timeout=int(os.getenv("AI_TIMEOUT", 30))
        )
        
        self.optimization = OptimizationConfig()
    
    @property
    def is_development(self) -> bool:
        """Verifica si está en modo desarrollo."""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        """Verifica si está en modo producción."""
        return self.environment == Environment.PRODUCTION
    
    async def get_httpx_config(self) -> Dict[str, Any]:
        """Obtiene configuración para httpx."""
        return {
            "timeout": self.optimization.httpx_timeout,
            "limits": {
                "max_keepalive_connections": self.optimization.httpx_max_keepalive,
                "max_connections": self.optimization.httpx_max_connections
            }
        }
    
    def get_lxml_config(self) -> Dict[str, Any]:
        """Obtiene configuración para lxml."""
        return {
            "encoding": self.optimization.lxml_encoding,
            "remove_blank_text": self.optimization.lxml_remove_blank_text
        }
    
    def get_orjson_config(self) -> Dict[str, Any]:
        """Obtiene configuración para orjson."""
        return {
            "option": self.optimization.orjson_option
        }
    
    def get_cachetools_config(self) -> Dict[str, Any]:
        """Obtiene configuración para cachetools."""
        return {
            "maxsize": self.cache.max_size,
            "ttl": self.cache.ttl
        }
    
    def get_tenacity_config(self) -> Dict[str, Any]:
        """Obtiene configuración para tenacity."""
        return {
            "stop_max_attempt_number": self.optimization.tenacity_max_attempts,
            "wait_exponential_multiplier": self.optimization.tenacity_multiplier,
            "wait_exponential_max": self.optimization.tenacity_max_wait
        }
    
    def get_selenium_options(self) -> Dict[str, Any]:
        """Obtiene opciones para Selenium."""
        return {
            "headless": self.selenium.headless,
            "timeout": self.selenium.timeout,
            "window_size": self.selenium.window_size,
            "user_agent": self.selenium.user_agent,
            "chrome_driver_path": self.selenium.chrome_driver_path
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración a diccionario."""
        return {
            "environment": self.environment.value,
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "workers": self.server.workers,
                "reload": self.server.reload,
                "log_level": self.server.log_level.value
            },
            "cache": {
                "ttl": self.cache.ttl,
                "max_size": self.cache.max_size,
                "redis_url": self.cache.redis_url,
                "enable_redis": self.cache.enable_redis
            },
            "selenium": {
                "headless": self.selenium.headless,
                "timeout": self.selenium.timeout,
                "window_size": self.selenium.window_size,
                "user_agent": self.selenium.user_agent
            },
            "performance": {
                "max_concurrent_requests": self.performance.max_concurrent_requests,
                "request_timeout": self.performance.request_timeout,
                "batch_size": self.performance.batch_size,
                "enable_tracemalloc": self.performance.enable_tracemalloc,
                "enable_metrics": self.performance.enable_metrics
            },
            "monitoring": {
                "metrics_port": self.monitoring.metrics_port,
                "health_check_interval": self.monitoring.health_check_interval,
                "log_format": self.monitoring.log_format,
                "enable_prometheus": self.monitoring.enable_prometheus
            },
            "ai": {
                "model_name": self.ai.model_name,
                "temperature": self.ai.temperature,
                "max_retries": self.ai.max_retries,
                "timeout": self.ai.timeout,
                "has_api_key": bool(self.ai.openai_api_key)
            },
            "optimization": {
                "httpx": self.get_httpx_config(),
                "lxml": self.get_lxml_config(),
                "orjson": self.get_orjson_config(),
                "cachetools": self.get_cachetools_config(),
                "tenacity": self.get_tenacity_config()
            }
        }
    
    def validate(self) -> bool:
        """Valida la configuración."""
        errors = []
        
        # Validar puerto
        if not (1024 <= self.server.port <= 65535):
            errors.append(f"Puerto inválido: {self.server.port}")
        
        # Validar workers
        if self.server.workers < 1:
            errors.append(f"Número de workers inválido: {self.server.workers}")
        
        # Validar cache
        if self.cache.max_size < 1:
            errors.append(f"Tamaño máximo de cache inválido: {self.cache.max_size}")
        
        if self.cache.ttl < 1:
            errors.append(f"TTL de cache inválido: {self.cache.ttl}")
        
        # Validar timeouts
        if self.performance.request_timeout < 1:
            errors.append(f"Timeout de request inválido: {self.performance.request_timeout}")
        
        if self.selenium.timeout < 1:
            errors.append(f"Timeout de Selenium inválido: {self.selenium.timeout}")
        
        # Validar batch size
        if self.performance.batch_size < 1 or self.performance.batch_size > 50:
            errors.append(f"Tamaño de lote inválido: {self.performance.batch_size}")
        
        if errors:
            raise ValueError(f"Errores de configuración: {', '.join(errors)}")
        
        return True
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Obtiene configuración de logging."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
                "json": {
                    "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "json" if self.monitoring.log_format == "json" else "default",
                    "level": self.server.log_level.value
                }
            },
            "root": {
                "handlers": ["console"],
                "level": self.server.log_level.value
            }
        }


# Instancia global de configuración
config = Config()

# Validar configuración al importar
try:
    config.validate()
except ValueError as e:
    print(f"Error de configuración: {e}")
    raise 