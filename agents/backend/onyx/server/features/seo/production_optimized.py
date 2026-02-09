from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
import sys
import logging
import asyncio
import signal
import time
import gc
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from functools import lru_cache, wraps
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest, Summary
import structlog
from structlog import get_logger
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
import redis.asyncio as redis
from redis.asyncio import ConnectionPool
from healthcheck import HealthCheck, EnvironmentDump
import psutil
import orjson
import httpx
from httpx import AsyncClient, Limits, Timeout
import uvloop
from cachetools import TTLCache, LRUCache
import asyncio_mqtt as mqtt
from dataclasses_json import dataclass_json
import aioredis
from aioredis import Redis
import aiofiles
import aiofiles.os
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
import weakref
from .config import Config, Environment
from .service import SEOService
from .api import ServiceManager, MetricsCollector, RequestValidator, ResponseFormatter
from .utils import PerformanceUtils, ValidationUtils, LoggingUtils
            import gzip
from typing import Any, List, Dict, Optional
"""
Configuración y utilidades ultra-optimizadas para producción del Servicio SEO.
Versión optimizada con mejoras de rendimiento, cache avanzado y monitoreo inteligente.
"""


# Configurar uvloop para mejor rendimiento
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())



@dataclass_json
@dataclass
class UltraOptimizedConfig:
    """Configuración ultra-optimizada para producción."""
    
    # Seguridad avanzada
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "your-secret-key-here"))
    allowed_hosts: List[str] = field(default_factory=lambda: os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(","))
    cors_origins: List[str] = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(","))
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "200"))
    rate_limit_burst: int = int(os.getenv("RATE_LIMIT_BURST", "50"))
    
    # Monitoreo inteligente
    sentry_dsn: Optional[str] = os.getenv("SENTRY_DSN")
    prometheus_enabled: bool = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
    health_check_enabled: bool = os.getenv("HEALTH_CHECK_ENABLED", "true").lower() == "true"
    metrics_interval: int = int(os.getenv("METRICS_INTERVAL", "15"))
    
    # Cache ultra-optimizado
    redis_url: Optional[str] = os.getenv("REDIS_URL")
    redis_pool_size: int = int(os.getenv("REDIS_POOL_SIZE", "50"))
    redis_max_connections: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "100"))
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))
    cache_max_size: int = int(os.getenv("CACHE_MAX_SIZE", "10000"))
    cache_compression: bool = os.getenv("CACHE_COMPRESSION", "true").lower() == "true"
    
    # Performance ultra-optimizada
    max_workers: int = int(os.getenv("MAX_WORKERS", "8"))
    worker_class: str = os.getenv("WORKER_CLASS", "uvicorn.workers.UvicornWorker")
    max_requests: int = int(os.getenv("MAX_REQUESTS", "2000"))
    max_requests_jitter: int = int(os.getenv("MAX_REQUESTS_JITTER", "200"))
    connection_pool_size: int = int(os.getenv("CONNECTION_POOL_SIZE", "100"))
    keepalive_timeout: int = int(os.getenv("KEEPALIVE_TIMEOUT", "60"))
    
    # HTTP Client optimizado
    http_timeout: float = float(os.getenv("HTTP_TIMEOUT", "30.0"))
    http_max_connections: int = int(os.getenv("HTTP_MAX_CONNECTIONS", "200"))
    http_max_keepalive: int = int(os.getenv("HTTP_MAX_KEEPALIVE", "50"))
    
    # Logging optimizado
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")
    log_file: Optional[str] = os.getenv("LOG_FILE")
    log_buffer_size: int = int(os.getenv("LOG_BUFFER_SIZE", "1000"))
    log_flush_interval: int = int(os.getenv("LOG_FLUSH_INTERVAL", "5"))
    
    # Circuit Breaker avanzado
    circuit_breaker_enabled: bool = os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
    circuit_breaker_failure_threshold: int = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
    circuit_breaker_recovery_timeout: int = int(os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60"))
    circuit_breaker_success_threshold: int = int(os.getenv("CIRCUIT_BREAKER_SUCCESS_THRESHOLD", "3"))
    
    # Graceful Shutdown
    graceful_timeout: int = int(os.getenv("GRACEFUL_TIMEOUT", "30"))
    shutdown_timeout: int = int(os.getenv("SHUTDOWN_TIMEOUT", "10"))
    
    # Background tasks
    background_tasks_enabled: bool = os.getenv("BACKGROUND_TASKS_ENABLED", "true").lower() == "true"
    task_queue_size: int = int(os.getenv("TASK_QUEUE_SIZE", "1000"))
    task_workers: int = int(os.getenv("TASK_WORKERS", "4"))


class UltraOptimizedCache:
    """Cache ultra-optimizado con múltiples niveles."""
    
    def __init__(self, config: UltraOptimizedConfig):
        
    """__init__ function."""
self.config = config
        self.logger = get_logger()
        
        # Cache en memoria (L1)
        self.l1_cache = TTLCache(
            maxsize=config.cache_max_size // 2,
            ttl=config.cache_ttl // 2
        )
        
        # Cache LRU para datos frecuentes (L2)
        self.l2_cache = LRUCache(maxsize=config.cache_max_size // 4)
        
        # Cache de compresión
        self.compression_cache = TTLCache(maxsize=1000, ttl=300)
        
        # Métricas de cache
        self.hit_counter = Counter('cache_hits_total', 'Cache hits', ['level'])
        self.miss_counter = Counter('cache_misses_total', 'Cache misses', ['level'])
        self.size_gauge = Gauge('cache_size', 'Cache size', ['level'])
        
        # Thread-safe locks
        self._l1_lock = threading.Lock()
        self._l2_lock = threading.Lock()
        
        # Actualizar métricas periódicamente
        asyncio.create_task(self._update_metrics())
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache con múltiples niveles."""
        # L1 Cache (más rápido)
        with self._l1_lock:
            if key in self.l1_cache:
                self.hit_counter.labels(level='l1').inc()
                return self.l1_cache[key]
        
        # L2 Cache
        with self._l2_lock:
            if key in self.l2_cache:
                self.hit_counter.labels(level='l2').inc()
                value = self.l2_cache[key]
                # Mover a L1
                with self._l1_lock:
                    self.l1_cache[key] = value
                return value
        
        self.miss_counter.labels(level='memory').inc()
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Establece valor en cache con compresión inteligente."""
        ttl = ttl or self.config.cache_ttl
        
        # Comprimir si es necesario
        if self.config.cache_compression and isinstance(value, (str, bytes)):
            compressed_value = await self._compress_value(value)
        else:
            compressed_value = value
        
        # L1 Cache
        with self._l1_lock:
            self.l1_cache[key] = compressed_value
        
        # L2 Cache (datos importantes)
        if len(str(value)) > 1000:  # Solo datos grandes en L2
            with self._l2_lock:
                self.l2_cache[key] = compressed_value
    
    async def _compress_value(self, value: Union[str, bytes]) -> bytes:
        """Comprime valor si es beneficioso."""
        if isinstance(value, str):
            value = value.encode('utf-8')
        
        # Cache de compresión
        if value in self.compression_cache:
            return self.compression_cache[value]
        
        # Comprimir solo si es beneficioso
        if len(value) > 1024:  # Solo comprimir datos grandes
            compressed = gzip.compress(value)
            if len(compressed) < len(value) * 0.8:  # Solo si ahorra >20%
                self.compression_cache[value] = compressed
                return compressed
        
        return value
    
    async def _update_metrics(self) -> Any:
        """Actualiza métricas de cache periódicamente."""
        while True:
            with self._l1_lock:
                self.size_gauge.labels(level='l1').set(len(self.l1_cache))
            with self._l2_lock:
                self.size_gauge.labels(level='l2').set(len(self.l2_cache))
            await asyncio.sleep(self.config.metrics_interval)


class UltraOptimizedConnectionPool:
    """Pool de conexiones ultra-optimizado."""
    
    def __init__(self, config: UltraOptimizedConfig):
        
    """__init__ function."""
self.config = config
        self.logger = get_logger()
        
        # Pool de conexiones HTTP
        self.http_pool = None
        self.redis_pool = None
        
        # Métricas de pool
        self.connection_gauge = Gauge('connection_pool_size', 'Connection pool size', ['type'])
        self.connection_duration = Histogram('connection_duration_seconds', 'Connection duration', ['type'])
        
        # Estadísticas de pool
        self.stats = defaultdict(lambda: {'total': 0, 'active': 0, 'idle': 0})
    
    async async def setup_http_pool(self) -> AsyncClient:
        """Configura pool de conexiones HTTP optimizado."""
        limits = Limits(
            max_connections=self.config.http_max_connections,
            max_keepalive_connections=self.config.http_max_keepalive,
            keepalive_expiry=self.config.keepalive_timeout
        )
        
        timeout = Timeout(
            connect=5.0,
            read=self.config.http_timeout,
            write=10.0,
            pool=30.0
        )
        
        self.http_pool = AsyncClient(
            limits=limits,
            timeout=timeout,
            http2=True,
            follow_redirects=True
        )
        
        self.logger.info("HTTP connection pool configured", 
                        max_connections=self.config.http_max_connections,
                        max_keepalive=self.config.http_max_keepalive)
        
        return self.http_pool
    
    async def setup_redis_pool(self) -> Redis:
        """Configura pool de conexiones Redis optimizado."""
        self.redis_pool = aioredis.from_url(
            self.config.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=self.config.redis_max_connections,
            socket_keepalive=True,
            socket_keepalive_options={},
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        self.logger.info("Redis connection pool configured",
                        max_connections=self.config.redis_max_connections)
        
        return self.redis_pool
    
    async def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del pool de conexiones."""
        stats = {}
        
        if self.http_pool:
            stats['http'] = {
                'active_connections': len(self.http_pool._transport._pool._connections),
                'max_connections': self.config.http_max_connections
            }
        
        if self.redis_pool:
            stats['redis'] = {
                'active_connections': self.redis_pool.connection_pool._created_connections,
                'max_connections': self.config.redis_max_connections
            }
        
        return stats


class UltraOptimizedRateLimiter:
    """Rate limiter ultra-optimizado con ventana deslizante."""
    
    def __init__(self, config: UltraOptimizedConfig):
        
    """__init__ function."""
self.config = config
        self.logger = get_logger()
        
        # Ventanas de tiempo para rate limiting
        self.windows = defaultdict(lambda: deque(maxlen=config.rate_limit_per_minute))
        self.burst_windows = defaultdict(lambda: deque(maxlen=config.rate_limit_burst))
        
        # Métricas
        self.rate_limit_counter = Counter('rate_limit_hits_total', 'Rate limit hits', ['client_ip'])
        self.rate_limit_gauge = Gauge('rate_limit_active', 'Active rate limits', ['client_ip'])
        
        # Limpiar ventanas periódicamente
        asyncio.create_task(self._cleanup_windows())
    
    async def is_allowed(self, client_ip: str) -> bool:
        """Verifica si la IP está permitida según rate limiting."""
        now = time.time()
        
        # Ventana principal
        window = self.windows[client_ip]
        window.append(now)
        
        # Ventana de burst
        burst_window = self.burst_windows[client_ip]
        burst_window.append(now)
        
        # Verificar límites
        if len(window) > self.config.rate_limit_per_minute:
            self.rate_limit_counter.labels(client_ip=client_ip).inc()
            self.rate_limit_gauge.labels(client_ip=client_ip).set(1)
            return False
        
        if len(burst_window) > self.config.rate_limit_burst:
            self.rate_limit_counter.labels(client_ip=client_ip).inc()
            return False
        
        self.rate_limit_gauge.labels(client_ip=client_ip).set(0)
        return True
    
    async def _cleanup_windows(self) -> Any:
        """Limpia ventanas antiguas periódicamente."""
        while True:
            now = time.time()
            cutoff = now - 60  # 1 minuto
            
            for ip in list(self.windows.keys()):
                # Limpiar ventana principal
                while self.windows[ip] and self.windows[ip][0] < cutoff:
                    self.windows[ip].popleft()
                
                # Limpiar ventana de burst
                while self.burst_windows[ip] and self.burst_windows[ip][0] < cutoff:
                    self.burst_windows[ip].popleft()
                
                # Eliminar IPs sin actividad
                if not self.windows[ip] and not self.burst_windows[ip]:
                    del self.windows[ip]
                    del self.burst_windows[ip]
            
            await asyncio.sleep(30)  # Limpiar cada 30 segundos


class UltraOptimizedCircuitBreaker:
    """Circuit breaker ultra-optimizado con estados avanzados."""
    
    def __init__(self, config: UltraOptimizedConfig):
        
    """__init__ function."""
self.config = config
        self.logger = get_logger()
        
        # Estados del circuit breaker
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        
        # Métricas
        self.circuit_state_gauge = Gauge('circuit_breaker_state', 'Circuit breaker state')
        self.circuit_failures = Counter('circuit_breaker_failures_total', 'Circuit breaker failures')
        self.circuit_successes = Counter('circuit_breaker_successes_total', 'Circuit breaker successes')
        
        # Thread-safe
        self._lock = threading.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """Ejecuta función con circuit breaker."""
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.config.circuit_breaker_recovery_timeout:
                    self.state = "HALF_OPEN"
                    self.logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            
            with self._lock:
                if self.state == "HALF_OPEN":
                    self.success_count += 1
                    self.circuit_successes.inc()
                    
                    if self.success_count >= self.config.circuit_breaker_success_threshold:
                        self.state = "CLOSED"
                        self.failure_count = 0
                        self.success_count = 0
                        self.logger.info("Circuit breaker transitioning to CLOSED")
                
                self.last_success_time = time.time()
                self.circuit_state_gauge.set(0 if self.state == "CLOSED" else 1)
            
            return result
            
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                self.circuit_failures.inc()
                
                if self.failure_count >= self.config.circuit_breaker_failure_threshold:
                    self.state = "OPEN"
                    self.logger.warning("Circuit breaker transitioning to OPEN", 
                                      failure_count=self.failure_count)
                
                self.circuit_state_gauge.set(2 if self.state == "OPEN" else 1)
            
            raise e


class UltraOptimizedProductionManager:
    """Gestor de producción ultra-optimizado."""
    
    def __init__(self) -> Any:
        self.config = Config()
        self.ultra_config = UltraOptimizedConfig()
        self.logger = get_logger()
        
        # Componentes optimizados
        self.cache = UltraOptimizedCache(self.ultra_config)
        self.connection_pool = UltraOptimizedConnectionPool(self.ultra_config)
        self.rate_limiter = UltraOptimizedRateLimiter(self.ultra_config)
        self.circuit_breaker = UltraOptimizedCircuitBreaker(self.ultra_config)
        
        # Servicios
        self.app = None
        self.service_manager = None
        self.metrics_collector = None
        self.health_check = None
        
        # Métricas ultra-optimizadas
        self.request_counter = Counter('seo_requests_total', 'Total requests', 
                                     ['method', 'endpoint', 'status', 'client_ip'])
        self.request_duration = Histogram('seo_request_duration_seconds', 'Request duration', 
                                        ['method', 'endpoint'])
        self.active_requests = Gauge('seo_active_requests', 'Active requests')
        self.memory_usage = Gauge('memory_usage_bytes', 'Memory usage')
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU usage')
        self.response_size = Summary('response_size_bytes', 'Response size')
        
        # Background tasks
        self.task_queue = asyncio.Queue(maxsize=self.ultra_config.task_queue_size)
        self.task_workers = []
        
        # Setup
        self._setup_logging()
        self._setup_sentry()
        self._setup_health_check()
        
        # Iniciar workers de background
        if self.ultra_config.background_tasks_enabled:
            asyncio.create_task(self._start_background_workers())
    
    def _setup_logging(self) -> Any:
        """Configura logging ultra-optimizado."""
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.add_log_level_number,
        ]
        
        if self.ultra_config.log_format == "json":
            processors.append(structlog.processors.JSONRenderer(serializer=orjson.dumps))
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Logging asíncrono a archivo
        if self.ultra_config.log_file:
            asyncio.create_task(self._setup_async_logging())
    
    async def _setup_async_logging(self) -> Any:
        """Configura logging asíncrono para mejor rendimiento."""
        log_buffer = deque(maxlen=self.ultra_config.log_buffer_size)
        
        async def flush_logs():
            
    """flush_logs function."""
if log_buffer:
                async with aiofiles.open(self.ultra_config.log_file, 'a') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    await f.write(''.join(log_buffer))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                log_buffer.clear()
        
        # Flush periódico
        while True:
            await asyncio.sleep(self.ultra_config.log_flush_interval)
            await flush_logs()
    
    def _setup_sentry(self) -> Any:
        """Configura Sentry con optimizaciones."""
        if self.ultra_config.sentry_dsn:
            sentry_sdk.init(
                dsn=self.ultra_config.sentry_dsn,
                integrations=[
                    FastApiIntegration(),
                    LoggingIntegration(
                        level=logging.INFO,
                        event_level=logging.ERROR
                    ),
                ],
                traces_sample_rate=0.05,  # Reducido para mejor rendimiento
                environment=self.config.environment.value,
                before_send=self._sentry_before_send,
                max_breadcrumbs=50,  # Reducido para mejor rendimiento
            )
            self.logger.info("Sentry configured with optimizations")
    
    def _sentry_before_send(self, event, hint) -> Any:
        """Filtra eventos de Sentry para mejor rendimiento."""
        # No enviar eventos de rate limiting
        if 'rate_limit' in str(event).lower():
            return None
        
        # No enviar eventos de cache miss
        if 'cache_miss' in str(event).lower():
            return None
        
        return event
    
    def _setup_health_check(self) -> Any:
        """Configura health checks ultra-optimizados."""
        if self.ultra_config.health_check_enabled:
            self.health_check = HealthCheck()
            self.health_check.add_section("environment", EnvironmentDump())
            
            # Health checks optimizados
            self.health_check.add_check(self._check_redis_connection)
            self.health_check.add_check(self._check_disk_space)
            self.health_check.add_check(self._check_memory_usage)
            self.health_check.add_check(self._check_cpu_usage)
            self.health_check.add_check(self._check_connection_pools)
    
    async def _check_connection_pools(self) -> Any:
        """Verifica estado de pools de conexión."""
        try:
            stats = await self.connection_pool.get_stats()
            
            for pool_type, pool_stats in stats.items():
                if pool_stats['active_connections'] > pool_stats['max_connections'] * 0.9:
                    return False, f"{pool_type} pool at 90% capacity"
            
            return True, "Connection pools OK"
        except Exception as e:
            return False, f"Connection pool check failed: {str(e)}"
    
    async def _start_background_workers(self) -> Any:
        """Inicia workers de background para tareas asíncronas."""
        for i in range(self.ultra_config.task_workers):
            worker = asyncio.create_task(self._background_worker(f"worker-{i}"))
            self.task_workers.append(worker)
        
        self.logger.info("Background workers started", 
                        worker_count=self.ultra_config.task_workers)
    
    async def _background_worker(self, worker_name: str):
        """Worker de background para tareas asíncronas."""
        while True:
            try:
                task = await self.task_queue.get()
                await task()
                self.task_queue.task_done()
            except Exception as e:
                self.logger.error("Background worker error", 
                                worker=worker_name, error=str(e))
    
    def _create_app(self) -> FastAPI:
        """Crea aplicación FastAPI ultra-optimizada."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            
    """lifespan function."""
# Startup optimizado
            await self._startup()
            yield
            # Shutdown optimizado
            await self._shutdown()
        
        app = FastAPI(
            title="SEO Analysis Service - Ultra Optimized",
            description="Ultra-optimized SEO analysis service for production",
            version="2.0.0",
            docs_url="/docs" if not self.config.is_production else None,
            redoc_url="/redoc" if not self.config.is_production else None,
            lifespan=lifespan
        )
        
        # Middleware ultra-optimizado
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=self.ultra_config.allowed_hosts
        )
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.ultra_config.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
        
        app.add_middleware(GZipMiddleware, minimum_size=500)  # Reducido para mejor rendimiento
        
        # Middleware de métricas ultra-optimizado
        @app.middleware("http")
        async def ultra_metrics_middleware(request: Request, call_next):
            
    """ultra_metrics_middleware function."""
start_time = time.time()
            client_ip = request.client.host
            
            self.active_requests.inc()
            
            try:
                response = await call_next(request)
                duration = time.time() - start_time
                
                # Métricas optimizadas
                self.request_counter.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status=response.status_code,
                    client_ip=client_ip
                ).inc()
                
                self.request_duration.labels(
                    method=request.method,
                    endpoint=request.url.path
                ).observe(duration)
                
                # Métricas de tamaño de respuesta
                if hasattr(response, 'body'):
                    self.response_size.observe(len(response.body))
                
                return response
                
            except Exception as e:
                self.logger.error("Request error", 
                                method=request.method,
                                endpoint=request.url.path,
                                error=str(e))
                raise
            finally:
                self.active_requests.dec()
        
        # Middleware de rate limiting ultra-optimizado
        @app.middleware("http")
        async def ultra_rate_limit_middleware(request: Request, call_next):
            
    """ultra_rate_limit_middleware function."""
client_ip = request.client.host
            
            if not await self.rate_limiter.is_allowed(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded", "retry_after": 60}
                )
            
            return await call_next(request)
        
        # Endpoints ultra-optimizados
        self._add_ultra_optimized_endpoints(app)
        
        return app
    
    def _add_ultra_optimized_endpoints(self, app: FastAPI):
        """Añade endpoints ultra-optimizados."""
        
        @app.get("/health")
        async def ultra_health_check():
            """Health check ultra-optimizado."""
            if self.health_check:
                return self.health_check.run()
            return {"status": "healthy", "version": "2.0.0"}
        
        @app.get("/metrics")
        async def ultra_metrics():
            """Métricas ultra-optimizadas."""
            if self.ultra_config.prometheus_enabled:
                return generate_latest()
            raise HTTPException(status_code=404, detail="Metrics disabled")
        
        @app.get("/status")
        async def ultra_status():
            """Estado ultra-optimizado del sistema."""
            # Métricas del sistema en tiempo real
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            self.memory_usage.set(memory.used)
            self.cpu_usage.set(cpu_percent)
            
            return {
                "status": "running",
                "version": "2.0.0",
                "environment": self.config.environment.value,
                "uptime": time.time() - getattr(self, 'start_time', time.time()),
                "memory_usage": {
                    "used": memory.used,
                    "total": memory.total,
                    "percent": memory.percent
                },
                "cpu_usage": cpu_percent,
                "cache_stats": {
                    "l1_size": len(self.cache.l1_cache),
                    "l2_size": len(self.cache.l2_cache)
                },
                "connection_pools": await self.connection_pool.get_stats()
            }
        
        @app.get("/cache/stats")
        async def cache_stats():
            """Estadísticas del cache ultra-optimizado."""
            return {
                "l1_cache": {
                    "size": len(self.cache.l1_cache),
                    "maxsize": self.cache.l1_cache.maxsize,
                    "ttl": self.cache.l1_cache.ttl
                },
                "l2_cache": {
                    "size": len(self.cache.l2_cache),
                    "maxsize": self.cache.l2_cache.maxsize
                },
                "compression_cache": {
                    "size": len(self.cache.compression_cache)
                }
            }
    
    async def _startup(self) -> Any:
        """Startup ultra-optimizado."""
        self.start_time = time.time()
        self.logger.info("Starting ultra-optimized SEO service")
        
        # Configurar pools de conexión
        await self.connection_pool.setup_http_pool()
        if self.ultra_config.redis_url:
            await self.connection_pool.setup_redis_pool()
        
        # Configurar servicios
        self.service_manager = ServiceManager()
        self.metrics_collector = MetricsCollector()
        
        # Configurar signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)
        
        # Garbage collection optimizado
        gc.set_threshold(700, 10, 10)  # Más agresivo para mejor rendimiento
        
        self.logger.info("Ultra-optimized SEO service started successfully")
    
    async def _shutdown(self) -> Any:
        """Shutdown ultra-optimizado."""
        self.logger.info("Shutting down ultra-optimized SEO service")
        
        # Cerrar pools de conexión
        if self.connection_pool.http_pool:
            await self.connection_pool.http_pool.aclose()
        
        if self.connection_pool.redis_pool:
            await self.connection_pool.redis_pool.close()
        
        # Cancelar workers de background
        for worker in self.task_workers:
            worker.cancel()
        
        # Limpiar caches
        self.cache.l1_cache.clear()
        self.cache.l2_cache.clear()
        self.cache.compression_cache.clear()
        
        # Garbage collection final
        gc.collect()
        
        self.logger.info("Ultra-optimized SEO service shutdown complete")
    
    def _signal_handler(self, signum, frame) -> Any:
        """Maneja señales de sistema para shutdown graceful."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
        asyncio.create_task(self._graceful_shutdown())
    
    async def _graceful_shutdown(self) -> Any:
        """Shutdown graceful ultra-optimizado."""
        try:
            await asyncio.wait_for(self._shutdown(), timeout=self.ultra_config.shutdown_timeout)
        except asyncio.TimeoutError:
            self.logger.warning("Graceful shutdown timeout, forcing exit")
        finally:
            sys.exit(0)
    
    def run(self) -> Any:
        """Ejecuta la aplicación ultra-optimizada."""
        self.app = self._create_app()
        
        uvicorn.run(
            self.app,
            host=self.config.server.host,
            port=self.config.server.port,
            workers=self.ultra_config.max_workers,
            worker_class=self.ultra_config.worker_class,
            max_requests=self.ultra_config.max_requests,
            max_requests_jitter=self.ultra_config.max_requests_jitter,
            access_log=True,
            log_level=self.ultra_config.log_level.lower(),
            loop="uvloop",  # Usar uvloop para mejor rendimiento
        )


# Función principal ultra-optimizada
def main():
    """Función principal ultra-optimizada."""
    manager = UltraOptimizedProductionManager()
    manager.run()


match __name__:
    case "__main__":
    main() 