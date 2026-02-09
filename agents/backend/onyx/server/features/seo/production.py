from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
import sys
import logging
import asyncio
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog
from structlog import get_logger
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
import redis.asyncio as redis
from healthcheck import HealthCheck, EnvironmentDump
import psutil
import time
from .config import Config, Environment
from .service import SEOService
from .api import ServiceManager, MetricsCollector, RequestValidator, ResponseFormatter
from .utils import PerformanceUtils, ValidationUtils, LoggingUtils
from typing import Any, List, Dict, Optional
"""
Configuración y utilidades para producción del Servicio SEO Ultra-Optimizado.
"""




@dataclass
class ProductionConfig:
    """Configuración específica para producción."""
    
    # Seguridad
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "your-secret-key-here"))
    allowed_hosts: List[str] = field(default_factory=lambda: os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(","))
    cors_origins: List[str] = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(","))
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
    
    # Monitoreo
    sentry_dsn: Optional[str] = os.getenv("SENTRY_DSN")
    prometheus_enabled: bool = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
    health_check_enabled: bool = os.getenv("HEALTH_CHECK_ENABLED", "true").lower() == "true"
    
    # Base de datos y Cache
    redis_url: Optional[str] = os.getenv("REDIS_URL")
    database_url: Optional[str] = os.getenv("DATABASE_URL")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")
    log_file: Optional[str] = os.getenv("LOG_FILE")
    
    # Performance
    max_workers: int = int(os.getenv("MAX_WORKERS", "4"))
    worker_class: str = os.getenv("WORKER_CLASS", "uvicorn.workers.UvicornWorker")
    max_requests: int = int(os.getenv("MAX_REQUESTS", "1000"))
    max_requests_jitter: int = int(os.getenv("MAX_REQUESTS_JITTER", "100"))
    
    # SSL/TLS
    ssl_keyfile: Optional[str] = os.getenv("SSL_KEYFILE")
    ssl_certfile: Optional[str] = os.getenv("SSL_CERTFILE")
    
    # Graceful Shutdown
    graceful_timeout: int = int(os.getenv("GRACEFUL_TIMEOUT", "30"))
    
    # Circuit Breaker
    circuit_breaker_enabled: bool = os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
    circuit_breaker_failure_threshold: int = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
    circuit_breaker_recovery_timeout: int = int(os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60"))


class ProductionManager:
    """Gestor de producción con configuración avanzada."""
    
    def __init__(self) -> Any:
        self.config = Config()
        self.prod_config = ProductionConfig()
        self.logger = get_logger()
        self.app = None
        self.service_manager = None
        self.metrics_collector = None
        self.redis_client = None
        self.health_check = None
        
        # Métricas de Prometheus
        self.request_counter = Counter('seo_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
        self.request_duration = Histogram('seo_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
        self.active_requests = Gauge('seo_active_requests', 'Active requests')
        self.cache_hits = Counter('seo_cache_hits_total', 'Cache hits')
        self.cache_misses = Counter('seo_cache_misses_total', 'Cache misses')
        self.errors_total = Counter('seo_errors_total', 'Total errors', ['type'])
        
        # Circuit Breaker
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_open = False
        
        self._setup_logging()
        self._setup_sentry()
        self._setup_health_check()
    
    def _setup_logging(self) -> Any:
        """Configura logging estructurado para producción."""
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
        
        if self.prod_config.log_format == "json":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Configurar logging a archivo si se especifica
        if self.prod_config.log_file:
            logging.basicConfig(
                filename=self.prod_config.log_file,
                level=getattr(logging, self.prod_config.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def _setup_sentry(self) -> Any:
        """Configura Sentry para monitoreo de errores."""
        if self.prod_config.sentry_dsn:
            sentry_sdk.init(
                dsn=self.prod_config.sentry_dsn,
                integrations=[
                    FastApiIntegration(),
                    LoggingIntegration(
                        level=logging.INFO,
                        event_level=logging.ERROR
                    ),
                ],
                traces_sample_rate=0.1,
                environment=self.config.environment.value,
            )
            self.logger.info("Sentry configured for error monitoring")
    
    def _setup_health_check(self) -> Any:
        """Configura health checks."""
        if self.prod_config.health_check_enabled:
            self.health_check = HealthCheck()
            self.health_check.add_section("environment", EnvironmentDump())
            
            # Health checks personalizados
            self.health_check.add_check(self._check_redis_connection)
            self.health_check.add_check(self._check_disk_space)
            self.health_check.add_check(self._check_memory_usage)
            self.health_check.add_check(self._check_cpu_usage)
    
    async def _check_redis_connection(self) -> Any:
        """Verifica conexión a Redis."""
        if self.redis_client:
            try:
                await self.redis_client.ping()
                return True, "Redis connection OK"
            except Exception as e:
                return False, f"Redis connection failed: {str(e)}"
        return True, "Redis not configured"
    
    def _check_disk_space(self) -> Any:
        """Verifica espacio en disco."""
        try:
            disk_usage = psutil.disk_usage('/')
            free_percent = (disk_usage.free / disk_usage.total) * 100
            if free_percent > 10:
                return True, f"Disk space OK: {free_percent:.1f}% free"
            else:
                return False, f"Low disk space: {free_percent:.1f}% free"
        except Exception as e:
            return False, f"Disk check failed: {str(e)}"
    
    def _check_memory_usage(self) -> Any:
        """Verifica uso de memoria."""
        try:
            memory = psutil.virtual_memory()
            if memory.percent < 90:
                return True, f"Memory usage OK: {memory.percent:.1f}%"
            else:
                return False, f"High memory usage: {memory.percent:.1f}%"
        except Exception as e:
            return False, f"Memory check failed: {str(e)}"
    
    def _check_cpu_usage(self) -> Any:
        """Verifica uso de CPU."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent < 80:
                return True, f"CPU usage OK: {cpu_percent:.1f}%"
            else:
                return False, f"High CPU usage: {cpu_percent:.1f}%"
        except Exception as e:
            return False, f"CPU check failed: {str(e)}"
    
    async def _setup_redis(self) -> Any:
        """Configura conexión a Redis."""
        if self.prod_config.redis_url:
            try:
                self.redis_client = redis.from_url(
                    self.prod_config.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self.redis_client.ping()
                self.logger.info("Redis connection established")
            except Exception as e:
                self.logger.error(f"Failed to connect to Redis: {e}")
                self.redis_client = None
    
    def _create_app(self) -> FastAPI:
        """Crea la aplicación FastAPI con configuración de producción."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            
    """lifespan function."""
# Startup
            await self._startup()
            yield
            # Shutdown
            await self._shutdown()
        
        app = FastAPI(
            title="SEO Analysis Service",
            description="Ultra-optimized SEO analysis service for production",
            version="1.0.0",
            docs_url="/docs" if not self.config.is_production else None,
            redoc_url="/redoc" if not self.config.is_production else None,
            lifespan=lifespan
        )
        
        # Middleware de seguridad
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=self.prod_config.allowed_hosts
        )
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.prod_config.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
        
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Middleware de métricas
        @app.middleware("http")
        async def metrics_middleware(request: Request, call_next):
            
    """metrics_middleware function."""
start_time = time.time()
            self.active_requests.inc()
            
            try:
                response = await call_next(request)
                duration = time.time() - start_time
                
                self.request_counter.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status=response.status_code
                ).inc()
                
                self.request_duration.labels(
                    method=request.method,
                    endpoint=request.url.path
                ).observe(duration)
                
                return response
            except Exception as e:
                self.errors_total.labels(type=type(e).__name__).inc()
                raise
            finally:
                self.active_requests.dec()
        
        # Middleware de rate limiting
        @app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            
    """rate_limit_middleware function."""
if self.redis_client:
                client_ip = request.client.host
                key = f"rate_limit:{client_ip}"
                
                try:
                    current = await self.redis_client.get(key)
                    if current and int(current) >= self.prod_config.rate_limit_per_minute:
                        return JSONResponse(
                            status_code=429,
                            content={"error": "Rate limit exceeded"}
                        )
                    
                    pipe = self.redis_client.pipeline()
                    pipe.incr(key)
                    pipe.expire(key, 60)
                    await pipe.execute()
                    
                except Exception as e:
                    self.logger.warning(f"Rate limiting failed: {e}")
            
            return await call_next(request)
        
        # Endpoints de producción
        self._add_production_endpoints(app)
        
        return app
    
    def _add_production_endpoints(self, app: FastAPI):
        """Añade endpoints específicos para producción."""
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            if self.health_check:
                return self.health_check.run()
            return {"status": "healthy"}
        
        @app.get("/metrics")
        async def metrics():
            """Endpoint de métricas de Prometheus."""
            if self.prod_config.prometheus_enabled:
                return generate_latest()
            raise HTTPException(status_code=404, detail="Metrics disabled")
        
        @app.get("/status")
        async def status():
            """Endpoint de estado del sistema."""
            return {
                "status": "running",
                "environment": self.config.environment.value,
                "version": "1.0.0",
                "uptime": time.time() - self.start_time,
                "memory_usage": psutil.virtual_memory()._asdict(),
                "cpu_usage": psutil.cpu_percent(),
                "disk_usage": psutil.disk_usage('/')._asdict(),
            }
        
        @app.post("/shutdown")
        async def shutdown():
            """Endpoint para shutdown graceful."""
            if self.config.is_production:
                # Solo permitir shutdown en producción con autenticación
                raise HTTPException(status_code=403, detail="Shutdown not allowed")
            
            self.logger.info("Graceful shutdown initiated")
            asyncio.create_task(self._graceful_shutdown())
            return {"message": "Shutdown initiated"}
    
    async def _startup(self) -> Any:
        """Inicialización de la aplicación."""
        self.start_time = time.time()
        self.logger.info("Starting SEO service in production mode")
        
        # Configurar servicios
        await self._setup_redis()
        self.service_manager = ServiceManager()
        self.metrics_collector = MetricsCollector()
        
        # Configurar signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)
        
        self.logger.info("SEO service started successfully")
    
    async def _shutdown(self) -> Any:
        """Shutdown graceful de la aplicación."""
        self.logger.info("Shutting down SEO service")
        
        if self.redis_client:
            await self.redis_client.close()
        
        # Cerrar conexiones y limpiar recursos
        if hasattr(self, 'service_manager') and self.service_manager:
            await self.service_manager.cleanup()
        
        self.logger.info("SEO service shutdown complete")
    
    def _signal_handler(self, signum, frame) -> Any:
        """Maneja señales de sistema para shutdown graceful."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
        asyncio.create_task(self._graceful_shutdown())
    
    async def _graceful_shutdown(self) -> Any:
        """Shutdown graceful con timeout."""
        try:
            await asyncio.wait_for(self._shutdown(), timeout=self.prod_config.graceful_timeout)
        except asyncio.TimeoutError:
            self.logger.warning("Graceful shutdown timeout, forcing exit")
        finally:
            sys.exit(0)
    
    def run(self) -> Any:
        """Ejecuta la aplicación en modo producción."""
        self.app = self._create_app()
        
        uvicorn.run(
            self.app,
            host=self.config.server.host,
            port=self.config.server.port,
            workers=self.prod_config.max_workers,
            worker_class=self.prod_config.worker_class,
            max_requests=self.prod_config.max_requests,
            max_requests_jitter=self.prod_config.max_requests_jitter,
            ssl_keyfile=self.prod_config.ssl_keyfile,
            ssl_certfile=self.prod_config.ssl_certfile,
            access_log=True,
            log_level=self.prod_config.log_level.lower(),
        )


class CircuitBreaker:
    """Implementación de Circuit Breaker para resiliencia."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        
    """__init__ function."""
self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs) -> Any:
        """Ejecuta función con circuit breaker."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e


# Función principal para ejecutar en producción
def main():
    """Función principal para ejecutar el servicio en producción."""
    manager = ProductionManager()
    manager.run()


match __name__:
    case "__main__":
    main() 