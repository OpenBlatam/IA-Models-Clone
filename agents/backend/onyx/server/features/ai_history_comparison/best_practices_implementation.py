"""
Best Practices Implementation for AI History Comparison System
Implementación de Mejores Prácticas para el Sistema de Comparación de Historial de IA
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone
import json
import hashlib

# FastAPI and dependencies
from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Database
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func, Index

# Caching and performance
import redis.asyncio as redis
from functools import lru_cache

# Monitoring
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# =============================================================================
# 1. CONFIGURACIÓN Y SETTINGS
# =============================================================================

class Settings:
    """Configuración centralizada con mejores prácticas"""
    
    def __init__(self):
        # Database
        self.database_url = "postgresql+asyncpg://user:pass@localhost/ai_history"
        self.database_pool_size = 20
        self.database_max_overflow = 30
        
        # Redis
        self.redis_url = "redis://localhost:6379"
        self.redis_max_connections = 20
        
        # Security
        self.secret_key = "your-secret-key-here"
        self.access_token_expire_minutes = 30
        
        # Performance
        self.max_concurrent_requests = 100
        self.request_timeout = 30
        
        # Monitoring
        self.log_level = "INFO"
        self.enable_metrics = True

settings = Settings()

# =============================================================================
# 2. LOGGING ESTRUCTURADO
# =============================================================================

class StructuredLogger:
    """Logger estructurado con mejores prácticas"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, settings.log_level))
        
        # Formatter estructurado
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def log_request(self, request_id: str, method: str, path: str, 
                   client_ip: str, processing_time: float, status_code: int):
        """Log estructurado de requests"""
        self.logger.info(
            f"Request completed",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "client_ip": client_ip,
                "processing_time": processing_time,
                "status_code": status_code,
                "event_type": "request"
            }
        )
    
    def log_llm_usage(self, model: str, provider: str, tokens: int, 
                     processing_time: float, success: bool):
        """Log estructurado de uso de LLM"""
        self.logger.info(
            f"LLM usage",
            extra={
                "model": model,
                "provider": provider,
                "tokens": tokens,
                "processing_time": processing_time,
                "success": success,
                "event_type": "llm_usage"
            }
        )

logger = StructuredLogger(__name__)

# =============================================================================
# 3. MÉTRICAS Y MONITOREO
# =============================================================================

class MetricsCollector:
    """Recolector de métricas con Prometheus"""
    
    def __init__(self):
        # Métricas HTTP
        self.request_count = Counter(
            'http_requests_total', 
            'Total HTTP requests', 
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration'
        )
        
        # Métricas LLM
        self.llm_requests = Counter(
            'llm_requests_total',
            'Total LLM requests',
            ['model', 'provider', 'status']
        )
        
        self.llm_tokens = Counter(
            'llm_tokens_total',
            'Total LLM tokens used',
            ['model', 'provider']
        )
        
        # Métricas de sistema
        self.active_connections = Gauge(
            'active_connections',
            'Active connections'
        )
        
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_type']
        )
        
        self.cache_misses = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_type']
        )
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Registrar métrica de request"""
        self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.observe(duration)
    
    def record_llm_usage(self, model: str, provider: str, tokens: int, success: bool):
        """Registrar métrica de uso de LLM"""
        status = "success" if success else "error"
        self.llm_requests.labels(model=model, provider=provider, status=status).inc()
        if success:
            self.llm_tokens.labels(model=model, provider=provider).inc(tokens)
    
    def record_cache_hit(self, cache_type: str):
        """Registrar cache hit"""
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Registrar cache miss"""
        self.cache_misses.labels(cache_type=cache_type).inc()

metrics = MetricsCollector()

# =============================================================================
# 4. CACHING MULTI-NIVEL
# =============================================================================

class MultiLevelCache:
    """Caché multi-nivel con mejores prácticas"""
    
    def __init__(self):
        self.l1_cache = {}  # In-memory cache
        self.redis_client = None
        self.cache_stats = {"hits": 0, "misses": 0}
    
    async def initialize_redis(self):
        """Inicializar cliente Redis"""
        try:
            self.redis_client = redis.from_url(settings.redis_url)
            await self.redis_client.ping()
            logger.logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.logger.warning(f"Redis cache not available: {str(e)}")
            self.redis_client = None
    
    def _generate_key(self, prefix: str, *args) -> str:
        """Generar clave de caché"""
        key_data = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, prefix: str, *args) -> Optional[Any]:
        """Obtener valor del caché"""
        key = self._generate_key(prefix, *args)
        
        # L1: In-memory cache
        if key in self.l1_cache:
            self.cache_stats["hits"] += 1
            metrics.record_cache_hit("l1")
            return self.l1_cache[key]
        
        # L2: Redis cache
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(key)
                if cached_data:
                    data = json.loads(cached_data)
                    self.l1_cache[key] = data  # Promote to L1
                    self.cache_stats["hits"] += 1
                    metrics.record_cache_hit("l2")
                    return data
            except Exception as e:
                logger.logger.warning(f"Redis get error: {str(e)}")
        
        self.cache_stats["misses"] += 1
        metrics.record_cache_miss("all")
        return None
    
    async def set(self, prefix: str, value: Any, ttl: int = 3600, *args):
        """Establecer valor en caché"""
        key = self._generate_key(prefix, *args)
        
        # L1: In-memory cache
        self.l1_cache[key] = value
        
        # L2: Redis cache
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    key, 
                    ttl, 
                    json.dumps(value, default=str)
                )
            except Exception as e:
                logger.logger.warning(f"Redis set error: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del caché"""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total if total > 0 else 0
        
        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": hit_rate,
            "l1_size": len(self.l1_cache)
        }

cache = MultiLevelCache()

# =============================================================================
# 5. CIRCUIT BREAKER
# =============================================================================

class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreaker:
    """Circuit breaker con mejores prácticas"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func: Callable, *args, **kwargs):
        """Ejecutar función con circuit breaker"""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Service temporarily unavailable"
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Manejar éxito"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Manejar fallo"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Circuit breakers para servicios externos
llm_circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30)
database_circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)

# =============================================================================
# 6. RATE LIMITING
# =============================================================================

class RateLimiter:
    """Rate limiter con mejores prácticas"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = {}
    
    async def is_allowed(self, client_ip: str) -> bool:
        """Verificar si el request está permitido"""
        now = time.time()
        
        # Limpiar requests antiguos
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if now - req_time < 60
            ]
        else:
            self.requests[client_ip] = []
        
        # Verificar límite
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False
        
        # Agregar request actual
        self.requests[client_ip].append(now)
        return True

rate_limiter = RateLimiter(requests_per_minute=100)

# =============================================================================
# 7. MIDDLEWARE DE MEJORES PRÁCTICAS
# =============================================================================

async def metrics_middleware(request: Request, call_next):
    """Middleware para métricas"""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    metrics.record_request(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
        duration=duration
    )
    
    return response

async def rate_limiting_middleware(request: Request, call_next):
    """Middleware para rate limiting"""
    client_ip = request.client.host
    
    if not await rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": "Too many requests, please try again later",
                "retry_after": 60
            }
        )
    
    response = await call_next(request)
    return response

async def request_logging_middleware(request: Request, call_next):
    """Middleware para logging de requests"""
    start_time = time.time()
    request_id = hashlib.md5(f"{time.time()}{request.client.host}".encode()).hexdigest()[:8]
    
    # Agregar request ID al estado
    request.state.request_id = request_id
    
    response = await call_next(request)
    
    processing_time = time.time() - start_time
    logger.log_request(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host,
        processing_time=processing_time,
        status_code=response.status_code
    )
    
    # Agregar headers de respuesta
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{processing_time:.3f}"
    
    return response

# =============================================================================
# 8. HEALTH CHECKS AVANZADOS
# =============================================================================

class HealthChecker:
    """Health checker con mejores prácticas"""
    
    def __init__(self):
        self.checks = {}
    
    def register_check(self, name: str, check_func: Callable):
        """Registrar check de salud"""
        self.checks[name] = check_func
    
    async def check_all(self) -> Dict[str, Any]:
        """Ejecutar todos los checks"""
        results = {}
        overall_status = "healthy"
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                result = await check_func()
                duration = time.time() - start_time
                
                results[name] = {
                    "status": "healthy",
                    "duration": duration,
                    "details": result
                }
            except Exception as e:
                results[name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": results
        }

health_checker = HealthChecker()

# Registrar checks
@health_checker.register_check("database")
async def check_database():
    """Check de base de datos"""
    # Implementar check real
    return {"connection": "ok", "response_time": "< 10ms"}

@health_checker.register_check("redis")
async def check_redis():
    """Check de Redis"""
    if cache.redis_client:
        await cache.redis_client.ping()
        return {"connection": "ok"}
    else:
        return {"connection": "not_configured"}

@health_checker.register_check("cache")
async def check_cache():
    """Check de caché"""
    return cache.get_stats()

# =============================================================================
# 9. CONFIGURACIÓN DE FASTAPI CON MEJORES PRÁCTICAS
# =============================================================================

def create_app() -> FastAPI:
    """Crear aplicación FastAPI con mejores prácticas"""
    
    app = FastAPI(
        title="AI History Comparison System",
        description="Sistema completo para análisis y comparación de historial de IA",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Agregar middleware en orden correcto
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configurar apropiadamente para producción
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Middleware personalizado
    app.middleware("http")(request_logging_middleware)
    app.middleware("http")(rate_limiting_middleware)
    app.middleware("http")(metrics_middleware)
    
    # Endpoints de sistema
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return await health_checker.check_all()
    
    @app.get("/metrics")
    async def metrics_endpoint():
        """Métricas de Prometheus"""
        if settings.enable_metrics:
            return Response(generate_latest(), media_type="text/plain")
        else:
            raise HTTPException(status_code=404, detail="Metrics disabled")
    
    @app.get("/cache/stats")
    async def cache_stats():
        """Estadísticas del caché"""
        return cache.get_stats()
    
    # Inicialización
    @app.on_event("startup")
    async def startup_event():
        """Evento de startup"""
        await cache.initialize_redis()
        logger.logger.info("🚀 Application started successfully")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Evento de shutdown"""
        if cache.redis_client:
            await cache.redis_client.close()
        logger.logger.info("🛑 Application shutdown complete")
    
    return app

# =============================================================================
# 10. EJEMPLO DE USO
# =============================================================================

async def example_usage():
    """Ejemplo de uso de las mejores prácticas"""
    
    # Crear aplicación
    app = create_app()
    
    # Ejemplo de endpoint con todas las mejores prácticas
    @app.post("/api/v1/analyze")
    async def analyze_content(request: Request, content_data: dict):
        """Endpoint de ejemplo con mejores prácticas"""
        
        # Rate limiting (ya manejado por middleware)
        
        # Circuit breaker para LLM
        async def call_llm_service():
            # Simular llamada a LLM
            await asyncio.sleep(0.1)
            return {"analysis": "result", "tokens": 100}
        
        try:
            result = await llm_circuit_breaker.call(call_llm_service)
            
            # Registrar métricas
            metrics.record_llm_usage(
                model="gpt-4",
                provider="openai",
                tokens=result["tokens"],
                success=True
            )
            
            # Log estructurado
            logger.log_llm_usage(
                model="gpt-4",
                provider="openai",
                tokens=result["tokens"],
                processing_time=0.1,
                success=True
            )
            
            return {"success": True, "data": result}
            
        except Exception as e:
            # Registrar métricas de error
            metrics.record_llm_usage(
                model="gpt-4",
                provider="openai",
                tokens=0,
                success=False
            )
            
            raise HTTPException(status_code=500, detail=str(e))
    
    return app

if __name__ == "__main__":
    # Ejecutar ejemplo
    import uvicorn
    app = asyncio.run(example_usage())
    uvicorn.run(app, host="0.0.0.0", port=8000)







