"""
Enhanced FastAPI Application - Aplicación mejorada con todas las optimizaciones
"""

import logging
import time
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from .routes import export, tasks, health, monitoring
from .middleware.security import SecurityMiddleware, APIKeyMiddleware, RequestLoggingMiddleware
from ..core.optimized_engine import get_optimized_export_engine
from ..utils.error_handler import get_error_handler, ErrorCategory, ErrorSeverity
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manejar el ciclo de vida de la aplicación."""
    # Startup
    logger.info("Iniciando Export IA API...")
    
    try:
        # Inicializar motor optimizado
        engine = get_optimized_export_engine()
        await engine.initialize()
        
        # Inicializar manejador de errores
        error_handler = get_error_handler()
        
        logger.info("Export IA API iniciada exitosamente")
        
    except Exception as e:
        logger.error(f"Error al iniciar la aplicación: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Cerrando Export IA API...")
    
    try:
        # Cerrar motor
        engine = get_optimized_export_engine()
        await engine.shutdown()
        
        logger.info("Export IA API cerrada exitosamente")
        
    except Exception as e:
        logger.error(f"Error al cerrar la aplicación: {e}")


def create_enhanced_app() -> FastAPI:
    """Crear aplicación FastAPI mejorada."""
    
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Configurar middleware de seguridad
    security_middleware = SecurityMiddleware()
    api_key_middleware = APIKeyMiddleware()
    request_logging_middleware = RequestLoggingMiddleware()
    
    # Agregar middleware
    app.middleware("http")(security_middleware)
    app.middleware("http")(api_key_middleware)
    app.middleware("http")(request_logging_middleware)
    
    # Configurar CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # Configurar Trusted Host
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1"]
    )
    
    # Incluir rutas
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(export.router, prefix="/api/v1", tags=["export"])
    app.include_router(tasks.router, prefix="/api/v1", tags=["tasks"])
    app.include_router(monitoring.router, prefix="/api/v1", tags=["monitoring"])
    
    # Manejo global de errores
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Manejar excepciones HTTP."""
        error_handler = get_error_handler()
        
        error_info = error_handler.handle_error(
            error=exc,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.MEDIUM,
            context={
                "path": request.url.path,
                "method": request.method,
                "status_code": exc.status_code
            },
            request_id=request.headers.get("X-Request-ID")
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "error_id": error_info.error_id,
                "timestamp": error_info.timestamp.isoformat()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Manejar excepciones generales."""
        error_handler = get_error_handler()
        
        error_info = error_handler.handle_error(
            error=exc,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            context={
                "path": request.url.path,
                "method": request.method
            },
            request_id=request.headers.get("X-Request-ID")
        )
        
        logger.error(f"Error no manejado: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Error interno del servidor",
                "error_id": error_info.error_id,
                "timestamp": error_info.timestamp.isoformat()
            }
        )
    
    # Middleware de timing
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Agregar header de tiempo de procesamiento."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Middleware de request ID
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """Agregar ID único a cada request."""
        import uuid
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    
    # Endpoint de información del sistema
    @app.get("/api/v1/system/info")
    async def get_system_info():
        """Obtener información del sistema."""
        engine = get_optimized_export_engine()
        error_handler = get_error_handler()
        
        return {
            "name": settings.api_title,
            "version": settings.api_version,
            "description": settings.api_description,
            "status": "running",
            "timestamp": time.time(),
            "performance_metrics": await engine.get_performance_metrics(),
            "error_statistics": error_handler.get_error_statistics(),
            "security_stats": security_middleware.get_security_stats(),
            "api_key_stats": api_key_middleware.get_api_key_stats(),
            "request_stats": request_logging_middleware.get_request_stats()
        }
    
    # Endpoint de optimización
    @app.post("/api/v1/system/optimize")
    async def optimize_system():
        """Optimizar sistema."""
        engine = get_optimized_export_engine()
        optimization_result = await engine.optimize_performance()
        
        return {
            "message": "Sistema optimizado",
            "optimization_result": optimization_result,
            "timestamp": time.time()
        }
    
    # Endpoint de configuración
    @app.get("/api/v1/system/config")
    async def get_system_config():
        """Obtener configuración del sistema (solo en debug)."""
        if not settings.debug:
            raise HTTPException(status_code=403, detail="Configuración no disponible en producción")
        
        return {
            "debug": settings.debug,
            "database_url": settings.database_url,
            "exports_dir": settings.exports_dir,
            "max_file_size": settings.max_file_size,
            "log_level": settings.log_level,
            "max_concurrent_tasks": settings.max_concurrent_tasks,
            "task_timeout": settings.task_timeout,
            "default_quality_level": settings.default_quality_level,
            "enable_quality_validation": settings.enable_quality_validation,
            "enable_content_enhancement": settings.enable_content_enhancement
        }
    
    # Endpoint de logs recientes
    @app.get("/api/v1/system/logs")
    async def get_recent_logs(limit: int = 100):
        """Obtener logs recientes (solo en debug)."""
        if not settings.debug:
            raise HTTPException(status_code=403, detail="Logs no disponibles en producción")
        
        request_logging_middleware = None
        for middleware in app.user_middleware:
            if hasattr(middleware, 'cls') and middleware.cls == RequestLoggingMiddleware:
                request_logging_middleware = middleware
                break
        
        if request_logging_middleware:
            logs = request_logging_middleware.get_request_logs(limit)
            return {"logs": logs, "count": len(logs)}
        
        return {"logs": [], "count": 0}
    
    return app


# Crear instancia de la aplicación mejorada
app = create_enhanced_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.api.enhanced_app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )




