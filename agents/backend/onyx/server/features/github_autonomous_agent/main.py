"""
GitHub Autonomous Agent - Main Entry Point
==========================================

Punto de entrada principal para el agente autónomo de GitHub.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import (
    agent_routes, github_routes, task_routes, llm_routes,
    websocket_routes, stats_routes, batch_routes,
    audit_routes, notification_routes, monitoring_routes,
    auth_routes, plugin_routes, config_routes,
    queue_routes, batch_processor_routes, analytics_routes, search_routes,
    validation_routes
)
from api.middleware import (
    LoggingMiddleware,
    ErrorHandlingMiddleware,
    SecurityHeadersMiddleware,
    RateLimitMiddleware
)
from config.settings import settings
from config.logging_config import setup_logging, get_logger
from config.di_setup import setup_dependencies, get_service
from config.validation import validate_configuration, print_config_summary

# Setup logging primero
setup_logging()
logger = get_logger(__name__)

# Validar configuración antes de continuar
try:
    validate_configuration()
    print_config_summary()
except Exception as e:
    logger.error(f"Error de validación de configuración: {e}")
    if not settings.DEBUG:
        raise  # En producción, fallar si hay errores críticos

# Setup dependency injection early
setup_dependencies()


def create_app() -> FastAPI:
    """Crear aplicación FastAPI."""
    from api.openapi_custom import custom_openapi
    
    app = FastAPI(
        title="GitHub Autonomous Agent API",
        description="Agente autónomo para ejecutar tareas en repositorios de GitHub",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Custom OpenAPI schema
    app.openapi = lambda: custom_openapi(app)

    # Middleware de CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Middleware personalizado (orden inverso: último agregado se ejecuta primero)
    # Rate limiting por endpoint (más externo)
    try:
        from api.middleware.rate_limit_per_endpoint import EndpointRateLimitMiddleware, get_default_endpoint_limits
        endpoint_limits = get_default_endpoint_limits()
        app.add_middleware(EndpointRateLimitMiddleware, endpoint_limits=endpoint_limits)
        logger.info("Endpoint rate limiting habilitado")
    except Exception as e:
        logger.warning(f"Endpoint rate limiting no disponible: {e}")
    
    # Rate limiting global
    try:
        rate_limit_service = get_service("rate_limit_service")
        app.add_middleware(RateLimitMiddleware, rate_limit_service=rate_limit_service)
    except Exception:
        logger.warning("Rate limit service no disponible, usando rate limiting básico")
        app.add_middleware(RateLimitMiddleware)
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Error handling
    app.add_middleware(ErrorHandlingMiddleware)
    
    # LLM Middleware (si está disponible)
    try:
        from api.middleware import (
            LLMValidationMiddleware,
            LLMRateLimitMiddleware,
            LLMLoggingMiddleware
        )
        app.add_middleware(LLMValidationMiddleware)
        app.add_middleware(LLMRateLimitMiddleware)
        app.add_middleware(LLMLoggingMiddleware)
        logger.info("✅ Middleware LLM registrado")
    except ImportError:
        logger.debug("Middleware LLM no disponible, usando middleware estándar")
    
    # Logging (más interno, se ejecuta último)
    app.add_middleware(LoggingMiddleware)

    app.include_router(agent_routes.router, prefix="/api/v1/agent", tags=["agent"])
    app.include_router(github_routes.router, prefix="/api/v1/github", tags=["github"])
    app.include_router(task_routes.router, prefix="/api/v1/tasks", tags=["tasks"])
    app.include_router(batch_routes.router, prefix="/api/v1/batch", tags=["batch"])
    app.include_router(llm_routes.router, prefix="/api/v1/llm", tags=["llm"])
    app.include_router(stats_routes.router, prefix="/api/v1/stats", tags=["stats"])
    app.include_router(audit_routes.router, prefix="/api/v1/audit", tags=["audit"])
    app.include_router(notification_routes.router, prefix="/api/v1/notifications", tags=["notifications"])
    app.include_router(monitoring_routes.router, prefix="/api/v1/monitoring", tags=["monitoring"])
    app.include_router(auth_routes.router, prefix="/api/v1/auth", tags=["authentication"])
    app.include_router(plugin_routes.router, prefix="/api/v1/plugins", tags=["plugins"])
    app.include_router(config_routes.router, prefix="/api/v1/config", tags=["configuration"])
    app.include_router(queue_routes.router, prefix="/api/v1/queue", tags=["queue"])
    app.include_router(batch_processor_routes.router, prefix="/api/v1/batch-processor", tags=["batch-processor"])
    app.include_router(analytics_routes.router, prefix="/api/v1/analytics", tags=["analytics"])
    app.include_router(search_routes.router, prefix="/api/v1/search", tags=["search"])
    app.include_router(validation_routes.router, prefix="/api/v1/validation", tags=["validation"])
    app.include_router(websocket_routes.router, tags=["websocket"])
    
    # Incluir rutas LLM adicionales
    try:
        from api.routes import llm_health, llm_models, llm_analytics, llm_optimization
        app.include_router(llm_health.router, prefix="/api/v1", tags=["llm-health"])
        app.include_router(llm_models.router, prefix="/api/v1", tags=["llm-models"])
        app.include_router(llm_analytics.router, prefix="/api/v1", tags=["llm-analytics"])
        app.include_router(llm_optimization.router, prefix="/api/v1", tags=["llm-optimization"])
        logger.info("✅ Rutas LLM adicionales registradas")
    except ImportError as e:
        logger.warning(f"Algunas rutas LLM no disponibles: {e}")

    @app.on_event("startup")
    async def startup_event():
        """Inicializar servicios al arrancar."""
        logger.info("🚀 Iniciando GitHub Autonomous Agent...")
        
        # Initialize database
        storage = get_service("storage")
        await storage.init_db()
        logger.info("✅ Base de datos inicializada")
        
        # Start worker manager
        worker_manager = get_service("worker_manager")
        await worker_manager.start()
        app.state.worker_manager = worker_manager
        logger.info("✅ Worker manager iniciado")
        
        # Start monitoring service
        try:
            monitoring_service = get_service("monitoring_service")
            await monitoring_service.start_monitoring(interval_seconds=60)
            logger.info("✅ Monitoring service iniciado")
        except Exception as e:
            logger.warning(f"Monitoring service no disponible: {e}")
        
        # Start cache warming
        try:
            cache_warming = get_service("cache_warming_service")
            await cache_warming.start_auto_warming(interval_seconds=3600)
            logger.info("✅ Cache warming iniciado")
        except Exception as e:
            logger.warning(f"Cache warming no disponible: {e}")
        
        # Start scheduler
        try:
            scheduler_service = get_service("scheduler_service")
            await scheduler_service.start_scheduler(check_interval=60.0)
            logger.info("✅ Scheduler iniciado")
        except Exception as e:
            logger.warning(f"Scheduler no disponible: {e}")
        
        logger.info("✅ Servicios iniciados correctamente")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Limpiar recursos al cerrar."""
        logger.info("🛑 Deteniendo servicios...")
        if hasattr(app.state, "worker_manager"):
            await app.state.worker_manager.stop()
        
        # Stop monitoring service
        try:
            monitoring_service = get_service("monitoring_service")
            await monitoring_service.stop_monitoring()
        except Exception:
            pass
        
        # Stop cache warming
        try:
            cache_warming = get_service("cache_warming_service")
            await cache_warming.stop_auto_warming()
        except Exception:
            pass
        
        logger.info("✅ Servicios detenidos correctamente")

    @app.get("/")
    async def root():
        """Endpoint raíz."""
        return {
            "name": "GitHub Autonomous Agent API",
            "version": "1.0.0",
            "status": "running"
        }

    @app.get("/health")
    async def health():
        """Health check endpoint con verificaciones robustas."""
        from api.response_models import HealthResponse
        from config.di_setup import get_service
        from core.exceptions import StorageError, GitHubClientError
        
        services_status = {}
        health_details = {}
        overall_healthy = True
        
        # Verificar storage con prueba real
        try:
            storage = get_service("storage")
            # Intentar una operación simple para verificar conectividad
            await storage.init_db()
            services_status["storage"] = True
            health_details["storage"] = {"status": "ok", "message": "Database connection healthy"}
        except StorageError as e:
            services_status["storage"] = False
            health_details["storage"] = {"status": "error", "message": str(e)}
            overall_healthy = False
        except Exception as e:
            logger.warning(f"Storage health check failed: {e}")
            services_status["storage"] = False
            health_details["storage"] = {"status": "error", "message": "Storage service unavailable"}
            overall_healthy = False
        
        # Verificar GitHub client
        try:
            github_client = get_service("github_client")
            # Verificar que el token esté configurado
            if github_client.token:
                services_status["github_client"] = True
                health_details["github_client"] = {"status": "ok", "message": "GitHub client initialized"}
            else:
                services_status["github_client"] = False
                health_details["github_client"] = {"status": "warning", "message": "GitHub token not configured"}
        except GitHubClientError as e:
            services_status["github_client"] = False
            health_details["github_client"] = {"status": "error", "message": str(e)}
            overall_healthy = False
        except Exception as e:
            logger.warning(f"GitHub client health check failed: {e}")
            services_status["github_client"] = False
            health_details["github_client"] = {"status": "error", "message": "GitHub client unavailable"}
            overall_healthy = False
        
        # Verificar worker manager
        try:
            if hasattr(app.state, "worker_manager"):
                worker_manager = app.state.worker_manager
                services_status["worker_manager"] = worker_manager.is_running
                health_details["worker_manager"] = {
                    "status": "ok" if worker_manager.is_running else "stopped",
                    "message": "Running" if worker_manager.is_running else "Stopped"
                }
            else:
                services_status["worker_manager"] = False
                health_details["worker_manager"] = {"status": "error", "message": "Worker manager not initialized"}
                overall_healthy = False
        except Exception as e:
            logger.warning(f"Worker manager health check failed: {e}")
            services_status["worker_manager"] = False
            health_details["worker_manager"] = {"status": "error", "message": "Worker manager unavailable"}
            overall_healthy = False
        
        # Verificar servicios opcionales
        try:
            cache_service = get_service("cache_service")
            cache_stats = cache_service.get_stats()
            services_status["cache_service"] = True
            health_details["cache_service"] = {
                "status": "ok",
                "message": f"Cache size: {cache_stats['size']}/{cache_stats['max_size']}, Hit rate: {cache_stats['hit_rate']}%"
            }
        except Exception:
            services_status["cache_service"] = False
            health_details["cache_service"] = {"status": "warning", "message": "Cache service not available"}
        
        try:
            metrics_service = get_service("metrics_service")
            metrics = metrics_service.get_metrics()
            services_status["metrics_service"] = True
            health_details["metrics_service"] = {
                "status": "ok",
                "message": "Metrics service active"
            }
        except Exception:
            services_status["metrics_service"] = False
            health_details["metrics_service"] = {"status": "warning", "message": "Metrics service not available"}
        
        try:
            llm_service = get_service("llm_service")
            if llm_service:
                services_status["llm_service"] = True
                health_details["llm_service"] = {
                    "status": "ok",
                    "message": f"LLM service active with {len(llm_service.default_models)} models"
                }
            else:
                services_status["llm_service"] = False
                health_details["llm_service"] = {"status": "disabled", "message": "LLM service disabled"}
        except Exception:
            services_status["llm_service"] = False
            health_details["llm_service"] = {"status": "warning", "message": "LLM service not available"}
        
        # Verificar Redis (si está configurado)
        try:
            import redis
            from config.settings import settings
            if settings.REDIS_URL:
                redis_client = redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
                redis_client.ping()
                services_status["redis"] = True
                health_details["redis"] = {"status": "ok", "message": "Redis connection healthy"}
                redis_client.close()
        except ImportError:
            services_status["redis"] = None
            health_details["redis"] = {"status": "not_checked", "message": "Redis client not installed"}
        except Exception as e:
            services_status["redis"] = False
            health_details["redis"] = {"status": "error", "message": f"Redis connection failed: {str(e)[:50]}"}
        
        status = "healthy" if overall_healthy else "degraded"
        
        return HealthResponse(
            status=status,
            version="1.0.0",
            services=services_status,
            details=health_details
        ).dict()

    return app


def main():
    """Función principal."""
    app = create_app()
    
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level="info",
        reload=settings.DEBUG
    )


if __name__ == "__main__":
    main()

