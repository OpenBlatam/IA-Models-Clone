"""
Enhanced App V2 - Aplicación FastAPI mejorada con todas las funcionalidades avanzadas
"""

from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime

# Importar todos los sistemas avanzados
from .middleware.security import SecurityMiddleware, APIKeyMiddleware, RequestLoggingMiddleware
from .middleware.performance import PerformanceMiddleware
from .middleware.rate_limiting import RateLimitingMiddleware
from ..core.optimized_engine import get_optimized_export_engine
from ..utils.error_handler import get_error_handler
from ..nlp.enhanced_engine import get_enhanced_nlp_engine
from ..security.enhanced_security import EnhancedSecurity, SecurityLevel
from ..automation.workflow_automation import WorkflowAutomation
from ..data.advanced_data_manager import AdvancedDataManager
from ..monitoring.advanced_monitoring import AdvancedMonitoring
from ..config import settings

# Importar routers
from .routes import export_router, nlp_router, enhanced_nlp_router, system_router
from .routes.advanced_features import router as advanced_features_router

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Instancias globales de sistemas avanzados
security_manager: Optional[EnhancedSecurity] = None
automation_manager: Optional[WorkflowAutomation] = None
data_manager: Optional[AdvancedDataManager] = None
monitoring_manager: Optional[AdvancedMonitoring] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación."""
    global security_manager, automation_manager, data_manager, monitoring_manager
    
    try:
        logger.info("🚀 Iniciando sistemas avanzados...")
        
        # Inicializar sistemas avanzados
        security_manager = EnhancedSecurity(SecurityLevel.HIGH)
        automation_manager = WorkflowAutomation()
        data_manager = AdvancedDataManager()
        monitoring_manager = AdvancedMonitoring(collection_interval=30)
        
        # Inicializar todos los sistemas
        await security_manager.initialize()
        await automation_manager.initialize()
        await data_manager.initialize()
        await monitoring_manager.initialize()
        
        # Inicializar sistemas existentes
        export_engine = get_optimized_export_engine()
        await export_engine.initialize()
        
        nlp_engine = get_enhanced_nlp_engine()
        await nlp_engine.initialize()
        
        error_handler = get_error_handler()
        
        logger.info("✅ Todos los sistemas inicializados exitosamente")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Error durante la inicialización: {e}")
        raise
    finally:
        logger.info("🔄 Cerrando sistemas avanzados...")
        
        # Cerrar todos los sistemas
        if monitoring_manager:
            await monitoring_manager.shutdown()
        if data_manager:
            await data_manager.shutdown()
        if automation_manager:
            await automation_manager.shutdown()
        if security_manager:
            await security_manager.shutdown()
        
        logger.info("✅ Sistemas cerrados exitosamente")


def create_enhanced_app_v2() -> FastAPI:
    """Crear aplicación FastAPI mejorada con todas las funcionalidades."""
    
    app = FastAPI(
        title="Export IA - Sistema Avanzado",
        description="""
        ## Sistema Export IA con Funcionalidades Avanzadas
        
        ### 🚀 Características Principales:
        - **Export IA Engine**: Procesamiento optimizado de documentos
        - **NLP Avanzado**: Análisis de texto con IA y modelos transformer
        - **Sistema de Seguridad**: Autenticación, autorización y monitoreo de amenazas
        - **Automatización**: Flujos de trabajo automatizados
        - **Gestión de Datos**: Almacenamiento avanzado y búsqueda
        - **Monitoreo**: Métricas en tiempo real y alertas
        
        ### 🔧 Funcionalidades Técnicas:
        - Procesamiento paralelo y asíncrono
        - Cache inteligente con TTL
        - Optimización de memoria
        - Manejo avanzado de errores
        - Logging estructurado
        - Rate limiting y protección DDoS
        - Middleware de seguridad
        - API Key authentication
        - Monitoreo de rendimiento
        - Alertas automáticas
        
        ### 📊 Sistemas Integrados:
        - **Export IA**: Generación de documentos en múltiples formatos
        - **NLP Engine**: Análisis de texto, sentimientos, traducción
        - **Business Agents**: Agentes de negocio automatizados
        - **Security System**: Protección y autenticación
        - **Automation**: Flujos de trabajo automatizados
        - **Data Manager**: Gestión avanzada de datos
        - **Monitoring**: Monitoreo en tiempo real
        """,
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Configurar CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # En producción, especificar dominios
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Middleware de seguridad
    security_middleware = SecurityMiddleware()
    api_key_middleware = APIKeyMiddleware()
    request_logging_middleware = RequestLoggingMiddleware()
    performance_middleware = PerformanceMiddleware()
    rate_limiting_middleware = RateLimitingMiddleware()
    
    app.middleware("http")(security_middleware)
    app.middleware("http")(api_key_middleware)
    app.middleware("http")(request_logging_middleware)
    app.middleware("http")(performance_middleware)
    app.middleware("http")(rate_limiting_middleware)
    
    # Manejadores de excepciones globales
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        error_handler = get_error_handler()
        error_info = error_handler.handle_error(
            error=exc,
            category="HTTP_ERROR",
            severity="MEDIUM",
            context={"status_code": exc.status_code, "detail": exc.detail},
            request_id=getattr(request.state, "request_id", None)
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "error_id": error_info.error_id,
                "status_code": exc.status_code,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        error_handler = get_error_handler()
        error_info = error_handler.handle_error(
            error=exc,
            category="SYSTEM_ERROR",
            severity="HIGH",
            context={"endpoint": str(request.url)},
            request_id=getattr(request.state, "request_id", None)
        )
        
        logger.error(f"Error no manejado: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Error interno del servidor",
                "error_id": error_info.error_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # Middleware para agregar headers de respuesta
    @app.middleware("http")
    async def add_response_headers(request: Request, call_next):
        response = await call_next(request)
        
        # Agregar headers de seguridad
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        # Agregar headers de información
        if hasattr(request.state, "request_id"):
            response.headers["X-Request-ID"] = request.state.request_id
        
        if hasattr(request.state, "processing_time"):
            response.headers["X-Process-Time"] = f"{request.state.processing_time:.4f}s"
        
        return response
    
    # Rutas principales
    @app.get("/", include_in_schema=False)
    async def root():
        """Endpoint raíz con información del sistema."""
        return {
            "name": "Export IA - Sistema Avanzado",
            "version": "2.0.0",
            "description": "Sistema de exportación de documentos con IA y funcionalidades avanzadas",
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "features": [
                "Export IA Engine",
                "NLP Avanzado",
                "Sistema de Seguridad",
                "Automatización",
                "Gestión de Datos",
                "Monitoreo Avanzado"
            ],
            "endpoints": {
                "docs": "/docs",
                "redoc": "/redoc",
                "health": "/health",
                "system_info": "/api/v1/system/info"
            }
        }
    
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Verificación de salud del sistema."""
        try:
            # Verificar salud de todos los sistemas
            export_engine = get_optimized_export_engine()
            nlp_engine = get_enhanced_nlp_engine()
            error_handler = get_error_handler()
            
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "systems": {
                    "export_engine": await export_engine.health_check(),
                    "nlp_engine": await nlp_engine.health_check(),
                    "error_handler": error_handler.get_error_statistics()
                }
            }
            
            # Agregar sistemas avanzados si están disponibles
            if security_manager:
                health_status["systems"]["security"] = await security_manager.health_check()
            
            if automation_manager:
                health_status["systems"]["automation"] = await automation_manager.health_check()
            
            if data_manager:
                health_status["systems"]["data_manager"] = await data_manager.health_check()
            
            if monitoring_manager:
                health_status["systems"]["monitoring"] = await monitoring_manager.health_check()
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error en health check: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # Incluir routers
    app.include_router(export_router, prefix="/api/v1")
    app.include_router(nlp_router, prefix="/api/v1")
    app.include_router(enhanced_nlp_router, prefix="/api/v1")
    app.include_router(system_router, prefix="/api/v1")
    app.include_router(advanced_features_router)
    
    # Rutas adicionales del sistema
    @app.get("/api/v1/system/info", tags=["System"])
    async def get_system_info():
        """Información detallada del sistema."""
        try:
            export_engine = get_optimized_export_engine()
            nlp_engine = get_enhanced_nlp_engine()
            error_handler = get_error_handler()
            
            system_info = {
                "name": "Export IA - Sistema Avanzado",
                "version": "2.0.0",
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "performance_metrics": await export_engine.get_performance_metrics(),
                "error_statistics": error_handler.get_error_statistics(),
                "nlp_metrics": await nlp_engine.get_enhanced_metrics(),
                "features": {
                    "export_ia": True,
                    "nlp_advanced": True,
                    "security_system": security_manager is not None,
                    "automation": automation_manager is not None,
                    "data_management": data_manager is not None,
                    "monitoring": monitoring_manager is not None
                }
            }
            
            # Agregar estadísticas de sistemas avanzados
            if security_manager:
                system_info["security_stats"] = await security_manager.get_security_stats()
            
            if automation_manager:
                system_info["automation_stats"] = await automation_manager.get_automation_stats()
            
            if data_manager:
                system_info["data_stats"] = await data_manager.get_data_stats()
            
            if monitoring_manager:
                system_info["monitoring_stats"] = await monitoring_manager.get_monitoring_stats()
            
            return system_info
            
        except Exception as e:
            logger.error(f"Error al obtener información del sistema: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/system/optimize", tags=["System"])
    async def optimize_system():
        """Optimizar rendimiento del sistema."""
        try:
            export_engine = get_optimized_export_engine()
            nlp_engine = get_enhanced_nlp_engine()
            
            # Optimizar sistemas
            await export_engine.optimize_performance()
            await nlp_engine.optimize_performance()
            
            return {
                "message": "Sistema optimizado exitosamente",
                "timestamp": datetime.now().isoformat(),
                "optimizations": [
                    "Export engine performance optimized",
                    "NLP engine performance optimized",
                    "Cache cleaned and optimized",
                    "Memory usage optimized"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error al optimizar sistema: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/system/logs", tags=["System"])
    async def get_system_logs(limit: int = 100):
        """Obtener logs del sistema."""
        try:
            error_handler = get_error_handler()
            recent_errors = error_handler.get_recent_errors(limit=limit)
            
            return {
                "logs": recent_errors,
                "count": len(recent_errors),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error al obtener logs: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# Crear instancia de la aplicación
app = create_enhanced_app_v2()


if __name__ == "__main__":
    uvicorn.run(
        "enhanced_app_v2:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )




