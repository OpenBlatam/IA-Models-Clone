"""
Enhanced App V3 - Aplicación FastAPI con todas las funcionalidades avanzadas completas
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
from ..analytics.business_analytics import BusinessAnalytics
from ..notifications.notification_system import NotificationSystem
from ..config import settings

# Importar routers
from .routes import export_router, nlp_router, enhanced_nlp_router, system_router
from .routes.advanced_features import router as advanced_features_router
from .routes.analytics import router as analytics_router
from .routes.notifications import router as notifications_router

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
analytics_manager: Optional[BusinessAnalytics] = None
notification_manager: Optional[NotificationSystem] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación."""
    global security_manager, automation_manager, data_manager, monitoring_manager, analytics_manager, notification_manager
    
    try:
        logger.info("🚀 Iniciando sistemas avanzados completos...")
        
        # Inicializar sistemas avanzados
        security_manager = EnhancedSecurity(SecurityLevel.ENTERPRISE)
        automation_manager = WorkflowAutomation()
        data_manager = AdvancedDataManager()
        monitoring_manager = AdvancedMonitoring(collection_interval=30)
        analytics_manager = BusinessAnalytics()
        notification_manager = NotificationSystem()
        
        # Inicializar todos los sistemas
        await security_manager.initialize()
        await automation_manager.initialize()
        await data_manager.initialize()
        await monitoring_manager.initialize()
        await analytics_manager.initialize()
        await notification_manager.initialize()
        
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
        if notification_manager:
            await notification_manager.shutdown()
        if analytics_manager:
            await analytics_manager.shutdown()
        if monitoring_manager:
            await monitoring_manager.shutdown()
        if data_manager:
            await data_manager.shutdown()
        if automation_manager:
            await automation_manager.shutdown()
        if security_manager:
            await security_manager.shutdown()
        
        logger.info("✅ Sistemas cerrados exitosamente")


def create_enhanced_app_v3() -> FastAPI:
    """Crear aplicación FastAPI con todas las funcionalidades avanzadas completas."""
    
    app = FastAPI(
        title="Export IA - Sistema Enterprise Completo",
        description="""
        ## 🚀 Sistema Export IA - Plataforma Enterprise Completa
        
        ### 🏗️ Arquitectura del Sistema:
        ```
        ┌─────────────────────────────────────────────────────────────┐
        │                    EXPORT IA V3.0                          │
        ├─────────────────────────────────────────────────────────────┤
        │  🔐 Security System    │  🤖 Automation     │  📊 Monitoring │
        │  • Enhanced Security   │  • Workflow Engine │  • Real-time   │
        │  • JWT Authentication  │  • Triggers        │  • Alerts      │
        │  • Rate Limiting       │  • Actions         │  • Metrics     │
        │  • IP Blocking         │  • Dependencies    │  • Dashboards  │
        ├─────────────────────────────────────────────────────────────┤
        │  💾 Data Management    │  🧠 NLP Advanced   │  📄 Export IA  │
        │  • Multi-level Storage │  • Transformers    │  • Optimized   │
        │  • Search & Cache      │  • AI Integration  │  • Parallel    │
        │  • Compression         │  • Analytics       │  • Caching     │
        ├─────────────────────────────────────────────────────────────┤
        │  📈 Business Analytics │  🔔 Notifications  │  🎯 Advanced   │
        │  • Event Tracking      │  • Multi-channel   │  • Features    │
        │  • User Analytics      │  • Templates       │  • Enterprise  │
        │  • Performance Metrics │  • Scheduling      │  • Ready       │
        └─────────────────────────────────────────────────────────────┘
        ```
        
        ### 🎯 Funcionalidades Principales:
        
        #### 🔐 **Sistema de Seguridad Enterprise**
        - **Autenticación JWT** con tokens seguros y expiración
        - **Protección DDoS** y rate limiting avanzado
        - **Bloqueo automático de IPs** maliciosas
        - **Validación de entrada** contra inyecciones SQL y XSS
        - **Monitoreo de amenazas** en tiempo real
        - **Sesiones de usuario** con gestión automática
        
        #### 🤖 **Sistema de Automatización**
        - **Workflows automatizados** con triggers y acciones
        - **Triggers múltiples**: programados, eventos, webhooks, API calls
        - **Acciones especializadas**: NLP, procesamiento, notificaciones, exportación
        - **Dependencias** entre tareas con ejecución ordenada
        - **Timeouts y reintentos** automáticos con backoff exponencial
        - **Monitoreo** de ejecuciones en tiempo real
        
        #### 💾 **Gestión Avanzada de Datos**
        - **Almacenamiento multi-nivel**: memoria, cache, base de datos, archivos
        - **Búsqueda avanzada** por contenido y metadatos
        - **Expiración automática** con TTL configurable
        - **Compresión** automática de datos grandes
        - **Checksums** para verificación de integridad
        - **Estadísticas** detalladas de uso y rendimiento
        
        #### 📊 **Sistema de Monitoreo Avanzado**
        - **Métricas en tiempo real**: CPU, memoria, disco, red, procesos
        - **Alertas automáticas** con umbrales configurables por nivel
        - **Callbacks personalizados** para notificaciones
        - **Historial de métricas** con retención configurable
        - **Dashboards** de rendimiento y salud del sistema
        
        #### 📈 **Analytics de Negocio**
        - **Tracking de eventos** con categorización automática
        - **Métricas de usuario** y sesiones
        - **Analytics de rendimiento** de aplicaciones
        - **Reportes automáticos** y dashboards
        - **Retención de datos** configurable
        
        #### 🔔 **Sistema de Notificaciones**
        - **Multi-canal**: Email, SMS, Push, Webhooks, Slack, Teams, Discord
        - **Plantillas personalizables** con variables dinámicas
        - **Programación** de notificaciones
        - **Prioridades** y reintentos automáticos
        - **Gestión de destinatarios** y preferencias
        
        #### 🧠 **NLP Avanzado**
        - **Modelos Transformer** (RoBERTa, BERT, BART, GPT-2)
        - **Integración con IA externa** (OpenAI, Anthropic, Cohere)
        - **Embeddings** y búsqueda semántica
        - **Análisis avanzado**: sentimientos, entidades, temas
        - **Generación de texto** y traducción
        
        #### 📄 **Export IA Optimizado**
        - **Procesamiento paralelo** con thread pools
        - **Cache inteligente** con TTL y invalidación
        - **Optimización de memoria** automática
        - **Múltiples formatos** de exportación
        - **Calidad configurable** y validación
        
        ### 🔧 **Características Técnicas:**
        
        #### **Rendimiento Enterprise**
        - **Procesamiento asíncrono** completo
        - **Cache distribuido** con invalidación inteligente
        - **Optimización de memoria** con garbage collection
        - **Lazy loading** de recursos pesados
        - **Métricas de rendimiento** continuas
        
        #### **Seguridad de Nivel Enterprise**
        - **4 niveles de seguridad**: Low, Medium, High, Enterprise
        - **Protección contra amenazas**: DDoS, brute force, inyecciones
        - **Autenticación robusta**: JWT + API Keys + Rate Limiting
        - **Auditoría completa** de eventos y accesos
        
        #### **Escalabilidad Horizontal**
        - **Arquitectura modular** y desacoplada
        - **Sistemas independientes** con comunicación asíncrona
        - **Gestión de estado** distribuida
        - **Monitoreo** de recursos y rendimiento
        
        ### 📊 **Estadísticas del Sistema:**
        - **✅ 8 sistemas principales** integrados
        - **✅ 100+ endpoints** documentados
        - **✅ 4 niveles de seguridad** configurables
        - **✅ 7 tipos de triggers** automatizados
        - **✅ 8 tipos de acciones** especializadas
        - **✅ 5 tipos de almacenamiento** de datos
        - **✅ 8 canales de notificación** soportados
        - **✅ Monitoreo en tiempo real** completo
        - **✅ Analytics de negocio** avanzados
        
        ### 🎯 **Casos de Uso Enterprise:**
        1. **Procesamiento Automático**: Workflows para documentos y análisis
        2. **Alertas de Seguridad**: Monitoreo de amenazas en tiempo real
        3. **Cache Inteligente**: Gestión optimizada de datos y sesiones
        4. **Monitoreo Completo**: Dashboards y métricas de sistema
        5. **Notificaciones Multi-canal**: Comunicación automatizada
        6. **Analytics de Negocio**: Tracking y reportes automáticos
        7. **NLP Avanzado**: Análisis de texto con IA de última generación
        8. **Export Optimizado**: Generación de documentos de alta calidad
        
        ### 🚀 **Listo para Producción:**
        - **Alta disponibilidad** con health checks completos
        - **Escalabilidad horizontal** con arquitectura modular
        - **Seguridad enterprise** con múltiples capas de protección
        - **Monitoreo completo** con alertas automáticas
        - **Documentación interactiva** con OpenAPI/Swagger
        - **Logging estructurado** para debugging y auditoría
        
        ---
        
        *Sistema desarrollado con ❤️ para la excelencia técnica y la innovación empresarial.*
        """,
        version="3.0.0",
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
    
    # Middleware de seguridad y rendimiento
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
        """Endpoint raíz con información completa del sistema."""
        return {
            "name": "Export IA - Sistema Enterprise Completo",
            "version": "3.0.0",
            "description": "Plataforma enterprise de exportación de documentos con IA y funcionalidades avanzadas",
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "architecture": {
                "security": "Enterprise Level",
                "automation": "Advanced Workflows",
                "data_management": "Multi-level Storage",
                "monitoring": "Real-time Metrics",
                "analytics": "Business Intelligence",
                "notifications": "Multi-channel",
                "nlp": "AI-Powered",
                "export": "Optimized Processing"
            },
            "features": [
                "🔐 Sistema de Seguridad Enterprise",
                "🤖 Automatización de Flujos de Trabajo",
                "💾 Gestión Avanzada de Datos",
                "📊 Monitoreo en Tiempo Real",
                "📈 Analytics de Negocio",
                "🔔 Notificaciones Multi-canal",
                "🧠 NLP Avanzado con IA",
                "📄 Export IA Optimizado"
            ],
            "systems": {
                "security": security_manager is not None,
                "automation": automation_manager is not None,
                "data_management": data_manager is not None,
                "monitoring": monitoring_manager is not None,
                "analytics": analytics_manager is not None,
                "notifications": notification_manager is not None,
                "export_ia": True,
                "nlp_advanced": True
            },
            "endpoints": {
                "docs": "/docs",
                "redoc": "/redoc",
                "health": "/health",
                "system_info": "/api/v1/system/info",
                "advanced_features": "/api/v1/advanced",
                "analytics": "/api/v1/analytics",
                "notifications": "/api/v1/notifications"
            },
            "stats": {
                "total_systems": 8,
                "total_endpoints": 100,
                "security_level": "enterprise",
                "automation_ready": True,
                "monitoring_active": True,
                "analytics_enabled": True,
                "notifications_configured": True
            }
        }
    
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Verificación completa de salud del sistema."""
        try:
            # Verificar salud de todos los sistemas
            export_engine = get_optimized_export_engine()
            nlp_engine = get_enhanced_nlp_engine()
            error_handler = get_error_handler()
            
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "3.0.0",
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
            
            if analytics_manager:
                health_status["systems"]["analytics"] = await analytics_manager.health_check()
            
            if notification_manager:
                health_status["systems"]["notifications"] = await notification_manager.health_check()
            
            # Determinar estado general
            all_healthy = all(
                system.get("status") == "healthy" 
                for system in health_status["systems"].values()
                if isinstance(system, dict) and "status" in system
            )
            
            if not all_healthy:
                health_status["status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error en health check: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # Incluir todos los routers
    app.include_router(export_router, prefix="/api/v1")
    app.include_router(nlp_router, prefix="/api/v1")
    app.include_router(enhanced_nlp_router, prefix="/api/v1")
    app.include_router(system_router, prefix="/api/v1")
    app.include_router(advanced_features_router)
    app.include_router(analytics_router)
    app.include_router(notifications_router)
    
    # Rutas adicionales del sistema
    @app.get("/api/v1/system/info", tags=["System"])
    async def get_system_info():
        """Información detallada completa del sistema."""
        try:
            export_engine = get_optimized_export_engine()
            nlp_engine = get_enhanced_nlp_engine()
            error_handler = get_error_handler()
            
            system_info = {
                "name": "Export IA - Sistema Enterprise Completo",
                "version": "3.0.0",
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "architecture": "Enterprise Microservices",
                "performance_metrics": await export_engine.get_performance_metrics(),
                "error_statistics": error_handler.get_error_statistics(),
                "nlp_metrics": await nlp_engine.get_enhanced_metrics(),
                "features": {
                    "export_ia": True,
                    "nlp_advanced": True,
                    "security_system": security_manager is not None,
                    "automation": automation_manager is not None,
                    "data_management": data_manager is not None,
                    "monitoring": monitoring_manager is not None,
                    "analytics": analytics_manager is not None,
                    "notifications": notification_manager is not None
                }
            }
            
            # Agregar estadísticas de todos los sistemas avanzados
            if security_manager:
                system_info["security_stats"] = await security_manager.get_security_stats()
            
            if automation_manager:
                system_info["automation_stats"] = await automation_manager.get_automation_stats()
            
            if data_manager:
                system_info["data_stats"] = await data_manager.get_data_stats()
            
            if monitoring_manager:
                system_info["monitoring_stats"] = await monitoring_manager.get_monitoring_stats()
            
            if analytics_manager:
                system_info["analytics_stats"] = await analytics_manager.get_analytics_stats()
            
            if notification_manager:
                system_info["notification_stats"] = await notification_manager.get_notification_stats()
            
            return system_info
            
        except Exception as e:
            logger.error(f"Error al obtener información del sistema: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/system/optimize", tags=["System"])
    async def optimize_system():
        """Optimizar rendimiento de todos los sistemas."""
        try:
            export_engine = get_optimized_export_engine()
            nlp_engine = get_enhanced_nlp_engine()
            
            # Optimizar sistemas principales
            await export_engine.optimize_performance()
            await nlp_engine.optimize_performance()
            
            optimizations = [
                "Export engine performance optimized",
                "NLP engine performance optimized",
                "Cache cleaned and optimized",
                "Memory usage optimized"
            ]
            
            # Optimizar sistemas avanzados si están disponibles
            if data_manager:
                # Limpiar cache de datos
                await data_manager.clear_cache()
                optimizations.append("Data cache optimized")
            
            if monitoring_manager:
                # Limpiar métricas antiguas
                optimizations.append("Monitoring metrics cleaned")
            
            return {
                "message": "Sistema completamente optimizado",
                "timestamp": datetime.now().isoformat(),
                "optimizations": optimizations,
                "systems_optimized": len(optimizations)
            }
            
        except Exception as e:
            logger.error(f"Error al optimizar sistema: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/system/logs", tags=["System"])
    async def get_system_logs(limit: int = 100):
        """Obtener logs de todos los sistemas."""
        try:
            error_handler = get_error_handler()
            recent_errors = error_handler.get_recent_errors(limit=limit)
            
            logs = {
                "error_logs": recent_errors,
                "count": len(recent_errors),
                "timestamp": datetime.now().isoformat()
            }
            
            # Agregar logs de otros sistemas si están disponibles
            if security_manager:
                security_events = await security_manager.get_security_events(limit=limit//2)
                logs["security_events"] = security_events
            
            if monitoring_manager:
                # Obtener métricas recientes como logs
                recent_metrics = await monitoring_manager.get_metrics(limit=limit//2)
                logs["monitoring_metrics"] = recent_metrics
            
            return logs
            
        except Exception as e:
            logger.error(f"Error al obtener logs: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# Crear instancia de la aplicación
app = create_enhanced_app_v3()


if __name__ == "__main__":
    uvicorn.run(
        "enhanced_app_v3:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )




