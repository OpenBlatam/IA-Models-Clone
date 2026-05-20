"""
Export IA V5.0 - ULTIMATE ENTERPRISE SYSTEM
Aplicación FastAPI con todas las funcionalidades avanzadas
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List
from datetime import datetime
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import json

# Importar todos los sistemas
from .middleware.security import SecurityMiddleware, APIKeyMiddleware, RequestLoggingMiddleware
from .middleware.performance import PerformanceMiddleware
from .middleware.rate_limiting import RateLimitingMiddleware
from .routes.export import router as export_router
from .routes.nlp import router as nlp_router
from .routes.enhanced_nlp import router as enhanced_nlp_router
from .routes.ai_engine import router as ai_router
from .routes.blockchain import router as blockchain_router
from .routes.web3 import router as web3_router
from .routes.analytics import router as analytics_router
from .routes.notifications import router as notifications_router
from .routes.system import router as system_router

# Importar motores principales
from ..core.optimized_engine import OptimizedExportEngine
from ..nlp.enhanced_engine import EnhancedNLPEngine
from ..ai.machine_learning_engine import MachineLearningEngine
from ..ai.deep_learning_engine import DeepLearningEngine
from ..ai.computer_vision_engine import ComputerVisionEngine
from ..blockchain.blockchain_engine import BlockchainEngine
from ..web3.web3_engine import Web3Engine
from ..security.enhanced_security import EnhancedSecurityManager
from ..automation.workflow_automation import WorkflowAutomationSystem
from ..data.advanced_data_manager import AdvancedDataManager
from ..monitoring.advanced_monitoring import AdvancedMonitoringSystem
from ..analytics.business_analytics import BusinessAnalyticsSystem
from ..notifications.notification_system import NotificationSystem

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('export_ia_v5.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Instancias globales de los motores
engines = {}
security_manager = None
automation_system = None
data_manager = None
monitoring_system = None
analytics_system = None
notification_system = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionar el ciclo de vida de la aplicación."""
    global engines, security_manager, automation_system, data_manager, monitoring_system, analytics_system, notification_system
    
    logger.info("🚀 Iniciando Export IA V5.0 - ULTIMATE ENTERPRISE SYSTEM")
    
    try:
        # Inicializar sistemas de seguridad
        logger.info("🔐 Inicializando sistema de seguridad...")
        security_manager = EnhancedSecurityManager()
        await security_manager.initialize()
        
        # Inicializar sistema de monitoreo
        logger.info("📊 Inicializando sistema de monitoreo...")
        monitoring_system = AdvancedMonitoringSystem()
        await monitoring_system.initialize()
        
        # Inicializar sistema de datos
        logger.info("💾 Inicializando sistema de datos...")
        data_manager = AdvancedDataManager()
        await data_manager.initialize()
        
        # Inicializar sistema de automatización
        logger.info("⚙️ Inicializando sistema de automatización...")
        automation_system = WorkflowAutomationSystem()
        await automation_system.initialize()
        
        # Inicializar sistema de analytics
        logger.info("📈 Inicializando sistema de analytics...")
        analytics_system = BusinessAnalyticsSystem()
        await analytics_system.initialize()
        
        # Inicializar sistema de notificaciones
        logger.info("🔔 Inicializando sistema de notificaciones...")
        notification_system = NotificationSystem()
        await notification_system.initialize()
        
        # Inicializar motores principales
        logger.info("🔧 Inicializando motores principales...")
        
        # Motor de Export IA optimizado
        engines['export'] = OptimizedExportEngine()
        await engines['export'].initialize()
        
        # Motor de NLP mejorado
        engines['nlp'] = EnhancedNLPEngine()
        await engines['nlp'].initialize()
        
        # Motores de IA
        engines['ml'] = MachineLearningEngine()
        await engines['ml'].initialize()
        
        engines['dl'] = DeepLearningEngine()
        await engines['dl'].initialize()
        
        engines['cv'] = ComputerVisionEngine()
        await engines['cv'].initialize()
        
        # Motor de Blockchain
        engines['blockchain'] = BlockchainEngine()
        await engines['blockchain'].initialize()
        
        # Motor de Web3
        engines['web3'] = Web3Engine()
        await engines['web3'].initialize()
        
        logger.info("✅ Todos los sistemas inicializados exitosamente")
        logger.info("🌟 Export IA V5.0 - ULTIMATE ENTERPRISE SYSTEM está listo")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Error durante la inicialización: {e}")
        raise
    
    finally:
        # Cerrar todos los sistemas
        logger.info("🔄 Cerrando sistemas...")
        
        try:
            if notification_system:
                await notification_system.shutdown()
            
            if analytics_system:
                await analytics_system.shutdown()
            
            if automation_system:
                await automation_system.shutdown()
            
            if data_manager:
                await data_manager.shutdown()
            
            if monitoring_system:
                await monitoring_system.shutdown()
            
            if security_manager:
                await security_manager.shutdown()
            
            # Cerrar motores
            for name, engine in engines.items():
                try:
                    if hasattr(engine, 'shutdown'):
                        await engine.shutdown()
                    logger.info(f"✅ Motor {name} cerrado")
                except Exception as e:
                    logger.error(f"❌ Error al cerrar motor {name}: {e}")
            
            logger.info("✅ Todos los sistemas cerrados")
            
        except Exception as e:
            logger.error(f"❌ Error durante el cierre: {e}")


def create_enhanced_app() -> FastAPI:
    """Crear aplicación FastAPI mejorada."""
    
    app = FastAPI(
        title="Export IA V5.0 - ULTIMATE ENTERPRISE SYSTEM",
        description="""
        ## 🚀 Sistema Export IA V5.0 - ULTIMATE ENTERPRISE
        
        **El sistema más avanzado y completo para exportación de datos, procesamiento de lenguaje natural, inteligencia artificial, blockchain y Web3.**
        
        ### 🌟 Características Principales:
        
        #### 🔧 **Motores Principales:**
        - **Export IA Optimizado**: Motor de exportación de datos de alto rendimiento
        - **NLP Mejorado**: Procesamiento avanzado de lenguaje natural con IA
        - **Machine Learning**: Entrenamiento y predicción con modelos ML
        - **Deep Learning**: Redes neuronales avanzadas con GPU
        - **Computer Vision**: Análisis y procesamiento de imágenes
        - **Blockchain**: Sistema blockchain completo con contratos inteligentes
        - **Web3**: Motor Web3 y DeFi para criptomonedas
        
        #### 🛡️ **Seguridad Avanzada:**
        - Autenticación JWT
        - Protección DDoS
        - Bloqueo de IPs
        - Validación de entrada
        - Monitoreo de amenazas
        
        #### ⚙️ **Automatización:**
        - Flujos de trabajo automatizados
        - Triggers programados y por eventos
        - Dependencias de tareas
        - Reintentos automáticos
        
        #### 💾 **Gestión de Datos:**
        - Almacenamiento multi-nivel
        - Búsqueda avanzada
        - Compresión de datos
        - Verificación de integridad
        
        #### 📊 **Monitoreo y Analytics:**
        - Métricas en tiempo real
        - Alertas automáticas
        - Analytics de negocio
        - Dashboards interactivos
        
        #### 🔔 **Notificaciones:**
        - Multi-canal (Email, SMS, Slack, Teams, Discord)
        - Plantillas personalizables
        - Programación de envíos
        - Seguimiento de entrega
        
        ### 🎯 **Casos de Uso:**
        - Exportación masiva de datos
        - Análisis de sentimientos en tiempo real
        - Predicciones con ML/DL
        - Procesamiento de imágenes
        - Transacciones blockchain
        - Operaciones DeFi
        - Automatización de procesos
        - Monitoreo de sistemas
        
        ### 🔗 **Integraciones:**
        - OpenAI GPT-4
        - Anthropic Claude
        - Hugging Face
        - Ethereum/Polygon/BSC
        - Slack/Teams/Discord
        - Email/SMS
        
        ---
        
        **Desarrollado con ❤️ usando FastAPI, Python y tecnologías de vanguardia.**
        """,
        version="5.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Configurar CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Middleware de compresión
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Middleware personalizado
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(PerformanceMiddleware)
    app.add_middleware(RateLimitingMiddleware)
    
    # Middleware para headers personalizados
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = str(hash(request.url))
        return response
    
    # Manejadores de excepciones globales
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "message": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Error no manejado: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": "Error interno del servidor",
                "status_code": 500,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url)
            }
        )
    
    # Incluir routers
    app.include_router(export_router)
    app.include_router(nlp_router)
    app.include_router(enhanced_nlp_router)
    app.include_router(ai_router)
    app.include_router(blockchain_router)
    app.include_router(web3_router)
    app.include_router(analytics_router)
    app.include_router(notifications_router)
    app.include_router(system_router)
    
    # Rutas principales
    @app.get("/", tags=["Sistema"])
    async def root():
        """Endpoint raíz del sistema."""
        return {
            "message": "🚀 Export IA V5.0 - ULTIMATE ENTERPRISE SYSTEM",
            "version": "5.0.0",
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "features": [
                "Export IA Optimizado",
                "NLP Mejorado con IA",
                "Machine Learning Engine",
                "Deep Learning Engine", 
                "Computer Vision Engine",
                "Blockchain Engine",
                "Web3 Engine",
                "Seguridad Avanzada",
                "Automatización de Flujos",
                "Gestión Avanzada de Datos",
                "Monitoreo en Tiempo Real",
                "Analytics de Negocio",
                "Sistema de Notificaciones"
            ],
            "endpoints": {
                "docs": "/docs",
                "redoc": "/redoc",
                "openapi": "/openapi.json",
                "health": "/api/v1/system/health",
                "info": "/api/v1/system/info"
            }
        }
    
    @app.get("/health", tags=["Sistema"])
    async def health_check():
        """Verificar salud del sistema."""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "5.0.0",
                "engines": {},
                "systems": {}
            }
            
            # Verificar motores
            for name, engine in engines.items():
                try:
                    if hasattr(engine, 'health_check'):
                        health = await engine.health_check()
                        health_status["engines"][name] = health.get("status", "unknown")
                    else:
                        health_status["engines"][name] = "no_health_check"
                except Exception as e:
                    health_status["engines"][name] = f"error: {str(e)}"
            
            # Verificar sistemas
            systems = {
                "security": security_manager,
                "monitoring": monitoring_system,
                "data": data_manager,
                "automation": automation_system,
                "analytics": analytics_system,
                "notifications": notification_system
            }
            
            for name, system in systems.items():
                if system:
                    try:
                        if hasattr(system, 'health_check'):
                            health = await system.health_check()
                            health_status["systems"][name] = health.get("status", "unknown")
                        else:
                            health_status["systems"][name] = "operational"
                    except Exception as e:
                        health_status["systems"][name] = f"error: {str(e)}"
                else:
                    health_status["systems"][name] = "not_initialized"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error en health check: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    @app.get("/api/v1/system/info", tags=["Sistema"])
    async def system_info():
        """Información detallada del sistema."""
        try:
            info = {
                "system": {
                    "name": "Export IA V5.0 - ULTIMATE ENTERPRISE SYSTEM",
                    "version": "5.0.0",
                    "description": "Sistema empresarial completo con IA, Blockchain y Web3",
                    "status": "operational",
                    "uptime_seconds": (datetime.now() - datetime.now()).total_seconds()
                },
                "engines": {
                    "export": "Motor de exportación optimizado con procesamiento paralelo",
                    "nlp": "Procesamiento de lenguaje natural con IA avanzada",
                    "ml": "Machine Learning con AutoML y predicciones en tiempo real",
                    "dl": "Deep Learning con redes neuronales y GPU",
                    "cv": "Computer Vision con detección de objetos y OCR",
                    "blockchain": "Blockchain completo con contratos inteligentes",
                    "web3": "Motor Web3 y DeFi para criptomonedas"
                },
                "systems": {
                    "security": "Seguridad avanzada con JWT, DDoS protection y monitoreo",
                    "automation": "Automatización de flujos de trabajo con triggers",
                    "data": "Gestión avanzada de datos multi-nivel",
                    "monitoring": "Monitoreo en tiempo real con alertas",
                    "analytics": "Analytics de negocio con métricas avanzadas",
                    "notifications": "Sistema de notificaciones multi-canal"
                },
                "features": [
                    "Exportación masiva de datos",
                    "Procesamiento de lenguaje natural",
                    "Machine Learning y Deep Learning",
                    "Computer Vision",
                    "Blockchain y Web3",
                    "Seguridad empresarial",
                    "Automatización de procesos",
                    "Monitoreo en tiempo real",
                    "Analytics avanzados",
                    "Notificaciones multi-canal"
                ],
                "integrations": [
                    "OpenAI GPT-4",
                    "Anthropic Claude",
                    "Hugging Face",
                    "Ethereum/Polygon/BSC",
                    "Slack/Teams/Discord",
                    "Email/SMS"
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error al obtener información del sistema: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Montar archivos estáticos
    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    except Exception:
        logger.warning("No se pudo montar directorio static")
    
    return app


# Crear aplicación
app = create_enhanced_app()


if __name__ == "__main__":
    uvicorn.run(
        "enhanced_app_v5:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )




