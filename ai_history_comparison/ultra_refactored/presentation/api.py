"""
FastAPI Application Factory - Factory de Aplicación FastAPI
=========================================================

Factory para crear la aplicación FastAPI con todas las configuraciones
y dependencias necesarias.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import time
from contextlib import asynccontextmanager
from loguru import logger

from ..domain.exceptions import DomainException
from ..application.services import HistoryService, ComparisonService, QualityService, AnalysisService
from ..infrastructure.repositories import InMemoryHistoryRepository, InMemoryComparisonRepository
from ..infrastructure.services import TextContentAnalyzer, BasicQualityAssessor, CosineSimilarityCalculator
from .controllers import HistoryController, ComparisonController, QualityController
from .middleware import ErrorHandlerMiddleware, LoggingMiddleware
from .dependencies import setup_dependencies


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación."""
    # Startup
    logger.info("Starting AI History Comparison API")
    
    # Configurar dependencias
    setup_dependencies()
    
    # Inicializar servicios
    history_repo = InMemoryHistoryRepository()
    comparison_repo = InMemoryComparisonRepository()
    content_analyzer = TextContentAnalyzer()
    quality_assessor = BasicQualityAssessor()
    similarity_calculator = CosineSimilarityCalculator()
    
    # Crear servicios de aplicación
    history_service = HistoryService(history_repo, content_analyzer, quality_assessor)
    comparison_service = ComparisonService(history_repo, comparison_repo, similarity_calculator)
    quality_service = QualityService(history_repo, quality_assessor)
    analysis_service = AnalysisService(history_repo, comparison_repo, content_analyzer, quality_assessor)
    
    # Almacenar en el estado de la aplicación
    app.state.history_service = history_service
    app.state.comparison_service = comparison_service
    app.state.quality_service = quality_service
    app.state.analysis_service = analysis_service
    
    logger.info("AI History Comparison API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI History Comparison API")


def create_app() -> FastAPI:
    """
    Crear aplicación FastAPI.
    
    Returns:
        FastAPI: Aplicación configurada
    """
    app = FastAPI(
        title="AI History Comparison API",
        description="""
        API ultra-refactorizada para análisis y comparación de historial de IA.
        
        ## Características
        
        * **Análisis de Contenido**: Análisis detallado de contenido generado por IA
        * **Comparación de Entradas**: Comparación de similitud entre entradas
        * **Evaluación de Calidad**: Evaluación automática de calidad del contenido
        * **Análisis en Lote**: Procesamiento masivo de entradas
        
        ## Arquitectura
        
        * **Clean Architecture**: Separación clara de responsabilidades
        * **Domain-Driven Design**: Modelos de dominio ricos
        * **Repository Pattern**: Abstracción de acceso a datos
        * **Dependency Injection**: Inyección de dependencias
        """,
        version="2.0.0",
        contact={
            "name": "AI History Team",
            "email": "support@ai-history.com",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        lifespan=lifespan
    )
    
    # Configurar CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # En producción, especificar dominios permitidos
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Agregar middleware personalizado
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)
    
    # Configurar rutas
    _setup_routes(app)
    
    # Configurar documentación personalizada
    _setup_documentation(app)
    
    return app


def _setup_routes(app: FastAPI):
    """Configurar rutas de la API."""
    
    # Health check
    @app.get("/health", tags=["System"])
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "2.0.0",
            "service": "AI History Comparison API"
        }
    
    # Root endpoint
    @app.get("/", tags=["System"])
    async def root():
        """Root endpoint."""
        return {
            "message": "AI History Comparison API",
            "version": "2.0.0",
            "docs": "/docs",
            "health": "/health"
        }
    
    # Incluir controladores
    app.include_router(
        HistoryController().router,
        prefix="/api/v1/history",
        tags=["History"]
    )
    
    app.include_router(
        ComparisonController().router,
        prefix="/api/v1/comparisons",
        tags=["Comparisons"]
    )
    
    app.include_router(
        QualityController().router,
        prefix="/api/v1/quality",
        tags=["Quality"]
    )


def _setup_documentation(app: FastAPI):
    """Configurar documentación personalizada."""
    
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title="AI History Comparison API",
            version="2.0.0",
            description="""
            API ultra-refactorizada para análisis y comparación de historial de IA.
            
            ## Características Principales
            
            * **Análisis de Contenido**: Análisis detallado de contenido generado por IA
            * **Comparación de Entradas**: Comparación de similitud entre entradas
            * **Evaluación de Calidad**: Evaluación automática de calidad del contenido
            * **Análisis en Lote**: Procesamiento masivo de entradas
            
            ## Arquitectura
            
            * **Clean Architecture**: Separación clara de responsabilidades
            * **Domain-Driven Design**: Modelos de dominio ricos
            * **Repository Pattern**: Abstracción de acceso a datos
            * **Dependency Injection**: Inyección de dependencias
            
            ## Modelos de Datos
            
            * **HistoryEntry**: Entrada de historial de IA
            * **ComparisonResult**: Resultado de comparación
            * **QualityReport**: Reporte de calidad
            * **AnalysisJob**: Trabajo de análisis
            
            ## Endpoints Principales
            
            * **POST /api/v1/history/entries**: Crear entrada de historial
            * **GET /api/v1/history/entries**: Listar entradas
            * **POST /api/v1/comparisons**: Comparar entradas
            * **GET /api/v1/quality/reports**: Evaluar calidad
            """,
            routes=app.routes,
        )
        
        # Agregar información adicional
        openapi_schema["info"]["x-logo"] = {
            "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi


# Función para manejar errores de dominio
async def domain_exception_handler(request: Request, exc: DomainException):
    """Manejador de excepciones de dominio."""
    logger.error(f"Domain exception: {exc.message}", extra={"error_code": exc.error_code})
    
    return JSONResponse(
        status_code=400,
        content={
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "timestamp": time.time()
        }
    )


# Función para manejar errores generales
async def general_exception_handler(request: Request, exc: Exception):
    """Manejador de excepciones generales."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "timestamp": time.time()
        }
    )




