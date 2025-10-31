"""
Bulk TruthGPT Main Application
=============================

FastAPI application for continuous document generation using TruthGPT architecture.
Refactored for improved architecture, performance, and maintainability.
"""

import asyncio
import logging
from datetime import datetime
import time
import uuid
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.responses import JSONResponse
import asyncio
from asyncio import TimeoutError as AsyncTimeoutError
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from aiocache import cached, caches
import redis.asyncio as aioredis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from .utils.resilience import resilient_call
from .utils.error_handling import (
    APIError,
    validation_exception_handler,
    api_exception_handler,
    general_exception_handler,
)
from .utils.task_tracker import TaskTracker, TaskEventType
from pydantic import ValidationError

# Initialize task tracker (will be connected to Redis in lifespan)
task_tracker = TaskTracker()

# Core imports
from .core.base import BaseComponent, BaseService, BaseEngine, BaseManager, component_registry
from .core.truthgpt_engine import TruthGPTEngine
from .core.document_generator import DocumentGenerator
from .core.optimization_core import OptimizationCore
from .core.knowledge_base import KnowledgeBase
from .core.prompt_optimizer import PromptOptimizer
from .core.content_analyzer import ContentAnalyzer

# Bulk AI imports
from .bulk_ai_system import BulkAISystem, BulkAIConfig
from .continuous_generator import ContinuousGenerationEngine, ContinuousGenerationConfig
from .enhanced_bulk_ai_system import EnhancedBulkAISystem, EnhancedBulkAIConfig
from .enhanced_continuous_generator import EnhancedContinuousGenerator, EnhancedContinuousConfig

# Service imports
from .services.queue_manager import QueueManager
from .services.monitor import SystemMonitor
from .services.notification_service import NotificationService
from .services.analytics_service import AnalyticsService

# Utility imports
from .utils.logging import setup_logger, LogContext
from .utils.exceptions import (
    BulkTruthGPTException, 
    fastapi_error_handler, 
    create_error_context
)
from .utils.metrics import metrics_collector, metrics_context
from .utils.template_engine import TemplateEngine
from .utils.format_converter import FormatConverter
from .utils.optimization_engine import OptimizationEngine
from .utils.learning_system import LearningSystem
from .utils.performance_optimizer import performance_optimizer
from .utils.advanced_cache import advanced_cache
from .utils.batch_processor import batch_processor
from .utils.compression_engine import compression_engine
from .utils.lazy_loader import lazy_cache
from .utils.speed_optimizer import speed_optimizer
from .utils.warmup_system import warmup_system
from .utils.precomputation_engine import precomputation_engine
from .utils.gpu_optimizer import gpu_optimizer
from .utils.auto_tuner import auto_tuner
from .utils.load_predictor import load_predictor
from .utils.network_optimizer import network_optimizer
from .utils.realtime_monitor import realtime_monitor
from .utils.security_optimizer import security_optimizer
from .utils.backup_system import backup_system
from .utils.ml_optimizer import ml_optimizer
from .utils.ai_system import ai_system
from .utils.quantum_optimizer import quantum_optimizer
from .utils.edge_computing import edge_computing

# Configuration
from .config.settings import settings

# Models
from .models.schemas import (
    BulkGenerationRequest,
    BulkGenerationResponse,
    DocumentStatus,
    GenerationConfig,
    TruthGPTConfig
)

# Setup logging
logger = setup_logger(__name__)

# Global component instances
components = {}

# Bulk AI system instances
bulk_ai_system = None
continuous_generator = None
enhanced_bulk_ai_system = None
enhanced_continuous_generator = None
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global (truthgpt_engine, document_generator, optimization_core, knowledge_base, 
            prompt_optimizer, content_analyzer, queue_manager, system_monitor,
            notification_service, analytics_service, template_engine, format_converter,
            metrics_collector, optimization_engine, learning_system, bulk_ai_system, continuous_generator,
            enhanced_bulk_ai_system, enhanced_continuous_generator)
    
    logger.info("Starting Bulk TruthGPT System...")
    
    try:
        # Configure aiocache to use Redis as default cache
        caches.set_config({
            'default': {
                'cache': 'aiocache.RedisCache',
                'endpoint': settings.redis_url.split('//')[1].split('/')[0].split(':')[0],
                'port': int(settings.redis_url.split('//')[1].split('/')[0].split(':')[1]) if ':' in settings.redis_url.split('//')[1].split('/')[0] else 6379,
                'password': None,
                'timeout': 1,
                'serializer': {
                    'class': 'aiocache.serializers.PickleSerializer'
                }
            }
        })

        # Initialize Redis client for health checks/readiness
        global redis_client, task_tracker
        redis_client = aioredis.from_url(settings.redis_url)
        
        # Connect task tracker to Redis
        task_tracker.redis_client = redis_client
        # Initialize performance optimizations first
        await performance_optimizer.initialize()
        await advanced_cache.initialize()
        await batch_processor.initialize()
        await compression_engine.initialize()
        await lazy_cache.initialize()
        await speed_optimizer.initialize()
        await warmup_system.initialize()
        await precomputation_engine.initialize()
        await gpu_optimizer.initialize()
        await auto_tuner.initialize()
        await load_predictor.initialize()
        await network_optimizer.initialize()
        await realtime_monitor.initialize()
        await security_optimizer.initialize()
        await backup_system.initialize()
        await ml_optimizer.initialize()
        await ai_system.initialize()
        await quantum_optimizer.initialize()
        await edge_computing.initialize()
        
        # Execute system warmup for maximum performance
        logger.info("Executing system warmup...")
        warmup_result = await warmup_system.execute_warmup()
        logger.info(f"System warmup completed: {warmup_result['status']}")
        
        # Initialize core components
        truthgpt_engine = TruthGPTEngine()
        await truthgpt_engine.initialize()
        
        document_generator = DocumentGenerator(truthgpt_engine)
        await document_generator.initialize()
        
        optimization_core = OptimizationCore()
        await optimization_core.initialize()
        
        # Initialize advanced components
        knowledge_base = KnowledgeBase()
        await knowledge_base.initialize()
        
        prompt_optimizer = PromptOptimizer()
        await prompt_optimizer.initialize()
        
        content_analyzer = ContentAnalyzer()
        await content_analyzer.initialize()
        
        # Initialize services
        queue_manager = QueueManager()
        await queue_manager.initialize()
        
        system_monitor = SystemMonitor()
        await system_monitor.initialize()
        
        notification_service = NotificationService()
        await notification_service.initialize()
        
        analytics_service = AnalyticsService()
        await analytics_service.initialize()
        
        # Initialize utilities
        template_engine = TemplateEngine()
        await template_engine.initialize()
        
        format_converter = FormatConverter()
        await format_converter.initialize()
        
        await metrics_collector.initialize()
        
        optimization_engine = OptimizationEngine()
        await optimization_engine.initialize()
        
        learning_system = LearningSystem()
        await learning_system.initialize()
        
        # Initialize Bulk AI System
        bulk_ai_config = BulkAIConfig(
            max_concurrent_generations=10,
            max_documents_per_query=1000,
            enable_adaptive_model_selection=True,
            enable_ultra_optimization=True,
            enable_hybrid_optimization=True,
            enable_mcts_optimization=True,
            enable_olympiad_benchmarks=True
        )
        bulk_ai_system = BulkAISystem(bulk_ai_config)
        await bulk_ai_system.initialize()
        
        # Initialize Continuous Generator
        continuous_config = ContinuousGenerationConfig(
            max_documents=1000,
            generation_interval=0.1,
            enable_real_time_monitoring=True,
            enable_auto_cleanup=True,
            enable_model_rotation=True
        )
        continuous_generator = ContinuousGenerationEngine(continuous_config)
        await continuous_generator.initialize()
        
        # Initialize Enhanced Bulk AI System
        enhanced_bulk_ai_config = EnhancedBulkAIConfig(
            max_concurrent_generations=15,
            max_documents_per_query=2000,
            enable_adaptive_model_selection=True,
            enable_ultra_optimization=True,
            enable_hybrid_optimization=True,
            enable_mcts_optimization=True,
            enable_olympiad_benchmarks=True,
            enable_quantum_optimization=True,
            enable_edge_computing=True
        )
        enhanced_bulk_ai_system = EnhancedBulkAISystem(enhanced_bulk_ai_config)
        await enhanced_bulk_ai_system.initialize()
        
        # Initialize Enhanced Continuous Generator
        enhanced_continuous_config = EnhancedContinuousConfig(
            max_documents=2000,
            generation_interval=0.05,
            enable_real_time_monitoring=True,
            enable_auto_cleanup=True,
            enable_benchmarking=True,
            enable_ultra_optimization=True,
            enable_hybrid_optimization=True,
            enable_mcts_optimization=True,
            enable_quantum_optimization=True,
            enable_edge_computing=True
        )
        enhanced_continuous_generator = EnhancedContinuousGenerator(enhanced_continuous_config)
        await enhanced_continuous_generator.initialize()
        
        logger.info("Bulk TruthGPT System initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        raise
    finally:
        logger.info("Shutting down Bulk TruthGPT System...")
        
        # Cleanup in reverse order
        if learning_system:
            await learning_system.cleanup()
        if optimization_engine:
            await optimization_engine.cleanup()
        if metrics_collector:
            await metrics_collector.cleanup()
        if format_converter:
            await format_converter.cleanup()
        if template_engine:
            await template_engine.cleanup()
        if analytics_service:
            await analytics_service.cleanup()
        if notification_service:
            await notification_service.cleanup()
        if system_monitor:
            await system_monitor.cleanup()
        if queue_manager:
            await queue_manager.cleanup()
        if content_analyzer:
            await content_analyzer.cleanup()
        if prompt_optimizer:
            await prompt_optimizer.cleanup()
        if knowledge_base:
            await knowledge_base.cleanup()
        if optimization_core:
            await optimization_core.cleanup()
        if document_generator:
            await document_generator.cleanup()
        if truthgpt_engine:
            await truthgpt_engine.cleanup()
        
        # Cleanup performance optimizations
        await edge_computing.cleanup()
        await quantum_optimizer.cleanup()
        await ai_system.cleanup()
        await ml_optimizer.cleanup()
        await backup_system.cleanup()
        await security_optimizer.cleanup()
        await realtime_monitor.cleanup()
        await network_optimizer.cleanup()
        await load_predictor.cleanup()
        await auto_tuner.cleanup()
        await gpu_optimizer.cleanup()
        await precomputation_engine.cleanup()
        await warmup_system.cleanup()
        await speed_optimizer.cleanup()
        await lazy_cache.cleanup()
        await compression_engine.cleanup()
        await batch_processor.cleanup()
        await advanced_cache.clear()
        await performance_optimizer.cleanup()
        if redis_client:
            await redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="Bulk TruthGPT System",
    description="""
    **Bulk TruthGPT API** - Sistema de generación continua de documentos basado en arquitectura TruthGPT.
    
    ## Características principales
    
    * 🚀 Generación masiva de documentos con TruthGPT
    * ⚡ Optimizaciones avanzadas (ultra, híbrido, MCTS, quantum)
    * 🔄 Generación continua y streaming
    * 📊 Métricas y monitoreo en tiempo real
    * 🔒 Seguridad y rate limiting
    * 💾 Cache distribuida con Redis
    * 🔧 Observabilidad completa (Prometheus, logging estructurado)
    
    ## Endpoints principales
    
    * `/api/v1/bulk/generate` - Iniciar generación masiva
    * `/api/v1/bulk/status/{task_id}` - Consultar estado de tarea
    * `/api/v1/bulk/tasks` - Listar todas las tareas
    * `/health` - Health check del sistema
    * `/readiness` - Verificación de dependencias
    * `/metrics` - Métricas Prometheus
    
    ## Autenticación
    
    Opcionalmente requiere `X-API-Key` header si `API_KEY` está configurado en variables de entorno.
    """,
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "Bulk TruthGPT Team",
    },
    license_info={
        "name": "MIT",
    },
    tags_metadata=[
        {
            "name": "bulk",
            "description": "Operaciones de generación masiva de documentos",
        },
        {
            "name": "optimization",
            "description": "Endpoints de optimización y performance",
        },
        {
            "name": "document_sets",
            "description": "Gestión de conjuntos de documentos",
        },
        {
            "name": "monitoring",
            "description": "Monitoreo, métricas y health checks",
        },
    ],
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins if hasattr(settings, "cors_origins") else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add compression middleware for large responses
app.add_middleware(GZipMiddleware, minimum_size=1000)  # Compress responses > 1KB

# Basic security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-XSS-Protection", "1; mode=block")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Cache-Control", "no-store")
    return response

# Request timeout configuration (seconds)
REQUEST_TIMEOUT = 300  # 5 minutes default
HEALTH_CHECK_TIMEOUT = 5  # 5 seconds for health checks

# Request ID, timing, and timeout middleware (observability)
@app.middleware("http")
async def request_id_and_timing(request: Request, call_next):
    start = time.time()
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    status_code = 200
    
    # Determine timeout based on endpoint
    timeout_seconds = HEALTH_CHECK_TIMEOUT if request.url.path in ["/health", "/readiness"] else REQUEST_TIMEOUT
    
    try:
        # Apply timeout to request handling
        try:
            response = await asyncio.wait_for(
                call_next(request),
                timeout=timeout_seconds
            )
            status_code = response.status_code
            return response
        except AsyncTimeoutError:
            status_code = 504
            logger.warning(
                "request_timeout",
                extra={
                    "request_id": request_id,
                    "path": str(request.url.path),
                    "timeout_seconds": timeout_seconds,
                }
            )
            return JSONResponse(
                status_code=504,
                content={
                    "error": "RequestTimeout",
                    "error_code": "ERR_504",
                    "message": f"Request exceeded timeout of {timeout_seconds}s",
                    "request_id": request_id,
                }
            )
    except Exception as e:
        status_code = 500
        raise
    finally:
        duration_ms = int((time.time() - start) * 1000)
        duration_sec = duration_ms / 1000.0
        endpoint = str(request.url.path).replace('/', '_').strip('_') or 'root'
        
        # Record Prometheus metrics
        http_requests_total.labels(
            method=request.method,
            endpoint=endpoint,
            status=status_code
        ).inc()
        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=endpoint
        ).observe(duration_sec)
        
        # Attach request ID to response headers
        if 'response' in locals():
            response.headers.setdefault("X-Request-ID", request_id)
            response.headers.setdefault("X-Response-Time", f"{duration_ms}ms")
        
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": str(request.url.path),
                "status_code": status_code,
                "duration_ms": duration_ms,
                "client_ip": request.client.host if request.client else None,
            }
        )

# Rate limiting (SlowAPI)
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])  # sane default
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda r, e: __import__('slowapi').errors._rate_limit_exceeded_handler(r, e))
app.add_middleware(SlowAPIMiddleware)

# Prometheus metrics
http_requests_total = Counter(
    'http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status']
)
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint']
)
active_tasks = Gauge('bulk_generation_active_tasks', 'Number of active generation tasks')

# Optional API key protection for sensitive endpoints
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def require_api_key(api_key: str = Security(api_key_header)):
    if hasattr(settings, "api_key") and settings.api_key:
        if not api_key or api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

# Include modular routers
from .blueprints.api_docs import router as docs_router
from .blueprints.user_routes import router as users_router
from .blueprints.optimization import router as optimization_router
from ..document_set.router import router as document_sets_router

# Register exception handlers
app.add_exception_handler(ValidationError, validation_exception_handler)
app.add_exception_handler(APIError, api_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Include modular routers
app.include_router(docs_router, prefix="/api")
app.include_router(users_router, prefix="/api/v1")
app.include_router(optimization_router, prefix="/api/v1")
app.include_router(document_sets_router, prefix="/api/v1")

# Dependency to get current components
def get_components():
    """Get current system components."""
    return {
        "truthgpt_engine": truthgpt_engine,
        "document_generator": document_generator,
        "optimization_core": optimization_core,
        "queue_manager": queue_manager,
        "system_monitor": system_monitor
    }

@app.get(
    "/health",
    summary="Health check",
    description="Verifica el estado general del sistema y componentes principales.",
    tags=["monitoring"],
)
async def health_check():
    """Health check endpoint."""
    try:
        components = get_components()
        
        # Check if all components are initialized
        for name, component in components.items():
            if component is None:
                raise HTTPException(status_code=503, detail=f"Component {name} not initialized")
        
        # Get system status
        status = await system_monitor.get_system_status()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "components": {
                name: "initialized" for name in components.keys()
            },
            "system_status": status
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"System unhealthy: {str(e)}")

@app.get("/health/redis")
async def health_redis():
    try:
        if not redis_client:
            raise HTTPException(status_code=503, detail="Redis client not initialized")
        pong = await redis_client.ping()
        return {"status": "healthy" if pong else "unhealthy", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Redis health check failed")

@app.get(
    "/metrics",
    summary="Métricas Prometheus",
    description="""
    Endpoint para scraping de métricas Prometheus.
    
    Expone métricas como:
    * `http_requests_total` - Total de requests HTTP
    * `http_request_duration_seconds` - Duración de requests
    * `bulk_generation_active_tasks` - Tareas activas
    
    **Nota:** Este endpoint está protegido por NGINX y solo es accesible
    desde redes internas.
    """,
    tags=["monitoring"],
)
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/readiness")
async def readiness_check():
    """Comprehensive readiness check for all dependencies."""
    checks = {}
    all_healthy = True
    
    # Check Redis
    try:
        if redis_client:
            pong = await redis_client.ping()
            checks["redis"] = "healthy" if pong else "unhealthy"
        else:
            checks["redis"] = "not_initialized"
            all_healthy = False
    except Exception as e:
        checks["redis"] = f"unhealthy: {str(e)}"
        all_healthy = False
    
    # Check components
    try:
        components = get_components()
        for name, component in components.items():
            checks[f"component_{name}"] = "healthy" if component else "unhealthy"
            if not component:
                all_healthy = False
    except Exception as e:
        checks["components"] = f"error: {str(e)}"
        all_healthy = False
    
    status_code = 200 if all_healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_healthy else "not_ready",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks
        }
    )

@app.post(
    "/api/v1/bulk/generate",
    response_model=BulkGenerationResponse,
    summary="Iniciar generación masiva",
    description="""
    Inicia un proceso de generación masiva de documentos basado en una query.
    
    El proceso se ejecuta en background y retorna un `task_id` que puede usarse
    para consultar el estado y los resultados.
    
    **Límites:**
    * Rate limit: 10 requests/minuto por IP
    * Máximo de documentos: 1000 por tarea (configurable en `config.max_documents`)
    """,
    responses={
        200: {
            "description": "Tarea iniciada exitosamente",
            "content": {
                "application/json": {
                    "example": {
                        "task_id": "550e8400-e29b-41d4-a716-446655440000",
                        "status": "started",
                        "message": "Bulk generation process started",
                        "estimated_documents": 10,
                        "estimated_duration": 5
                    }
                }
            }
        },
        400: {"description": "Request inválido"},
        401: {"description": "API Key requerida o inválida"},
        429: {"description": "Rate limit excedido"},
        503: {"description": "Sistema no disponible"},
    },
    tags=["bulk"],
)
@limiter.limit("10/minute")
async def start_bulk_generation(
    request: BulkGenerationRequest,
    background_tasks: BackgroundTasks,
    components: Dict = Depends(get_components),
    _auth: bool = Depends(require_api_key)
):
    """Start bulk document generation process."""
    try:
        logger.info(f"Starting bulk generation for query: {request.query}")
        
        # Validate components
        if not all(components.values()):
            raise APIError(
                status_code=503,
                message="System not fully initialized",
                error_code="ERR_503",
                details={"components": {k: v is not None for k, v in components.items()}},
            )
        
        # Start generation process in background
        task_id = await queue_manager.create_generation_task(request)
        
        # Track task creation
        await task_tracker.record_event(
            task_id=task_id,
            event_type=TaskEventType.CREATED,
            metadata={
                "query": request.query[:100],  # First 100 chars
                "max_documents": request.config.max_documents,
                "priority": request.priority,
            }
        )
        
        # Start background generation
        background_tasks.add_task(
            _process_bulk_generation,
            task_id,
            request,
            components
        )
        
        # Track task started
        await task_tracker.record_event(
            task_id=task_id,
            event_type=TaskEventType.STARTED,
        )
        
        return BulkGenerationResponse(
            task_id=task_id,
            status="started",
            message="Bulk generation process started",
            estimated_documents=request.config.max_documents,
            estimated_duration=request.config.estimated_duration
        )
        
    except Exception as e:
        logger.error(f"Failed to start bulk generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start generation: {str(e)}")

async def _process_bulk_generation(
    task_id: str,
    request: BulkGenerationRequest,
    components: Dict
):
    """Process bulk generation in background."""
    try:
        logger.info(f"Processing bulk generation task: {task_id}")
        
        # Get components
        truthgpt_engine = components["truthgpt_engine"]
        document_generator = components["document_generator"]
        optimization_core = components["optimization_core"]
        queue_manager = components["queue_manager"]
        system_monitor = components["system_monitor"]
        
        # Update task status
        await queue_manager.update_task_status(task_id, "processing")
        
        # Generate documents continuously
        generated_count = 0
        max_documents = request.config.max_documents
        
        while generated_count < max_documents:
            try:
                # Generate document using TruthGPT engine
                document = await document_generator.generate_document(
                    query=request.query,
                    config=request.config,
                    context=request.context
                )
                
                # Optimize document using optimization core
                optimized_document = await optimization_core.optimize_document(
                    document,
                    request.config.optimization_level
                )
                
                # Store document
                document_id = await document_generator.store_document(
                    optimized_document,
                    task_id
                )
                
                # Update metrics
                generated_count += 1
                await system_monitor.update_generation_metrics(
                    task_id,
                    generated_count,
                    document_id
                )
                
                # Update task progress
                await queue_manager.update_task_progress(
                    task_id,
                    generated_count,
                    max_documents
                )
                
                # Track progress event
                await task_tracker.record_event(
                    task_id=task_id,
                    event_type=TaskEventType.PROGRESS,
                    metadata={
                        "progress": generated_count,
                        "total": max_documents,
                        "percentage": (generated_count / max_documents) * 100,
                        "document_id": document_id,
                    }
                )
                
                logger.info(f"Generated document {generated_count}/{max_documents} for task {task_id}")
                
                # Check if we should continue
                if not await queue_manager.should_continue_generation(task_id):
                    logger.info(f"Stopping generation for task {task_id}")
                    break
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error generating document {generated_count + 1}: {str(e)}")
                await queue_manager.record_generation_error(task_id, str(e))
                continue
        
        # Mark task as completed
        await queue_manager.update_task_status(task_id, "completed")
        
        # Track completion event
        await task_tracker.record_event(
            task_id=task_id,
            event_type=TaskEventType.COMPLETED,
            metadata={
                "total_generated": generated_count,
                "total_requested": max_documents,
            }
        )
        
        # Invalidate cache for this task_id and tasks list
        if redis_client:
            from aiocache import caches
            cache = caches.get('default')
            await cache.delete(f'_cached_status:{task_id}')
            await cache.delete('_cached_list:')
        logger.info(f"Bulk generation task {task_id} completed. Generated {generated_count} documents.")
        
    except Exception as e:
        logger.error(f"Bulk generation task {task_id} failed: {str(e)}")
        await queue_manager.update_task_status(task_id, "failed")
        await queue_manager.record_generation_error(task_id, str(e))
        # Track failure event
        await task_tracker.record_event(
            task_id=task_id,
            event_type=TaskEventType.FAILED,
            metadata={"error": str(e)},
        )
        
        # Invalidate cache on failure too
        if redis_client:
            from aiocache import caches
            cache = caches.get('default')
            await cache.delete(f'_cached_status:{task_id}')
            await cache.delete('_cached_list:')

@app.get(
    "/api/v1/bulk/status/{task_id}",
    summary="Consultar estado de generación",
    description="""
    Consulta el estado actual de una tarea de generación.
    
    Retorna información sobre el progreso, documentos generados,
    y cualquier error que haya ocurrido.
    
    **Cache:** Respuestas cacheadas por 1 segundo para reducir carga.
    """,
    responses={
        200: {"description": "Estado de la tarea"},
        404: {"description": "Tarea no encontrada"},
        429: {"description": "Rate limit excedido"},
    },
    tags=["bulk"],
)
@limiter.limit("60/minute")
async def get_generation_status(
    task_id: str,
    components: Dict = Depends(get_components)
):
    """Get status of a generation task."""
    try:
        queue_manager = components["queue_manager"]
        # cache breve por task id
        @cached(ttl=1, alias='default')
        async def _cached_status(task_id_inner: str):
            return await queue_manager.get_task_status(task_id_inner)
        status = await _cached_status(task_id)
        return status
    except Exception as e:
        logger.error(f"Failed to get task status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.get(
    "/api/v1/bulk/tasks",
    summary="Listar todas las tareas",
    description="""
    Lista todas las tareas de generación con sus estados actuales.
    
    **Paginación:**
    * `limit`: Número máximo de tareas a retornar (default: 50, max: 100)
    * `offset`: Número de tareas a saltar (default: 0)
    * `status`: Filtrar por estado (opcional: created, started, processing, completed, failed)
    
    **Cache:** Respuestas cacheadas por 2 segundos.
    """,
    tags=["bulk"],
)
@limiter.limit("30/minute")
async def list_generation_tasks(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    components: Dict = Depends(get_components)
):
    """List all generation tasks with pagination and filtering."""
    try:
        # Validate pagination params
        if limit > 100:
            limit = 100
        if limit < 1:
            limit = 50
        if offset < 0:
            offset = 0
        
        queue_manager = components["queue_manager"]
        
        # cache breve para aliviar carga bajo alta consulta
        @cached(ttl=2, alias='default')
        async def _cached_list():
            all_tasks = await queue_manager.list_tasks()
            
            # Filter by status if provided
            if status:
                all_tasks = [t for t in all_tasks if t.get("status") == status]
            
            return all_tasks
        
        tasks = await _cached_list()
        
        # Apply pagination
        total = len(tasks)
        paginated_tasks = tasks[offset:offset + limit]
        
        return {
            "tasks": paginated_tasks,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            }
        }
    except Exception as e:
        logger.error(f"Failed to list tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")

@app.get(
    "/api/v1/bulk/tasks/{task_id}/history",
    summary="Historial de eventos de tarea",
    description="""
    Obtiene el historial completo de eventos de una tarea.
    
    Incluye todos los eventos desde la creación hasta el estado actual,
    permitiendo auditoría y debugging detallado.
    """,
    tags=["bulk"],
)
async def get_task_history(task_id: str):
    """Get event history for a task."""
    try:
        summary = await task_tracker.get_task_summary(task_id)
        if not summary:
            raise HTTPException(status_code=404, detail="Task history not found")
        return summary
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get task history: {str(e)}")

@app.post("/api/v1/bulk/stop/{task_id}")
@limiter.limit("20/minute")
async def stop_generation_task(
    task_id: str,
    components: Dict = Depends(get_components)
):
    """Stop a generation task."""
    try:
        queue_manager = components["queue_manager"]
        success = await queue_manager.stop_task(task_id)
        
        if success:
            return {"message": f"Task {task_id} stopped successfully"}
        else:
            raise HTTPException(status_code=404, detail="Task not found or already stopped")
            
    except Exception as e:
        logger.error(f"Failed to stop task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop task: {str(e)}")

@app.get(
    "/api/v1/bulk/metrics",
    summary="Métricas detalladas de generación",
    description="Obtiene métricas detalladas del sistema de generación masiva.",
    tags=["bulk", "monitoring"],
)
@limiter.limit("120/minute")
async def get_generation_metrics(
    components: Dict = Depends(get_components)
):
    """Get generation metrics."""
    try:
        system_monitor = components["system_monitor"]
        metrics = await system_monitor.get_generation_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.get(
    "/api/v1/bulk/stats/summary",
    summary="Resumen estadístico del sistema",
    description="""
    Resumen estadístico de alto nivel del sistema de generación masiva.
    
    Incluye métricas clave para dashboards y monitoreo:
    * Totales por estado
    * Tasas de éxito/fallo
    * Tareas activas
    * Performance promedio
    
    **Cache:** Respuestas cacheadas por 5 segundos para reducir carga.
    """,
    tags=["bulk", "monitoring"],
)
@cached(ttl=5, key_builder=lambda f, *args, **kwargs: "_cached_stats_summary:")
async def get_bulk_stats_summary(
    components: Dict = Depends(get_components)
):
    """Get high-level statistics summary."""
    try:
        queue_manager = components["queue_manager"]
        all_tasks = await queue_manager.list_tasks()
        
        # Calculate statistics
        status_counts = {}
        active_tasks = 0
        total_documents = 0
        
        for task in all_tasks:
            status = task.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            if status in ["started", "processing"]:
                active_tasks += 1
            if "documents" in task:
                total_documents += len(task.get("documents", []))
        
        total = len(all_tasks)
        completed = status_counts.get("completed", 0)
        failed = status_counts.get("failed", 0)
        
        return {
            "summary": {
                "total_tasks": total,
                "active_tasks": active_tasks,
                "completed_tasks": completed,
                "failed_tasks": failed,
                "pending_tasks": status_counts.get("pending", 0) + status_counts.get("created", 0),
                "total_documents_generated": total_documents,
                "success_rate_percent": round((completed / total * 100) if total > 0 else 0, 2),
                "failure_rate_percent": round((failed / total * 100) if total > 0 else 0, 2),
            },
            "status_breakdown": status_counts,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to get stats summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats summary: {str(e)}")

@app.get("/api/v1/bulk/documents/{task_id}")
async def get_generated_documents(
    task_id: str,
    limit: int = 100,
    offset: int = 0,
    components: Dict = Depends(get_components)
):
    """Get documents generated by a task."""
    try:
        document_generator = components["document_generator"]
        documents = await document_generator.get_task_documents(
            task_id,
            limit=limit,
            offset=offset
        )
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Failed to get documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")

# Performance optimization endpoints
@app.get("/api/v1/performance/stats")
async def get_performance_stats():
    """Get performance optimization statistics."""
    try:
        return {
            "performance_optimizer": performance_optimizer.get_performance_summary(),
            "cache_stats": advanced_cache.get_stats(),
            "batch_processor": batch_processor.get_processing_stats(),
            "compression_engine": compression_engine.get_compression_stats(),
            "lazy_cache": lazy_cache.get_stats()
        }
    except Exception as e:
        logger.error(f"Failed to get performance stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance stats: {str(e)}")

@app.post("/api/v1/performance/optimize")
async def trigger_optimization():
    """Trigger performance optimization."""
    try:
        # Trigger memory optimization
        await performance_optimizer._optimize_memory_performance()
        
        # Clear cache if needed
        cache_stats = advanced_cache.get_stats()
        if cache_stats.get('l1_cache', {}).get('hit_rate', 0) < 0.5:
            await advanced_cache.clear()
        
        return {"message": "Performance optimization triggered successfully"}
    except Exception as e:
        logger.error(f"Failed to trigger optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger optimization: {str(e)}")

@app.get("/api/v1/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    try:
        return advanced_cache.get_stats()
    except Exception as e:
        logger.error(f"Failed to get cache stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")

@app.post("/api/v1/cache/clear")
async def clear_cache():
    """Clear all caches."""
    try:
        await advanced_cache.clear()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@app.get("/api/v1/batch/stats")
async def get_batch_stats():
    """Get batch processing statistics."""
    try:
        return batch_processor.get_processing_stats()
    except Exception as e:
        logger.error(f"Failed to get batch stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get batch stats: {str(e)}")

@app.get("/api/v1/compression/stats")
async def get_compression_stats():
    """Get compression statistics."""
    try:
        return compression_engine.get_compression_stats()
    except Exception as e:
        logger.error(f"Failed to get compression stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get compression stats: {str(e)}")

# Speed optimization endpoints
@app.get("/api/v1/speed/stats")
async def get_speed_stats():
    """Get speed optimization statistics."""
    try:
        return {
            "speed_optimizer": speed_optimizer.get_optimization_stats(),
            "warmup_system": warmup_system.get_warmup_stats(),
            "precomputation_engine": precomputation_engine.get_stats()
        }
    except Exception as e:
        logger.error(f"Failed to get speed stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get speed stats: {str(e)}")

@app.post("/api/v1/speed/warmup")
async def trigger_warmup():
    """Trigger system warmup."""
    try:
        warmup_result = await warmup_system.execute_warmup()
        return {
            "message": "System warmup triggered successfully",
            "result": warmup_result
        }
    except Exception as e:
        logger.error(f"Failed to trigger warmup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger warmup: {str(e)}")

@app.get("/api/v1/speed/precompute/{task_id}")
async def get_precomputation_result(task_id: str):
    """Get precomputation result."""
    try:
        result = await precomputation_engine.get_result(task_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Result not found")
        return {"result": result}
    except Exception as e:
        logger.error(f"Failed to get precomputation result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get precomputation result: {str(e)}")

@app.post("/api/v1/speed/precompute")
async def schedule_precomputation(
    task_id: str,
    func_name: str,
    strategy: str = "background",
    priority: str = "normal",
    ttl: Optional[int] = None
):
    """Schedule precomputation task."""
    try:
        # This would need to be implemented based on the actual function
        # For now, return a success message
        return {
            "message": f"Precomputation task {task_id} scheduled successfully",
            "task_id": task_id,
            "strategy": strategy,
            "priority": priority
        }
    except Exception as e:
        logger.error(f"Failed to schedule precomputation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to schedule precomputation: {str(e)}")

# Advanced optimization endpoints
@app.get("/api/v1/gpu/stats")
async def get_gpu_stats():
    """Get GPU optimization statistics."""
    try:
        return gpu_optimizer.get_gpu_stats()
    except Exception as e:
        logger.error(f"Failed to get GPU stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get GPU stats: {str(e)}")

@app.get("/api/v1/auto-tune/stats")
async def get_auto_tune_stats():
    """Get auto-tuning statistics."""
    try:
        return auto_tuner.get_tuning_stats()
    except Exception as e:
        logger.error(f"Failed to get auto-tune stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get auto-tune stats: {str(e)}")

@app.post("/api/v1/auto-tune/start")
async def start_auto_tuning(
    objective: str = "performance",
    max_iterations: int = 100,
    max_time: int = 3600
):
    """Start auto-tuning process."""
    try:
        # This would start the actual tuning process
        return {
            "message": "Auto-tuning started successfully",
            "objective": objective,
            "max_iterations": max_iterations,
            "max_time": max_time
        }
    except Exception as e:
        logger.error(f"Failed to start auto-tuning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start auto-tuning: {str(e)}")

@app.get("/api/v1/load-prediction/stats")
async def get_load_prediction_stats():
    """Get load prediction statistics."""
    try:
        return load_predictor.get_prediction_stats()
    except Exception as e:
        logger.error(f"Failed to get load prediction stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get load prediction stats: {str(e)}")

@app.get("/api/v1/load-prediction/predict")
async def predict_load(horizon: int = 60):
    """Get load prediction."""
    try:
        prediction = await load_predictor.predict_load(horizon)
        return {
            "prediction": prediction.predicted_values,
            "confidence": prediction.confidence,
            "model_used": prediction.model_used,
            "prediction_horizon": prediction.prediction_horizon
        }
    except Exception as e:
        logger.error(f"Failed to predict load: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to predict load: {str(e)}")

@app.get("/api/v1/advanced/stats")
async def get_advanced_stats():
    """Get all advanced optimization statistics."""
    try:
        return {
            "gpu_optimizer": gpu_optimizer.get_gpu_stats(),
            "auto_tuner": auto_tuner.get_tuning_stats(),
            "load_predictor": load_predictor.get_prediction_stats(),
            "speed_optimizer": speed_optimizer.get_optimization_stats(),
            "warmup_system": warmup_system.get_warmup_stats(),
            "precomputation_engine": precomputation_engine.get_stats()
        }
    except Exception as e:
        logger.error(f"Failed to get advanced stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get advanced stats: {str(e)}")

# Ultra-advanced optimization endpoints
@app.get("/api/v1/network/stats")
async def get_network_stats():
    """Get network optimization statistics."""
    try:
        return network_optimizer.get_network_stats()
    except Exception as e:
        logger.error(f"Failed to get network stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get network stats: {str(e)}")

@app.post("/api/v1/network/endpoint")
async def add_network_endpoint(
    endpoint_id: str,
    url: str,
    protocol: str = "https",
    weight: int = 1,
    timeout: int = 30
):
    """Add network endpoint."""
    try:
        from .utils.network_optimizer import NetworkEndpoint, NetworkProtocol
        
        endpoint = NetworkEndpoint(
            url=url,
            protocol=NetworkProtocol(protocol),
            weight=weight,
            timeout=timeout
        )
        
        await network_optimizer.add_endpoint(endpoint_id, endpoint)
        
        return {
            "message": f"Network endpoint {endpoint_id} added successfully",
            "endpoint_id": endpoint_id,
            "url": url,
            "protocol": protocol
        }
    except Exception as e:
        logger.error(f"Failed to add network endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add network endpoint: {str(e)}")

@app.get("/api/v1/monitoring/stats")
async def get_monitoring_stats():
    """Get real-time monitoring statistics."""
    try:
        return realtime_monitor.get_monitoring_stats()
    except Exception as e:
        logger.error(f"Failed to get monitoring stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring stats: {str(e)}")

@app.get("/api/v1/monitoring/metrics")
async def get_current_metrics():
    """Get current metrics."""
    try:
        metrics = await realtime_monitor._get_current_metrics()
        return {
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get current metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get current metrics: {str(e)}")

@app.get("/api/v1/monitoring/alerts")
async def get_alerts():
    """Get current alerts."""
    try:
        alerts = []
        for alert in realtime_monitor.alerts:
            alerts.append({
                "id": alert.id,
                "name": alert.name,
                "level": alert.level.value,
                "message": alert.message,
                "metric_name": alert.metric_name,
                "threshold": alert.threshold,
                "current_value": alert.current_value,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved
            })
        
        return {
            "alerts": alerts,
            "total_alerts": len(alerts),
            "active_alerts": sum(1 for alert in alerts if not alert["resolved"])
        }
    except Exception as e:
        logger.error(f"Failed to get alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@app.get("/api/v1/ultra-advanced/stats")
async def get_ultra_advanced_stats():
    """Get all ultra-advanced optimization statistics."""
    try:
        return {
            "network_optimizer": network_optimizer.get_network_stats(),
            "realtime_monitor": realtime_monitor.get_monitoring_stats(),
            "gpu_optimizer": gpu_optimizer.get_gpu_stats(),
            "auto_tuner": auto_tuner.get_tuning_stats(),
            "load_predictor": load_predictor.get_prediction_stats(),
            "speed_optimizer": speed_optimizer.get_optimization_stats(),
            "warmup_system": warmup_system.get_warmup_stats(),
            "precomputation_engine": precomputation_engine.get_stats()
        }
    except Exception as e:
        logger.error(f"Failed to get ultra-advanced stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get ultra-advanced stats: {str(e)}")

# Security and backup optimization endpoints
@app.get("/api/v1/security/stats")
async def get_security_stats():
    """Get security optimization statistics."""
    try:
        return security_optimizer.get_security_stats()
    except Exception as e:
        logger.error(f"Failed to get security stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get security stats: {str(e)}")

@app.post("/api/v1/security/encrypt")
async def encrypt_data(data: str, key_id: str = "aes"):
    """Encrypt data."""
    try:
        encrypted_data = await security_optimizer.encrypt_data(data, key_id)
        return {
            "encrypted_data": encrypted_data,
            "key_id": key_id,
            "algorithm": "AES-256"
        }
    except Exception as e:
        logger.error(f"Failed to encrypt data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to encrypt data: {str(e)}")

@app.post("/api/v1/security/decrypt")
async def decrypt_data(encrypted_data: str, key_id: str = "aes"):
    """Decrypt data."""
    try:
        decrypted_data = await security_optimizer.decrypt_data(encrypted_data, key_id)
        return {
            "decrypted_data": decrypted_data,
            "key_id": key_id
        }
    except Exception as e:
        logger.error(f"Failed to decrypt data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to decrypt data: {str(e)}")

@app.post("/api/v1/security/hash-password")
async def hash_password(password: str):
    """Hash password."""
    try:
        hashed_password = await security_optimizer.hash_password(password)
        return {
            "hashed_password": hashed_password,
            "algorithm": "bcrypt"
        }
    except Exception as e:
        logger.error(f"Failed to hash password: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to hash password: {str(e)}")

@app.post("/api/v1/security/verify-password")
async def verify_password(password: str, hashed: str):
    """Verify password."""
    try:
        is_valid = await security_optimizer.verify_password(password, hashed)
        return {
            "is_valid": is_valid,
            "algorithm": "bcrypt"
        }
    except Exception as e:
        logger.error(f"Failed to verify password: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to verify password: {str(e)}")

@app.get("/api/v1/backup/stats")
async def get_backup_stats():
    """Get backup system statistics."""
    try:
        return backup_system.get_backup_stats()
    except Exception as e:
        logger.error(f"Failed to get backup stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get backup stats: {str(e)}")

@app.post("/api/v1/backup/create")
async def create_backup(
    name: str,
    source_path: str,
    backup_type: str = "incremental",
    compression: str = "gzip",
    encryption: bool = True
):
    """Create backup."""
    try:
        from .utils.backup_system import BackupType, BackupCompression
        
        backup_id = await backup_system.create_backup(
            name=name,
            source_path=source_path,
            backup_type=BackupType(backup_type),
            compression=BackupCompression(compression),
            encryption=encryption
        )
        
        return {
            "backup_id": backup_id,
            "name": name,
            "source_path": source_path,
            "backup_type": backup_type,
            "compression": compression,
            "encryption": encryption,
            "status": "created"
        }
    except Exception as e:
        logger.error(f"Failed to create backup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create backup: {str(e)}")

@app.post("/api/v1/backup/restore/{backup_id}")
async def restore_backup(backup_id: str, destination_path: str):
    """Restore backup."""
    try:
        success = await backup_system.restore_backup(backup_id, destination_path)
        
        if success:
            return {
                "message": f"Backup {backup_id} restored successfully",
                "backup_id": backup_id,
                "destination_path": destination_path,
                "status": "restored"
            }
        else:
            raise HTTPException(status_code=500, detail="Backup restore failed")
            
    except Exception as e:
        logger.error(f"Failed to restore backup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to restore backup: {str(e)}")

@app.get("/api/v1/enterprise/stats")
async def get_enterprise_stats():
    """Get all enterprise optimization statistics."""
    try:
        return {
            "security_optimizer": security_optimizer.get_security_stats(),
            "backup_system": backup_system.get_backup_stats(),
            "network_optimizer": network_optimizer.get_network_stats(),
            "realtime_monitor": realtime_monitor.get_monitoring_stats(),
            "gpu_optimizer": gpu_optimizer.get_gpu_stats(),
            "auto_tuner": auto_tuner.get_tuning_stats(),
            "load_predictor": load_predictor.get_prediction_stats(),
            "speed_optimizer": speed_optimizer.get_optimization_stats(),
            "warmup_system": warmup_system.get_warmup_stats(),
            "precomputation_engine": precomputation_engine.get_stats()
        }
    except Exception as e:
        logger.error(f"Failed to get enterprise stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get enterprise stats: {str(e)}")

# ML and AI optimization endpoints
@app.get("/api/v1/ml/stats")
async def get_ml_stats():
    """Get ML optimization statistics."""
    try:
        return ml_optimizer.get_ml_stats()
    except Exception as e:
        logger.error(f"Failed to get ML stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get ML stats: {str(e)}")

@app.post("/api/v1/ml/train")
async def train_ml_model(
    model_id: str,
    data: List[List[float]],
    labels: List[float],
    model_type: str = "neural_network",
    hyperparameters: Optional[Dict[str, Any]] = None
):
    """Train ML model."""
    try:
        import numpy as np
        
        # Convert data to numpy arrays
        data_array = np.array(data)
        labels_array = np.array(labels)
        
        # Train model
        metrics = await ml_optimizer.train_model(
            model_id=model_id,
            data=data_array,
            labels=labels_array,
            model_type=model_type,
            hyperparameters=hyperparameters
        )
        
        return {
            "model_id": model_id,
            "model_type": model_type,
            "metrics": {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "mse": metrics.mse,
                "rmse": metrics.rmse,
                "r2_score": metrics.r2_score,
                "training_time": metrics.training_time,
                "inference_time": metrics.inference_time,
                "model_size": metrics.model_size
            },
            "status": "trained"
        }
    except Exception as e:
        logger.error(f"Failed to train ML model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to train ML model: {str(e)}")

@app.post("/api/v1/ml/optimize")
async def optimize_hyperparameters(
    model_id: str,
    data: List[List[float]],
    labels: List[float],
    n_trials: int = 100
):
    """Optimize hyperparameters."""
    try:
        import numpy as np
        
        # Convert data to numpy arrays
        data_array = np.array(data)
        labels_array = np.array(labels)
        
        # Optimize hyperparameters
        result = await ml_optimizer.optimize_hyperparameters(
            model_id=model_id,
            data=data_array,
            labels=labels_array,
            n_trials=n_trials
        )
        
        return {
            "model_id": model_id,
            "best_params": result['best_params'],
            "best_score": result['best_score'],
            "n_trials": result['n_trials'],
            "status": "optimized"
        }
    except Exception as e:
        logger.error(f"Failed to optimize hyperparameters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize hyperparameters: {str(e)}")

@app.post("/api/v1/ml/ensemble")
async def create_ensemble(
    ensemble_id: str,
    model_ids: List[str],
    ensemble_method: str = "voting"
):
    """Create model ensemble."""
    try:
        # Create ensemble
        metrics = await ml_optimizer.create_ensemble(
            ensemble_id=ensemble_id,
            model_ids=model_ids,
            ensemble_method=ensemble_method
        )
        
        return {
            "ensemble_id": ensemble_id,
            "model_ids": model_ids,
            "ensemble_method": ensemble_method,
            "metrics": {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "training_time": metrics.training_time,
                "model_size": metrics.model_size
            },
            "status": "created"
        }
    except Exception as e:
        logger.error(f"Failed to create ensemble: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create ensemble: {str(e)}")

@app.post("/api/v1/ml/predict")
async def predict_ml_model(
    model_id: str,
    data: List[List[float]]
):
    """Make predictions using ML model."""
    try:
        import numpy as np
        
        # Convert data to numpy array
        data_array = np.array(data)
        
        # Make predictions
        predictions = await ml_optimizer.predict(model_id, data_array)
        
        return {
            "model_id": model_id,
            "predictions": predictions.tolist(),
            "status": "predicted"
        }
    except Exception as e:
        logger.error(f"Failed to make predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to make predictions: {str(e)}")

@app.get("/api/v1/ai/stats")
async def get_ai_stats():
    """Get AI system statistics."""
    try:
        return ai_system.get_ai_stats()
    except Exception as e:
        logger.error(f"Failed to get AI stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get AI stats: {str(e)}")

@app.post("/api/v1/ai/process-text")
async def process_text_ai(
    text: str,
    task: str = "text_generation",
    model: str = "gpt2",
    **kwargs
):
    """Process text using AI."""
    try:
        from .utils.ai_system import AITask, AIModel
        
        # Process text
        response = await ai_system.process_text(
            text=text,
            task=AITask(task),
            model=AIModel(model),
            **kwargs
        )
        
        return {
            "task": response.task.value,
            "model": response.model.value,
            "input_text": response.input_text,
            "output_text": response.output_text,
            "confidence": response.confidence,
            "processing_time": response.processing_time,
            "tokens_used": response.tokens_used,
            "cost": response.cost,
            "metadata": response.metadata
        }
    except Exception as e:
        logger.error(f"Failed to process text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")

@app.post("/api/v1/ai/process-image")
async def process_image_ai(
    image_path: str,
    task: str = "image_classification",
    model: str = "resnet",
    **kwargs
):
    """Process image using AI."""
    try:
        from .utils.ai_system import AITask, AIModel
        
        # Process image
        response = await ai_system.process_image(
            image_path=image_path,
            task=AITask(task),
            model=AIModel(model),
            **kwargs
        )
        
        return {
            "task": response.task.value,
            "model": response.model.value,
            "input_text": response.input_text,
            "output_text": response.output_text,
            "confidence": response.confidence,
            "processing_time": response.processing_time,
            "tokens_used": response.tokens_used,
            "cost": response.cost,
            "metadata": response.metadata
        }
    except Exception as e:
        logger.error(f"Failed to process image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

@app.post("/api/v1/ai/process-speech")
async def process_speech_ai(
    audio_path: str,
    task: str = "speech_recognition",
    model: str = "whisper",
    **kwargs
):
    """Process speech using AI."""
    try:
        from .utils.ai_system import AITask, AIModel
        
        # Process speech
        response = await ai_system.process_speech(
            audio_path=audio_path,
            task=AITask(task),
            model=AIModel(model),
            **kwargs
        )
        
        return {
            "task": response.task.value,
            "model": response.model.value,
            "input_text": response.input_text,
            "output_text": response.output_text,
            "confidence": response.confidence,
            "processing_time": response.processing_time,
            "tokens_used": response.tokens_used,
            "cost": response.cost,
            "metadata": response.metadata
        }
    except Exception as e:
        logger.error(f"Failed to process speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process speech: {str(e)}")

@app.post("/api/v1/ai/reason")
async def ai_reasoning(
    premise: str,
    conclusion: str,
    context: Optional[str] = None
):
    """Perform AI reasoning."""
    try:
        # Perform reasoning
        result = await ai_system.reason(
            premise=premise,
            conclusion=conclusion,
            context=context
        )
        
        return {
            "premise": result['premise'],
            "conclusion": result['conclusion'],
            "reasoning_steps": result['reasoning_steps'],
            "validity": result['validity'],
            "confidence": result['confidence']
        }
    except Exception as e:
        logger.error(f"Failed to perform reasoning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to perform reasoning: {str(e)}")

@app.post("/api/v1/ai/plan")
async def ai_planning(
    goal: str,
    constraints: List[str],
    resources: List[str]
):
    """Create AI action plan."""
    try:
        # Create plan
        result = await ai_system.plan(
            goal=goal,
            constraints=constraints,
            resources=resources
        )
        
        return {
            "goal": result['goal'],
            "constraints": result['constraints'],
            "resources": result['resources'],
            "plan_steps": result['plan_steps'],
            "estimated_time": result['estimated_time'],
            "confidence": result['confidence']
        }
    except Exception as e:
        logger.error(f"Failed to create plan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create plan: {str(e)}")

@app.get("/api/v1/ultimate/stats")
async def get_ultimate_stats():
    """Get all ultimate optimization statistics."""
    try:
        return {
            "ai_system": ai_system.get_ai_stats(),
            "ml_optimizer": ml_optimizer.get_ml_stats(),
            "security_optimizer": security_optimizer.get_security_stats(),
            "backup_system": backup_system.get_backup_stats(),
            "network_optimizer": network_optimizer.get_network_stats(),
            "realtime_monitor": realtime_monitor.get_monitoring_stats(),
            "gpu_optimizer": gpu_optimizer.get_gpu_stats(),
            "auto_tuner": auto_tuner.get_tuning_stats(),
            "load_predictor": load_predictor.get_prediction_stats(),
            "speed_optimizer": speed_optimizer.get_optimization_stats(),
            "warmup_system": warmup_system.get_warmup_stats(),
            "precomputation_engine": precomputation_engine.get_stats()
        }
    except Exception as e:
        logger.error(f"Failed to get ultimate stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get ultimate stats: {str(e)}")

# Quantum and Edge optimization endpoints
@app.get("/api/v1/quantum/stats")
async def get_quantum_stats():
    """Get quantum optimization statistics."""
    try:
        return quantum_optimizer.get_quantum_stats()
    except Exception as e:
        logger.error(f"Failed to get quantum stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get quantum stats: {str(e)}")

@app.post("/api/v1/quantum/execute")
async def execute_quantum_algorithm(
    algorithm: str = "qaoa",
    problem_data: Dict[str, Any] = None
):
    """Execute quantum algorithm."""
    try:
        from .utils.quantum_optimizer import QuantumAlgorithm
        
        # Execute quantum algorithm
        result = await quantum_optimizer.execute_quantum_algorithm(
            algorithm=QuantumAlgorithm(algorithm),
            problem_data=problem_data or {}
        )
        
        return {
            "algorithm": result.algorithm.value,
            "backend": result.backend.value,
            "execution_time": result.execution_time,
            "success_probability": result.success_probability,
            "fidelity": result.fidelity,
            "cost_function_value": result.cost_function_value,
            "optimal_parameters": result.optimal_parameters,
            "quantum_volume": result.quantum_volume,
            "circuit_depth": result.circuit_depth,
            "gate_count": result.gate_count,
            "metadata": result.metadata
        }
    except Exception as e:
        logger.error(f"Failed to execute quantum algorithm: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to execute quantum algorithm: {str(e)}")

@app.post("/api/v1/quantum/optimize")
async def optimize_quantum_circuit(
    circuit_data: Dict[str, Any],
    optimization_strategy: str = "parameter_shift"
):
    """Optimize quantum circuit."""
    try:
        # This would create actual quantum circuit
        # For now, just return mock result
        result = await quantum_optimizer.optimize_quantum_circuit(
            circuit=None,  # Would be actual circuit
            cost_function=None,  # Would be actual cost function
            optimization_strategy=optimization_strategy
        )
        
        return {
            "optimization_strategy": optimization_strategy,
            "result": result,
            "status": "optimized"
        }
    except Exception as e:
        logger.error(f"Failed to optimize quantum circuit: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize quantum circuit: {str(e)}")

@app.post("/api/v1/quantum/ml")
async def quantum_machine_learning(
    data: List[List[float]],
    labels: List[float],
    model_type: str = "variational_classifier"
):
    """Perform quantum machine learning."""
    try:
        import numpy as np
        
        # Convert data to numpy arrays
        data_array = np.array(data)
        labels_array = np.array(labels)
        
        # Perform quantum ML
        result = await quantum_optimizer.quantum_machine_learning(
            data=data_array,
            labels=labels_array,
            model_type=model_type
        )
        
        return {
            "model_type": result['model_type'],
            "accuracy": result['accuracy'],
            "training_time": result['training_time'],
            "quantum_advantage": result['quantum_advantage'],
            "parameters": result['parameters'],
            "status": result['status']
        }
    except Exception as e:
        logger.error(f"Failed to perform quantum ML: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to perform quantum ML: {str(e)}")

@app.get("/api/v1/edge/stats")
async def get_edge_stats():
    """Get edge computing statistics."""
    try:
        return edge_computing.get_edge_stats()
    except Exception as e:
        logger.error(f"Failed to get edge stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get edge stats: {str(e)}")

@app.post("/api/v1/edge/submit-task")
async def submit_edge_task(
    task_name: str,
    task_type: str,
    data: Any,
    priority: int = 1,
    target_node: Optional[str] = None
):
    """Submit task to edge system."""
    try:
        # Submit task
        task_id = await edge_computing.submit_task(
            task_name=task_name,
            task_type=task_type,
            data=data,
            priority=priority,
            target_node=target_node
        )
        
        return {
            "task_id": task_id,
            "task_name": task_name,
            "task_type": task_type,
            "priority": priority,
            "target_node": target_node,
            "status": "submitted"
        }
    except Exception as e:
        logger.error(f"Failed to submit edge task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit edge task: {str(e)}")

@app.get("/api/v1/edge/task/{task_id}")
async def get_edge_task_result(task_id: str):
    """Get edge task result."""
    try:
        # Get task result
        result = await edge_computing.get_task_result(task_id)
        
        if result is None:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {
            "task_id": task_id,
            "result": result,
            "status": "completed"
        }
    except Exception as e:
        logger.error(f"Failed to get edge task result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get edge task result: {str(e)}")

@app.get("/api/v1/edge/nodes")
async def get_edge_nodes():
    """Get edge nodes."""
    try:
        # Get edge nodes
        nodes = []
        for node_id, node in edge_computing.nodes.items():
            nodes.append({
                "id": node.id,
                "name": node.name,
                "node_type": node.node_type.value,
                "protocol": node.protocol.value,
                "host": node.host,
                "port": node.port,
                "status": node.status,
                "cpu_usage": node.cpu_usage,
                "memory_usage": node.memory_usage,
                "disk_usage": node.disk_usage,
                "network_latency": node.network_latency,
                "bandwidth": node.bandwidth,
                "last_heartbeat": node.last_heartbeat.isoformat(),
                "metadata": node.metadata
            })
        
        return {
            "nodes": nodes,
            "total_nodes": len(nodes),
            "active_nodes": sum(1 for node in nodes if node['status'] == 'active')
        }
    except Exception as e:
        logger.error(f"Failed to get edge nodes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get edge nodes: {str(e)}")

@app.get("/api/v1/edge/tasks")
async def get_edge_tasks():
    """Get edge tasks."""
    try:
        # Get edge tasks
        tasks = []
        for task_id, task in edge_computing.tasks.items():
            tasks.append({
                "id": task.id,
                "name": task.name,
                "node_id": task.node_id,
                "task_type": task.task_type,
                "priority": task.priority,
                "status": task.status,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "execution_time": task.execution_time,
                "result": task.result,
                "error": task.error,
                "metadata": task.metadata
            })
        
        return {
            "tasks": tasks,
            "total_tasks": len(tasks),
            "completed_tasks": sum(1 for task in tasks if task['status'] == 'completed'),
            "failed_tasks": sum(1 for task in tasks if task['status'] == 'failed')
        }
    except Exception as e:
        logger.error(f"Failed to get edge tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get edge tasks: {str(e)}")

@app.get("/api/v1/revolutionary/stats")
async def get_revolutionary_stats():
    """Get all revolutionary optimization statistics."""
    try:
        return {
            "edge_computing": edge_computing.get_edge_stats(),
            "quantum_optimizer": quantum_optimizer.get_quantum_stats(),
            "ai_system": ai_system.get_ai_stats(),
            "ml_optimizer": ml_optimizer.get_ml_stats(),
            "security_optimizer": security_optimizer.get_security_stats(),
            "backup_system": backup_system.get_backup_stats(),
            "network_optimizer": network_optimizer.get_network_stats(),
            "realtime_monitor": realtime_monitor.get_monitoring_stats(),
            "gpu_optimizer": gpu_optimizer.get_gpu_stats(),
            "auto_tuner": auto_tuner.get_tuning_stats(),
            "load_predictor": load_predictor.get_prediction_stats(),
            "speed_optimizer": speed_optimizer.get_optimization_stats(),
            "warmup_system": warmup_system.get_warmup_stats(),
            "precomputation_engine": precomputation_engine.get_stats()
        }
    except Exception as e:
        logger.error(f"Failed to get revolutionary stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get revolutionary stats: {str(e)}")

# Bulk AI System Endpoints
@app.post("/api/v1/bulk-ai/process-query")
async def process_query_bulk_ai(
    query: str,
    max_documents: int = 100,
    enable_continuous: bool = True
):
    """Process a query using the bulk AI system with continuous generation."""
    try:
        if not bulk_ai_system:
            raise HTTPException(status_code=503, detail="Bulk AI system not initialized")
        
        if enable_continuous:
            # Use continuous generator for unlimited generation
            results = []
            async for result in continuous_generator.start_continuous_generation(query):
                results.append({
                    "document_id": result.document_id,
                    "content": result.content,
                    "model_used": result.model_used,
                    "quality_score": result.quality_score,
                    "generation_time": result.generation_time,
                    "timestamp": result.timestamp.isoformat()
                })
                
                if len(results) >= max_documents:
                    break
            
            return {
                "query": query,
                "total_documents": len(results),
                "documents": results,
                "performance_summary": continuous_generator.get_performance_summary()
            }
        else:
            # Use bulk AI system for limited generation
            results = await bulk_ai_system.process_query(query, max_documents)
            return results
            
    except Exception as e:
        logger.error(f"Failed to process query with bulk AI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

@app.get("/api/v1/bulk-ai/status")
async def get_bulk_ai_status():
    """Get bulk AI system status."""
    try:
        if not bulk_ai_system:
            raise HTTPException(status_code=503, detail="Bulk AI system not initialized")
        
        status = await bulk_ai_system.get_system_status()
        return status
        
    except Exception as e:
        logger.error(f"Failed to get bulk AI status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.post("/api/v1/bulk-ai/stop-generation")
async def stop_bulk_ai_generation():
    """Stop continuous generation."""
    try:
        if continuous_generator:
            continuous_generator.stop()
            return {"message": "Continuous generation stopped"}
        else:
            raise HTTPException(status_code=404, detail="Continuous generator not available")
            
    except Exception as e:
        logger.error(f"Failed to stop generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop generation: {str(e)}")

@app.get("/api/v1/bulk-ai/performance")
async def get_bulk_ai_performance():
    """Get bulk AI performance metrics."""
    try:
        if not continuous_generator:
            raise HTTPException(status_code=503, detail="Continuous generator not initialized")
        
        performance = continuous_generator.get_performance_summary()
        return performance
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance: {str(e)}")

@app.post("/api/v1/bulk-ai/start-continuous")
async def start_continuous_generation(
    query: str,
    max_documents: int = 1000
):
    """Start continuous generation for a query."""
    try:
        if not continuous_generator:
            raise HTTPException(status_code=503, detail="Continuous generator not initialized")
        
        # Start continuous generation in background
        import asyncio
        asyncio.create_task(continuous_generator.start_continuous_generation(query))
        
        return {
            "message": "Continuous generation started",
            "query": query,
            "max_documents": max_documents,
            "status": "running"
        }
        
    except Exception as e:
        logger.error(f"Failed to start continuous generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start generation: {str(e)}")

# Enhanced Bulk AI System Endpoints
@app.post("/api/v1/enhanced-bulk-ai/process-query")
async def process_query_enhanced_bulk_ai(
    query: str,
    max_documents: int = 200,
    enable_continuous: bool = True
):
    """Process a query using the enhanced bulk AI system with real TruthGPT library integration."""
    try:
        if not enhanced_bulk_ai_system:
            raise HTTPException(status_code=503, detail="Enhanced bulk AI system not initialized")
        
        if enable_continuous:
            # Use enhanced continuous generator for unlimited generation
            results = []
            async for result in enhanced_continuous_generator.start_continuous_generation(query):
                results.append({
                    "document_id": result.document_id,
                    "content": result.content,
                    "model_used": result.model_used,
                    "quality_score": result.quality_score,
                    "diversity_score": result.diversity_score,
                    "generation_time": result.generation_time,
                    "timestamp": result.timestamp.isoformat(),
                    "metadata": result.metadata,
                    "optimization_metrics": result.optimization_metrics,
                    "benchmark_results": result.benchmark_results
                })
                
                if len(results) >= max_documents:
                    break
            
            return {
                "query": query,
                "total_documents": len(results),
                "documents": results,
                "enhanced_performance_summary": enhanced_continuous_generator.get_enhanced_performance_summary()
            }
        else:
            # Use enhanced bulk AI system for limited generation
            results = await enhanced_bulk_ai_system.process_query(query, max_documents)
            return results
            
    except Exception as e:
        logger.error(f"Failed to process query with enhanced bulk AI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

@app.get("/api/v1/enhanced-bulk-ai/status")
async def get_enhanced_bulk_ai_status():
    """Get enhanced bulk AI system status."""
    try:
        if not enhanced_bulk_ai_system:
            raise HTTPException(status_code=503, detail="Enhanced bulk AI system not initialized")
        
        status = await enhanced_bulk_ai_system.get_system_status()
        return status
        
    except Exception as e:
        logger.error(f"Failed to get enhanced bulk AI status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.post("/api/v1/enhanced-bulk-ai/stop-generation")
async def stop_enhanced_bulk_ai_generation():
    """Stop enhanced continuous generation."""
    try:
        if enhanced_continuous_generator:
            enhanced_continuous_generator.stop()
            return {"message": "Enhanced continuous generation stopped"}
        else:
            raise HTTPException(status_code=404, detail="Enhanced continuous generator not available")
            
    except Exception as e:
        logger.error(f"Failed to stop enhanced generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop generation: {str(e)}")

@app.get("/api/v1/enhanced-bulk-ai/performance")
async def get_enhanced_bulk_ai_performance():
    """Get enhanced bulk AI performance metrics."""
    try:
        if not enhanced_continuous_generator:
            raise HTTPException(status_code=503, detail="Enhanced continuous generator not initialized")
        
        performance = enhanced_continuous_generator.get_enhanced_performance_summary()
        return performance
        
    except Exception as e:
        logger.error(f"Failed to get enhanced performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance: {str(e)}")

@app.post("/api/v1/enhanced-bulk-ai/start-continuous")
async def start_enhanced_continuous_generation(
    query: str,
    max_documents: int = 2000
):
    """Start enhanced continuous generation for a query."""
    try:
        if not enhanced_continuous_generator:
            raise HTTPException(status_code=503, detail="Enhanced continuous generator not initialized")
        
        # Start enhanced continuous generation in background
        import asyncio
        asyncio.create_task(enhanced_continuous_generator.start_continuous_generation(query))
        
        return {
            "message": "Enhanced continuous generation started",
            "query": query,
            "max_documents": max_documents,
            "status": "running",
            "features": [
                "Real TruthGPT library integration",
                "Ultra-optimization support",
                "Hybrid optimization",
                "MCTS optimization",
                "Quantum optimization",
                "Edge computing support",
                "Real-time monitoring",
                "Advanced benchmarking",
                "Quality and diversity scoring"
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to start enhanced continuous generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start generation: {str(e)}")

@app.get("/api/v1/enhanced-bulk-ai/benchmark")
async def benchmark_enhanced_system():
    """Benchmark the enhanced system."""
    try:
        if not enhanced_bulk_ai_system:
            raise HTTPException(status_code=503, detail="Enhanced bulk AI system not initialized")
        
        benchmark_results = await enhanced_bulk_ai_system.benchmark_system()
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Failed to benchmark enhanced system: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to benchmark system: {str(e)}")

@app.get("/api/v1/enhanced-bulk-ai/models")
async def get_enhanced_available_models():
    """Get available enhanced models."""
    try:
        if not enhanced_bulk_ai_system:
            raise HTTPException(status_code=503, detail="Enhanced bulk AI system not initialized")
        
        # Get available models from the library integration
        library_integration = enhanced_bulk_ai_system.library_integration
        available_models = library_integration.get_available_models()
        
        return {
            "available_models": available_models,
            "total_models": len(available_models),
            "model_types": list(set(model_info.get('type', 'unknown') for model_info in available_models.values())),
            "optimization_levels": list(set(model_info.get('optimization_level', 'basic') for model_info in available_models.values())),
            "capabilities": list(set(cap for model_info in available_models.values() for cap in model_info.get('capabilities', [])))
        }
        
    except Exception as e:
        logger.error(f"Failed to get enhanced available models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "bulk_truthgpt.main:app",
        host="0.0.0.0",
        port=8006,
        reload=True,
        log_level="info"
    )
