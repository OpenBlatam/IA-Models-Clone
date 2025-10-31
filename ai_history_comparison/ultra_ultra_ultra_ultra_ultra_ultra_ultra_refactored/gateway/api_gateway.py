"""
API Gateway - Gateway de API
===========================

Gateway de API que orquesta todos los microservicios del sistema ultra refactorizado.
"""

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging
import httpx
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

from ..shared.models import HistoryEntry, ComparisonResult, QualityReport, AnalysisJob, SystemMetrics
from ..shared.config import Settings
from ..shared.messaging import MessageBroker
from ..shared.monitoring import MetricsCollector
from ..shared.circuit_breaker import CircuitBreaker

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración
settings = Settings()

# Servicios compartidos
message_broker = MessageBroker()
metrics = MetricsCollector()

# Circuit breakers para cada microservicio
history_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30,
    expected_exception=httpx.HTTPError
)

comparison_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30,
    expected_exception=httpx.HTTPError
)

quality_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30,
    expected_exception=httpx.HTTPError
)

# Cliente HTTP asíncrono
http_client = httpx.AsyncClient(timeout=30.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida del API Gateway."""
    # Startup
    logger.info("Starting API Gateway...")
    await message_broker.initialize()
    await metrics.initialize()
    logger.info("API Gateway initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API Gateway...")
    await http_client.aclose()
    await message_broker.close()
    await metrics.close()
    logger.info("API Gateway shutdown complete")


# Crear aplicación FastAPI
app = FastAPI(
    title="AI History Comparison API Gateway",
    description="Gateway de API que orquesta todos los microservicios del sistema ultra refactorizado",
    version="7.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Middleware de logging y métricas
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Middleware para logging y métricas."""
    start_time = datetime.utcnow()
    
    # Log de request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Procesar request
    response = await call_next(request)
    
    # Calcular tiempo de procesamiento
    process_time = (datetime.utcnow() - start_time).total_seconds()
    
    # Métricas
    await metrics.increment_counter("api_requests_total")
    await metrics.record_histogram("api_request_duration", process_time)
    
    # Log de response
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response


# Endpoints principales
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Endpoint raíz con información del sistema."""
    return {
        "message": "AI History Comparison API Gateway",
        "version": "7.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "history": "http://localhost:8001",
            "comparison": "http://localhost:8002",
            "quality": "http://localhost:8003",
            "analytics": "http://localhost:8004"
        },
        "endpoints": {
            "history": "/history",
            "comparisons": "/comparisons",
            "quality": "/quality",
            "analytics": "/analytics",
            "health": "/health"
        }
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Verificación de salud del sistema completo."""
    try:
        # Verificar salud de todos los microservicios
        services_health = await check_all_services_health()
        
        # Determinar estado general
        all_healthy = all(service["status"] == "healthy" for service in services_health.values())
        
        return {
            "gateway": "healthy",
            "overall_status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "services": services_health,
            "circuit_breakers": {
                "history": history_circuit_breaker.state,
                "comparison": comparison_circuit_breaker.state,
                "quality": quality_circuit_breaker.state
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


# Endpoints de historial (proxy al microservicio de historial)
@app.post("/history", response_model=HistoryEntry, status_code=201)
async def create_history_entry(entry: HistoryEntry):
    """Crear una nueva entrada de historial."""
    try:
        async with history_circuit_breaker:
            response = await http_client.post(
                "http://localhost:8001/entries",
                json=entry.model_dump()
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error creating history entry: {e}")
        await metrics.increment_counter("gateway_errors")
        raise HTTPException(status_code=500, detail="Failed to create history entry")


@app.get("/history", response_model=List[HistoryEntry])
async def get_history_entries(
    skip: int = 0,
    limit: int = 100,
    model_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    search_query: Optional[str] = None
):
    """Obtener entradas de historial con filtros."""
    try:
        params = {
            "skip": skip,
            "limit": limit
        }
        if model_type:
            params["model_type"] = model_type
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        if search_query:
            params["search_query"] = search_query
        
        async with history_circuit_breaker:
            response = await http_client.get(
                "http://localhost:8001/entries",
                params=params
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error getting history entries: {e}")
        await metrics.increment_counter("gateway_errors")
        raise HTTPException(status_code=500, detail="Failed to retrieve history entries")


@app.get("/history/{entry_id}", response_model=HistoryEntry)
async def get_history_entry(entry_id: str):
    """Obtener una entrada de historial específica."""
    try:
        async with history_circuit_breaker:
            response = await http_client.get(f"http://localhost:8001/entries/{entry_id}")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="History entry not found")
        raise HTTPException(status_code=500, detail="Failed to retrieve history entry")
    except Exception as e:
        logger.error(f"Error getting history entry {entry_id}: {e}")
        await metrics.increment_counter("gateway_errors")
        raise HTTPException(status_code=500, detail="Failed to retrieve history entry")


# Endpoints de comparación (proxy al microservicio de comparación)
@app.post("/comparisons", response_model=ComparisonResult, status_code=201)
async def create_comparison(
    entry_1_id: str,
    entry_2_id: str,
    comparison_type: str = "comprehensive"
):
    """Crear una comparación entre dos entradas."""
    try:
        async with comparison_circuit_breaker:
            response = await http_client.post(
                "http://localhost:8002/compare",
                params={
                    "entry_1_id": entry_1_id,
                    "entry_2_id": entry_2_id,
                    "comparison_type": comparison_type
                }
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="One or both history entries not found")
        raise HTTPException(status_code=500, detail="Failed to create comparison")
    except Exception as e:
        logger.error(f"Error creating comparison: {e}")
        await metrics.increment_counter("gateway_errors")
        raise HTTPException(status_code=500, detail="Failed to create comparison")


@app.get("/comparisons", response_model=List[ComparisonResult])
async def get_comparisons(
    skip: int = 0,
    limit: int = 100,
    entry_1_id: Optional[str] = None,
    entry_2_id: Optional[str] = None,
    min_similarity: Optional[float] = None,
    max_similarity: Optional[float] = None
):
    """Obtener comparaciones con filtros."""
    try:
        params = {
            "skip": skip,
            "limit": limit
        }
        if entry_1_id:
            params["entry_1_id"] = entry_1_id
        if entry_2_id:
            params["entry_2_id"] = entry_2_id
        if min_similarity is not None:
            params["min_similarity"] = min_similarity
        if max_similarity is not None:
            params["max_similarity"] = max_similarity
        
        async with comparison_circuit_breaker:
            response = await http_client.get(
                "http://localhost:8002/comparisons",
                params=params
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error getting comparisons: {e}")
        await metrics.increment_counter("gateway_errors")
        raise HTTPException(status_code=500, detail="Failed to retrieve comparisons")


@app.get("/comparisons/{comparison_id}", response_model=ComparisonResult)
async def get_comparison(comparison_id: str):
    """Obtener una comparación específica."""
    try:
        async with comparison_circuit_breaker:
            response = await http_client.get(f"http://localhost:8002/comparisons/{comparison_id}")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Comparison not found")
        raise HTTPException(status_code=500, detail="Failed to retrieve comparison")
    except Exception as e:
        logger.error(f"Error getting comparison {comparison_id}: {e}")
        await metrics.increment_counter("gateway_errors")
        raise HTTPException(status_code=500, detail="Failed to retrieve comparison")


# Endpoints de calidad (proxy al microservicio de calidad)
@app.post("/quality", response_model=QualityReport, status_code=201)
async def create_quality_report(entry_id: str):
    """Crear un reporte de calidad para una entrada."""
    try:
        async with quality_circuit_breaker:
            response = await http_client.post(
                "http://localhost:8003/quality",
                params={"entry_id": entry_id}
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="History entry not found")
        raise HTTPException(status_code=500, detail="Failed to create quality report")
    except Exception as e:
        logger.error(f"Error creating quality report: {e}")
        await metrics.increment_counter("gateway_errors")
        raise HTTPException(status_code=500, detail="Failed to create quality report")


@app.get("/quality", response_model=List[QualityReport])
async def get_quality_reports(
    skip: int = 0,
    limit: int = 100,
    entry_id: Optional[str] = None,
    min_quality: Optional[float] = None
):
    """Obtener reportes de calidad con filtros."""
    try:
        params = {
            "skip": skip,
            "limit": limit
        }
        if entry_id:
            params["entry_id"] = entry_id
        if min_quality is not None:
            params["min_quality"] = min_quality
        
        async with quality_circuit_breaker:
            response = await http_client.get(
                "http://localhost:8003/quality",
                params=params
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Error getting quality reports: {e}")
        await metrics.increment_counter("gateway_errors")
        raise HTTPException(status_code=500, detail="Failed to retrieve quality reports")


# Endpoints de analytics (orquestación de múltiples microservicios)
@app.get("/analytics/overview")
async def get_analytics_overview():
    """Obtener resumen de analytics del sistema."""
    try:
        # Obtener métricas de todos los microservicios
        analytics = await gather_analytics_from_all_services()
        
        return {
            "overview": analytics,
            "generated_at": datetime.utcnow().isoformat(),
            "services_queried": ["history", "comparison", "quality"]
        }
    except Exception as e:
        logger.error(f"Error getting analytics overview: {e}")
        await metrics.increment_counter("gateway_errors")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics overview")


@app.get("/analytics/trends")
async def get_analytics_trends(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    model_type: Optional[str] = None
):
    """Obtener análisis de tendencias."""
    try:
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Obtener tendencias de todos los microservicios
        trends = await gather_trends_from_all_services(start_date, end_date, model_type)
        
        return {
            "trends": trends,
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting analytics trends: {e}")
        await metrics.increment_counter("gateway_errors")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics trends")


# Endpoints de trabajos (orquestación)
@app.post("/jobs", response_model=AnalysisJob, status_code=201)
async def create_analysis_job(
    job: AnalysisJob,
    background_tasks: BackgroundTasks
):
    """Crear un trabajo de análisis que orquesta múltiples microservicios."""
    try:
        # Crear trabajo en el sistema
        job_id = f"job_{datetime.utcnow().timestamp()}"
        job.id = job_id
        job.status = "pending"
        
        # Procesar trabajo en segundo plano
        background_tasks.add_task(process_analysis_job, job)
        
        # Enviar evento de trabajo creado
        await message_broker.publish_event("job.created", {
            "job_id": job_id,
            "job_type": job.job_type,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Created analysis job: {job_id}")
        return job
        
    except Exception as e:
        logger.error(f"Error creating analysis job: {e}")
        await metrics.increment_counter("gateway_errors")
        raise HTTPException(status_code=500, detail="Failed to create analysis job")


# Funciones auxiliares
async def check_all_services_health() -> Dict[str, Dict[str, Any]]:
    """Verificar salud de todos los microservicios."""
    services = {
        "history": "http://localhost:8001/health",
        "comparison": "http://localhost:8002/health",
        "quality": "http://localhost:8003/health"
    }
    
    health_status = {}
    
    for service_name, health_url in services.items():
        try:
            response = await http_client.get(health_url, timeout=5.0)
            if response.status_code == 200:
                health_status[service_name] = response.json()
            else:
                health_status[service_name] = {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            health_status[service_name] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    return health_status


async def gather_analytics_from_all_services() -> Dict[str, Any]:
    """Recopilar analytics de todos los microservicios."""
    analytics = {
        "history": {},
        "comparison": {},
        "quality": {}
    }
    
    try:
        # Obtener analytics del microservicio de historial
        response = await http_client.get("http://localhost:8001/analytics")
        if response.status_code == 200:
            analytics["history"] = response.json()
    except Exception as e:
        logger.error(f"Error getting history analytics: {e}")
    
    try:
        # Obtener analytics del microservicio de comparación
        response = await http_client.get("http://localhost:8002/statistics/similarity")
        if response.status_code == 200:
            analytics["comparison"] = response.json()
    except Exception as e:
        logger.error(f"Error getting comparison analytics: {e}")
    
    try:
        # Obtener analytics del microservicio de calidad
        response = await http_client.get("http://localhost:8003/statistics/quality")
        if response.status_code == 200:
            analytics["quality"] = response.json()
    except Exception as e:
        logger.error(f"Error getting quality analytics: {e}")
    
    return analytics


async def gather_trends_from_all_services(
    start_date: datetime, 
    end_date: datetime, 
    model_type: Optional[str]
) -> Dict[str, Any]:
    """Recopilar tendencias de todos los microservicios."""
    trends = {
        "history": {},
        "comparison": {},
        "quality": {}
    }
    
    params = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat()
    }
    if model_type:
        params["model_type"] = model_type
    
    try:
        # Obtener tendencias del microservicio de historial
        response = await http_client.get(
            "http://localhost:8001/trends",
            params=params
        )
        if response.status_code == 200:
            trends["history"] = response.json()
    except Exception as e:
        logger.error(f"Error getting history trends: {e}")
    
    try:
        # Obtener tendencias del microservicio de comparación
        response = await http_client.get(
            "http://localhost:8002/trends/similarity",
            params=params
        )
        if response.status_code == 200:
            trends["comparison"] = response.json()
    except Exception as e:
        logger.error(f"Error getting comparison trends: {e}")
    
    try:
        # Obtener tendencias del microservicio de calidad
        response = await http_client.get(
            "http://localhost:8003/trends/quality",
            params=params
        )
        if response.status_code == 200:
            trends["quality"] = response.json()
    except Exception as e:
        logger.error(f"Error getting quality trends: {e}")
    
    return trends


async def process_analysis_job(job: AnalysisJob):
    """Procesar trabajo de análisis orquestando múltiples microservicios."""
    try:
        # Actualizar estado a procesando
        job.status = "processing"
        
        # Procesar según tipo de trabajo
        if job.job_type == "comprehensive_analysis":
            await process_comprehensive_analysis_job(job)
        elif job.job_type == "batch_comparison":
            await process_batch_comparison_job(job)
        elif job.job_type == "quality_assessment":
            await process_quality_assessment_job(job)
        
        # Actualizar estado a completado
        job.status = "completed"
        
        # Enviar evento de trabajo completado
        await message_broker.publish_event("job.completed", {
            "job_id": job.id,
            "job_type": job.job_type,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Analysis job completed: {job.id}")
        
    except Exception as e:
        logger.error(f"Error processing analysis job {job.id}: {e}")
        job.status = "failed"
        job.error_message = str(e)
        
        # Enviar evento de trabajo fallido
        await message_broker.publish_event("job.failed", {
            "job_id": job.id,
            "job_type": job.job_type,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })


async def process_comprehensive_analysis_job(job: AnalysisJob):
    """Procesar trabajo de análisis comprensivo."""
    # Implementar lógica de análisis comprensivo
    pass


async def process_batch_comparison_job(job: AnalysisJob):
    """Procesar trabajo de comparación en lote."""
    # Implementar lógica de comparación en lote
    pass


async def process_quality_assessment_job(job: AnalysisJob):
    """Procesar trabajo de evaluación de calidad."""
    # Implementar lógica de evaluación de calidad
    pass


# Manejo de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejador global de excepciones."""
    logger.error(f"Unhandled exception: {exc}")
    await metrics.increment_counter("gateway_unhandled_exceptions")
    return {"detail": "Internal server error"}


if __name__ == "__main__":
    uvicorn.run(
        "api_gateway:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )




