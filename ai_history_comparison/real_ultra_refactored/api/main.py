"""
Main API - API Principal
=======================

API principal del sistema ultra refactorizado real con FastAPI.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..core.models import (
    HistoryEntry, ComparisonResult, QualityReport, AnalysisJob, 
    TrendAnalysis, SystemMetrics, ModelType, AnalysisStatus
)
from ..services.analysis_service import ComparisonService, QualityAssessmentService, ContentAnalyzer
from ..database.database import DatabaseManager
from ..config.settings import Settings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración
settings = Settings()

# Servicios
db_manager = DatabaseManager()
comparison_service = ComparisonService()
quality_service = QualityAssessmentService()
content_analyzer = ContentAnalyzer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación."""
    # Startup
    logger.info("Starting AI History Comparison API...")
    await db_manager.initialize()
    logger.info("Database initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI History Comparison API...")
    await db_manager.close()
    logger.info("Database connection closed")


# Crear aplicación FastAPI
app = FastAPI(
    title="AI History Comparison API",
    description="API ultra refactorizada para comparación y análisis de historial de IA",
    version="1.0.0",
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


# Endpoints principales
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Endpoint raíz con información del sistema."""
    return {
        "message": "AI History Comparison API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "history": "/history",
            "comparisons": "/comparisons",
            "quality": "/quality",
            "jobs": "/jobs",
            "trends": "/trends",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Verificación de salud del sistema."""
    try:
        # Verificar conexión a base de datos
        db_status = await db_manager.health_check()
        
        return {
            "status": "healthy" if db_status else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected" if db_status else "disconnected",
            "services": {
                "comparison_service": "operational",
                "quality_service": "operational",
                "content_analyzer": "operational"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


# Endpoints de historial
@app.post("/history", response_model=HistoryEntry, status_code=201)
async def create_history_entry(entry: HistoryEntry):
    """Crear una nueva entrada de historial."""
    try:
        # Analizar contenido automáticamente
        analysis = content_analyzer.analyze_content(entry.response)
        entry.metadata.update(analysis)
        
        # Guardar en base de datos
        saved_entry = await db_manager.create_history_entry(entry)
        
        logger.info(f"Created history entry: {saved_entry.id}")
        return saved_entry
    except Exception as e:
        logger.error(f"Error creating history entry: {e}")
        raise HTTPException(status_code=500, detail="Failed to create history entry")


@app.get("/history", response_model=List[HistoryEntry])
async def get_history_entries(
    skip: int = 0,
    limit: int = 100,
    model_type: Optional[ModelType] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """Obtener entradas de historial con filtros."""
    try:
        entries = await db_manager.get_history_entries(
            skip=skip,
            limit=limit,
            model_type=model_type,
            start_date=start_date,
            end_date=end_date
        )
        return entries
    except Exception as e:
        logger.error(f"Error getting history entries: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history entries")


@app.get("/history/{entry_id}", response_model=HistoryEntry)
async def get_history_entry(entry_id: str):
    """Obtener una entrada de historial específica."""
    try:
        entry = await db_manager.get_history_entry(entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail="History entry not found")
        return entry
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting history entry {entry_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history entry")


@app.delete("/history/{entry_id}")
async def delete_history_entry(entry_id: str):
    """Eliminar una entrada de historial."""
    try:
        success = await db_manager.delete_history_entry(entry_id)
        if not success:
            raise HTTPException(status_code=404, detail="History entry not found")
        
        logger.info(f"Deleted history entry: {entry_id}")
        return {"message": "History entry deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting history entry {entry_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete history entry")


# Endpoints de comparación
@app.post("/comparisons", response_model=ComparisonResult, status_code=201)
async def create_comparison(entry_1_id: str, entry_2_id: str):
    """Crear una comparación entre dos entradas."""
    try:
        # Obtener entradas
        entry1 = await db_manager.get_history_entry(entry_1_id)
        entry2 = await db_manager.get_history_entry(entry_2_id)
        
        if not entry1 or not entry2:
            raise HTTPException(status_code=404, detail="One or both history entries not found")
        
        # Realizar comparación
        comparison = comparison_service.compare_entries(entry1, entry2)
        
        # Guardar comparación
        saved_comparison = await db_manager.create_comparison(comparison)
        
        logger.info(f"Created comparison: {saved_comparison.id}")
        return saved_comparison
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating comparison: {e}")
        raise HTTPException(status_code=500, detail="Failed to create comparison")


@app.get("/comparisons", response_model=List[ComparisonResult])
async def get_comparisons(
    skip: int = 0,
    limit: int = 100,
    entry_1_id: Optional[str] = None,
    entry_2_id: Optional[str] = None
):
    """Obtener comparaciones con filtros."""
    try:
        comparisons = await db_manager.get_comparisons(
            skip=skip,
            limit=limit,
            entry_1_id=entry_1_id,
            entry_2_id=entry_2_id
        )
        return comparisons
    except Exception as e:
        logger.error(f"Error getting comparisons: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve comparisons")


@app.get("/comparisons/{comparison_id}", response_model=ComparisonResult)
async def get_comparison(comparison_id: str):
    """Obtener una comparación específica."""
    try:
        comparison = await db_manager.get_comparison(comparison_id)
        if not comparison:
            raise HTTPException(status_code=404, detail="Comparison not found")
        return comparison
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting comparison {comparison_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve comparison")


# Endpoints de calidad
@app.post("/quality", response_model=QualityReport, status_code=201)
async def create_quality_report(entry_id: str):
    """Crear un reporte de calidad para una entrada."""
    try:
        # Obtener entrada
        entry = await db_manager.get_history_entry(entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail="History entry not found")
        
        # Realizar evaluación de calidad
        quality_report = quality_service.assess_quality(entry)
        
        # Guardar reporte
        saved_report = await db_manager.create_quality_report(quality_report)
        
        logger.info(f"Created quality report: {saved_report.id}")
        return saved_report
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating quality report: {e}")
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
        reports = await db_manager.get_quality_reports(
            skip=skip,
            limit=limit,
            entry_id=entry_id,
            min_quality=min_quality
        )
        return reports
    except Exception as e:
        logger.error(f"Error getting quality reports: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve quality reports")


@app.get("/quality/{report_id}", response_model=QualityReport)
async def get_quality_report(report_id: str):
    """Obtener un reporte de calidad específico."""
    try:
        report = await db_manager.get_quality_report(report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Quality report not found")
        return report
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quality report {report_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve quality report")


# Endpoints de trabajos
@app.post("/jobs", response_model=AnalysisJob, status_code=201)
async def create_analysis_job(
    job: AnalysisJob,
    background_tasks: BackgroundTasks
):
    """Crear un trabajo de análisis."""
    try:
        # Guardar trabajo
        saved_job = await db_manager.create_analysis_job(job)
        
        # Procesar trabajo en segundo plano
        background_tasks.add_task(process_analysis_job, saved_job.id)
        
        logger.info(f"Created analysis job: {saved_job.id}")
        return saved_job
    except Exception as e:
        logger.error(f"Error creating analysis job: {e}")
        raise HTTPException(status_code=500, detail="Failed to create analysis job")


@app.get("/jobs", response_model=List[AnalysisJob])
async def get_analysis_jobs(
    skip: int = 0,
    limit: int = 100,
    status: Optional[AnalysisStatus] = None,
    job_type: Optional[str] = None
):
    """Obtener trabajos de análisis con filtros."""
    try:
        jobs = await db_manager.get_analysis_jobs(
            skip=skip,
            limit=limit,
            status=status,
            job_type=job_type
        )
        return jobs
    except Exception as e:
        logger.error(f"Error getting analysis jobs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis jobs")


@app.get("/jobs/{job_id}", response_model=AnalysisJob)
async def get_analysis_job(job_id: str):
    """Obtener un trabajo de análisis específico."""
    try:
        job = await db_manager.get_analysis_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Analysis job not found")
        return job
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis job")


# Endpoints de métricas
@app.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics():
    """Obtener métricas del sistema."""
    try:
        metrics = await db_manager.get_system_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system metrics")


@app.get("/trends", response_model=List[TrendAnalysis])
async def get_trend_analysis(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    model_type: Optional[ModelType] = None
):
    """Obtener análisis de tendencias."""
    try:
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        trends = await db_manager.get_trend_analysis(
            start_date=start_date,
            end_date=end_date,
            model_type=model_type
        )
        return trends
    except Exception as e:
        logger.error(f"Error getting trend analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve trend analysis")


# Funciones auxiliares
async def process_analysis_job(job_id: str):
    """Procesar un trabajo de análisis en segundo plano."""
    try:
        # Actualizar estado a procesando
        await db_manager.update_job_status(job_id, AnalysisStatus.PROCESSING)
        
        # Obtener trabajo
        job = await db_manager.get_analysis_job(job_id)
        if not job:
            return
        
        # Procesar según tipo de trabajo
        if job.job_type == "comparison":
            await process_comparison_job(job)
        elif job.job_type == "quality_assessment":
            await process_quality_job(job)
        elif job.job_type == "trend_analysis":
            await process_trend_job(job)
        
        # Actualizar estado a completado
        await db_manager.update_job_status(job_id, AnalysisStatus.COMPLETED)
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        await db_manager.update_job_status(job_id, AnalysisStatus.FAILED, str(e))


async def process_comparison_job(job: AnalysisJob):
    """Procesar trabajo de comparación."""
    # Implementar lógica de comparación masiva
    pass


async def process_quality_job(job: AnalysisJob):
    """Procesar trabajo de evaluación de calidad."""
    # Implementar lógica de evaluación masiva
    pass


async def process_trend_job(job: AnalysisJob):
    """Procesar trabajo de análisis de tendencias."""
    # Implementar lógica de análisis de tendencias
    pass


# Manejo de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejador global de excepciones."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )




