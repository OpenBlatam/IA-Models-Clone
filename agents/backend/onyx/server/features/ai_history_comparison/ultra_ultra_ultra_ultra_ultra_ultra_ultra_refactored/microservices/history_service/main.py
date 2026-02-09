"""
History Microservice - Microservicio de Historial
===============================================

Microservicio especializado en gestión de historial de IA con arquitectura avanzada.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import json

from ..shared.models import HistoryEntry, ModelType, AnalysisStatus
from ..shared.database import DatabaseManager
from ..shared.config import Settings
from ..shared.messaging import MessageBroker
from ..shared.monitoring import MetricsCollector

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración
settings = Settings()

# Servicios compartidos
db_manager = DatabaseManager()
message_broker = MessageBroker()
metrics = MetricsCollector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida del microservicio."""
    # Startup
    logger.info("Starting History Microservice...")
    await db_manager.initialize()
    await message_broker.initialize()
    await metrics.initialize()
    logger.info("History Microservice initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down History Microservice...")
    await db_manager.close()
    await message_broker.close()
    await metrics.close()
    logger.info("History Microservice shutdown complete")


# Crear aplicación FastAPI
app = FastAPI(
    title="History Microservice",
    description="Microservicio especializado en gestión de historial de IA",
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


# Endpoints del microservicio
@app.get("/health")
async def health_check():
    """Verificación de salud del microservicio."""
    try:
        db_health = await db_manager.health_check()
        broker_health = await message_broker.health_check()
        
        return {
            "service": "history",
            "status": "healthy" if db_health and broker_health else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected" if db_health else "disconnected",
            "message_broker": "connected" if broker_health else "disconnected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.post("/entries", response_model=HistoryEntry, status_code=201)
async def create_history_entry(
    entry: HistoryEntry,
    background_tasks: BackgroundTasks
):
    """Crear una nueva entrada de historial."""
    try:
        # Validar entrada
        if not entry.prompt or not entry.response:
            raise HTTPException(status_code=400, detail="Prompt and response are required")
        
        # Guardar en base de datos
        saved_entry = await db_manager.create_history_entry(entry)
        
        # Enviar evento de creación
        await message_broker.publish_event("history.entry.created", {
            "entry_id": saved_entry.id,
            "model_type": saved_entry.model_type.value,
            "timestamp": saved_entry.timestamp.isoformat()
        })
        
        # Procesar análisis en segundo plano
        background_tasks.add_task(analyze_entry_async, saved_entry.id)
        
        # Métricas
        await metrics.increment_counter("history_entries_created")
        await metrics.record_histogram("entry_response_time", entry.response_time_ms or 0)
        
        logger.info(f"Created history entry: {saved_entry.id}")
        return saved_entry
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating history entry: {e}")
        await metrics.increment_counter("history_entries_creation_errors")
        raise HTTPException(status_code=500, detail="Failed to create history entry")


@app.get("/entries", response_model=List[HistoryEntry])
async def get_history_entries(
    skip: int = 0,
    limit: int = 100,
    model_type: Optional[ModelType] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    search_query: Optional[str] = None
):
    """Obtener entradas de historial con filtros avanzados."""
    try:
        # Validar límites
        if limit > 1000:
            limit = 1000
        
        entries = await db_manager.get_history_entries(
            skip=skip,
            limit=limit,
            model_type=model_type,
            start_date=start_date,
            end_date=end_date,
            search_query=search_query
        )
        
        # Métricas
        await metrics.increment_counter("history_entries_retrieved")
        await metrics.record_histogram("entries_retrieved_count", len(entries))
        
        return entries
        
    except Exception as e:
        logger.error(f"Error getting history entries: {e}")
        await metrics.increment_counter("history_entries_retrieval_errors")
        raise HTTPException(status_code=500, detail="Failed to retrieve history entries")


@app.get("/entries/{entry_id}", response_model=HistoryEntry)
async def get_history_entry(entry_id: str):
    """Obtener una entrada de historial específica."""
    try:
        entry = await db_manager.get_history_entry(entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail="History entry not found")
        
        # Métricas
        await metrics.increment_counter("history_entry_retrieved")
        
        return entry
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting history entry {entry_id}: {e}")
        await metrics.increment_counter("history_entry_retrieval_errors")
        raise HTTPException(status_code=500, detail="Failed to retrieve history entry")


@app.put("/entries/{entry_id}", response_model=HistoryEntry)
async def update_history_entry(
    entry_id: str,
    updates: Dict[str, Any]
):
    """Actualizar una entrada de historial."""
    try:
        # Verificar que la entrada existe
        existing_entry = await db_manager.get_history_entry(entry_id)
        if not existing_entry:
            raise HTTPException(status_code=404, detail="History entry not found")
        
        # Actualizar entrada
        updated_entry = await db_manager.update_history_entry(entry_id, updates)
        
        # Enviar evento de actualización
        await message_broker.publish_event("history.entry.updated", {
            "entry_id": entry_id,
            "updated_fields": list(updates.keys()),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Métricas
        await metrics.increment_counter("history_entries_updated")
        
        logger.info(f"Updated history entry: {entry_id}")
        return updated_entry
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating history entry {entry_id}: {e}")
        await metrics.increment_counter("history_entries_update_errors")
        raise HTTPException(status_code=500, detail="Failed to update history entry")


@app.delete("/entries/{entry_id}")
async def delete_history_entry(entry_id: str):
    """Eliminar una entrada de historial."""
    try:
        # Verificar que la entrada existe
        existing_entry = await db_manager.get_history_entry(entry_id)
        if not existing_entry:
            raise HTTPException(status_code=404, detail="History entry not found")
        
        # Eliminar entrada
        success = await db_manager.delete_history_entry(entry_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete history entry")
        
        # Enviar evento de eliminación
        await message_broker.publish_event("history.entry.deleted", {
            "entry_id": entry_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Métricas
        await metrics.increment_counter("history_entries_deleted")
        
        logger.info(f"Deleted history entry: {entry_id}")
        return {"message": "History entry deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting history entry {entry_id}: {e}")
        await metrics.increment_counter("history_entries_deletion_errors")
        raise HTTPException(status_code=500, detail="Failed to delete history entry")


@app.get("/entries/{entry_id}/analytics")
async def get_entry_analytics(entry_id: str):
    """Obtener análisis detallado de una entrada."""
    try:
        entry = await db_manager.get_history_entry(entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail="History entry not found")
        
        # Generar análisis
        analytics = await generate_entry_analytics(entry)
        
        # Métricas
        await metrics.increment_counter("entry_analytics_generated")
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating analytics for entry {entry_id}: {e}")
        await metrics.increment_counter("entry_analytics_errors")
        raise HTTPException(status_code=500, detail="Failed to generate analytics")


@app.get("/entries/batch/analytics")
async def get_batch_analytics(
    entry_ids: List[str],
    analysis_type: str = "comprehensive"
):
    """Obtener análisis en lote para múltiples entradas."""
    try:
        if len(entry_ids) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 entries per batch")
        
        # Obtener entradas
        entries = []
        for entry_id in entry_ids:
            entry = await db_manager.get_history_entry(entry_id)
            if entry:
                entries.append(entry)
        
        if not entries:
            raise HTTPException(status_code=404, detail="No valid entries found")
        
        # Generar análisis en lote
        batch_analytics = await generate_batch_analytics(entries, analysis_type)
        
        # Métricas
        await metrics.increment_counter("batch_analytics_generated")
        await metrics.record_histogram("batch_analytics_size", len(entries))
        
        return batch_analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating batch analytics: {e}")
        await metrics.increment_counter("batch_analytics_errors")
        raise HTTPException(status_code=500, detail="Failed to generate batch analytics")


# Funciones auxiliares
async def analyze_entry_async(entry_id: str):
    """Analizar entrada de forma asíncrona."""
    try:
        # Obtener entrada
        entry = await db_manager.get_history_entry(entry_id)
        if not entry:
            return
        
        # Realizar análisis
        analysis = await perform_entry_analysis(entry)
        
        # Guardar análisis
        await db_manager.save_entry_analysis(entry_id, analysis)
        
        # Enviar evento de análisis completado
        await message_broker.publish_event("history.entry.analyzed", {
            "entry_id": entry_id,
            "analysis_type": "comprehensive",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Analysis completed for entry: {entry_id}")
        
    except Exception as e:
        logger.error(f"Error analyzing entry {entry_id}: {e}")
        await message_broker.publish_event("history.entry.analysis_failed", {
            "entry_id": entry_id,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })


async def perform_entry_analysis(entry: HistoryEntry) -> Dict[str, Any]:
    """Realizar análisis completo de una entrada."""
    # Implementar análisis real con tecnologías funcionales
    analysis = {
        "word_count": len(entry.response.split()),
        "character_count": len(entry.response),
        "sentence_count": len(entry.response.split('.')),
        "analysis_timestamp": datetime.utcnow().isoformat(),
        "model_type": entry.model_type.value,
        "response_time_ms": entry.response_time_ms,
        "token_count": entry.token_count,
        "cost_usd": entry.cost_usd
    }
    
    return analysis


async def generate_entry_analytics(entry: HistoryEntry) -> Dict[str, Any]:
    """Generar análisis detallado de una entrada."""
    analysis = await perform_entry_analysis(entry)
    
    # Agregar métricas adicionales
    analytics = {
        "entry_id": entry.id,
        "basic_metrics": analysis,
        "performance_metrics": {
            "response_time_ms": entry.response_time_ms,
            "tokens_per_second": entry.token_count / (entry.response_time_ms / 1000) if entry.response_time_ms else 0,
            "cost_per_token": entry.cost_usd / entry.token_count if entry.token_count else 0
        },
        "quality_metrics": {
            "coherence_score": entry.coherence_score,
            "relevance_score": entry.relevance_score,
            "creativity_score": entry.creativity_score,
            "accuracy_score": entry.accuracy_score
        },
        "generated_at": datetime.utcnow().isoformat()
    }
    
    return analytics


async def generate_batch_analytics(entries: List[HistoryEntry], analysis_type: str) -> Dict[str, Any]:
    """Generar análisis en lote para múltiples entradas."""
    batch_analytics = {
        "analysis_type": analysis_type,
        "total_entries": len(entries),
        "generated_at": datetime.utcnow().isoformat(),
        "entries": []
    }
    
    for entry in entries:
        entry_analytics = await generate_entry_analytics(entry)
        batch_analytics["entries"].append(entry_analytics)
    
    # Calcular métricas agregadas
    if entries:
        batch_analytics["aggregate_metrics"] = {
            "average_response_time": sum(e.response_time_ms or 0 for e in entries) / len(entries),
            "total_tokens": sum(e.token_count or 0 for e in entries),
            "total_cost": sum(e.cost_usd or 0 for e in entries),
            "average_coherence": sum(e.coherence_score or 0 for e in entries) / len(entries),
            "average_relevance": sum(e.relevance_score or 0 for e in entries) / len(entries),
            "average_creativity": sum(e.creativity_score or 0 for e in entries) / len(entries),
            "average_accuracy": sum(e.accuracy_score or 0 for e in entries) / len(entries)
        }
    
    return batch_analytics


# Manejo de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejador global de excepciones."""
    logger.error(f"Unhandled exception: {exc}")
    await metrics.increment_counter("unhandled_exceptions")
    return {"detail": "Internal server error"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )




