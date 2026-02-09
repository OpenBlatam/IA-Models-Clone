"""
Observability API Endpoints
===========================

Endpoints para observability y tracing.
"""

from fastapi import APIRouter, HTTPException, Query, Header
from typing import Dict, Any, Optional
import logging

from ..core.observability_system import get_observability_system
from ..core.tracing_system import get_tracing_system

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/observability", tags=["observability"])


@router.post("/traces")
async def start_trace(
    trace_id: str,
    operation: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Iniciar trace."""
    try:
        system = get_observability_system()
        trace = system.start_trace(trace_id, operation, metadata)
        return {
            "trace_id": trace.trace_id,
            "operation": trace.operation,
            "status": trace.status,
            "start_time": trace.start_time
        }
    except Exception as e:
        logger.error(f"Error starting trace: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/traces/{trace_id}/end")
async def end_trace(
    trace_id: str,
    status: str = "completed",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Finalizar trace."""
    try:
        system = get_observability_system()
        trace = system.end_trace(trace_id, status, metadata)
        
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        
        return {
            "trace_id": trace.trace_id,
            "operation": trace.operation,
            "status": trace.status,
            "duration": trace.duration
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending trace: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: str) -> Dict[str, Any]:
    """Obtener trace."""
    try:
        system = get_observability_system()
        trace = system.get_trace(trace_id)
        
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        
        return {
            "trace_id": trace.trace_id,
            "operation": trace.operation,
            "status": trace.status,
            "start_time": trace.start_time,
            "end_time": trace.end_time,
            "duration": trace.duration,
            "spans_count": len(trace.spans)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trace: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/traces")
async def list_traces(
    operation: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
) -> Dict[str, Any]:
    """Listar traces."""
    try:
        system = get_observability_system()
        traces = system.get_traces(operation=operation, limit=limit)
        return {
            "traces": [
                {
                    "trace_id": t.trace_id,
                    "operation": t.operation,
                    "status": t.status,
                    "duration": t.duration
                }
                for t in traces
            ],
            "count": len(traces)
        }
    except Exception as e:
        logger.error(f"Error listing traces: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_observability_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de observabilidad."""
    try:
        system = get_observability_system()
        stats = system.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting observability statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tracing/start")
async def start_tracing(
    operation: str,
    trace_id: Optional[str] = None
) -> Dict[str, Any]:
    """Iniciar tracing."""
    try:
        tracing = get_tracing_system()
        context = tracing.start_trace(operation, trace_id)
        return {
            "trace_id": context.trace_id,
            "span_id": context.span_id,
            "parent_span_id": context.parent_span_id
        }
    except Exception as e:
        logger.error(f"Error starting tracing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tracing/context")
async def get_tracing_context() -> Dict[str, Any]:
    """Obtener contexto de tracing actual."""
    try:
        tracing = get_tracing_system()
        context = tracing.get_current_context()
        
        if not context:
            return {"context": None}
        
        return {
            "trace_id": context.trace_id,
            "span_id": context.span_id,
            "parent_span_id": context.parent_span_id,
            "baggage": context.baggage
        }
    except Exception as e:
        logger.error(f"Error getting tracing context: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






