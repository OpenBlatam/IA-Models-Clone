"""
Agent Routes - Rutas para control del agente.
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Optional, Dict, Any

from api.schemas import AgentStatusResponse, AgentControlRequest, AgentMetricsResponse, WorkerMetricsResponse
from api.dependencies import get_storage
from api.utils import handle_api_errors
from core.storage import TaskStorage
from core.constants import SuccessMessages
from config.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _create_agent_state(is_running: bool, current_task_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Crear diccionario de estado del agente.
    
    Args:
        is_running: Si el agente está corriendo
        current_task_id: ID de la tarea actual (opcional)
        
    Returns:
        Diccionario con el estado del agente
    """
    return {
        "id": "main",
        "is_running": is_running,
        "current_task_id": current_task_id,
        "last_activity": datetime.now().isoformat(),
        "metadata": {}
    }


@router.get("/status", response_model=AgentStatusResponse)
@handle_api_errors
async def get_agent_status(storage: TaskStorage = Depends(get_storage)):
    """Obtener estado del agente."""
    state = await storage.get_agent_state()
    return AgentStatusResponse(
        is_running=state.get("is_running", False),
        current_task_id=state.get("current_task_id"),
        last_activity=state.get("last_activity"),
        metadata=state.get("metadata", {})
    )


@router.post("/start")
@handle_api_errors
async def start_agent(request: Request, storage: TaskStorage = Depends(get_storage)):
    """Iniciar el agente."""
    await storage.update_agent_state(_create_agent_state(is_running=True))
    
    if hasattr(request.app.state, "worker_manager"):
        worker_manager = request.app.state.worker_manager
        if not worker_manager.is_running:
            await worker_manager.start()
    
    # Broadcast WebSocket update
    try:
        from api.routes.websocket_routes import broadcast_agent_status
        agent_state = await storage.get_agent_state()
        await broadcast_agent_status(agent_state)
    except Exception as e:
        logger.debug(f"WebSocket broadcast failed (non-critical): {e}")
    
    return {"message": SuccessMessages.AGENT_STARTED, "status": "running"}


@router.post("/stop")
@handle_api_errors
async def stop_agent(request: Request, storage: TaskStorage = Depends(get_storage)):
    """Detener el agente."""
    await storage.update_agent_state({
        "id": "main",
        "is_running": False,
        "current_task_id": None,
        "last_activity": datetime.now().isoformat(),
        "metadata": {}
    })
    
    if hasattr(request.app.state, "worker_manager"):
        worker_manager = request.app.state.worker_manager
        if worker_manager.is_running:
            await worker_manager.stop()
    
    # Broadcast WebSocket update
    try:
        from api.routes.websocket_routes import broadcast_agent_status
        agent_state = await storage.get_agent_state()
        await broadcast_agent_status(agent_state)
    except Exception as e:
        logger.debug(f"WebSocket broadcast failed (non-critical): {e}")
    
    return {"message": "Agente detenido", "status": "stopped"}


@router.post("/pause")
@handle_api_errors
async def pause_agent(storage: TaskStorage = Depends(get_storage)):
    """Pausar el agente."""
    await storage.update_agent_state({
        "id": "main",
        "is_running": False,
        "current_task_id": None,
        "last_activity": datetime.now().isoformat(),
        "metadata": {}
    })
    
    # Broadcast WebSocket update
    try:
        from api.routes.websocket_routes import broadcast_agent_status
        agent_state = await storage.get_agent_state()
        await broadcast_agent_status(agent_state)
    except Exception as e:
        logger.debug(f"WebSocket broadcast failed (non-critical): {e}")
    
    return {"message": "Agente pausado", "status": "paused"}


@router.post("/resume")
@handle_api_errors
async def resume_agent(storage: TaskStorage = Depends(get_storage)):
    """Reanudar el agente."""
    await storage.update_agent_state({
        "id": "main",
        "is_running": True,
        "current_task_id": None,
        "last_activity": datetime.now().isoformat(),
        "metadata": {}
    })
    
    # Broadcast WebSocket update
    try:
        from api.routes.websocket_routes import broadcast_agent_status
        agent_state = await storage.get_agent_state()
        await broadcast_agent_status(agent_state)
    except Exception as e:
        logger.debug(f"WebSocket broadcast failed (non-critical): {e}")
    
    return {"message": "Agente reanudado", "status": "running"}


@router.get("/metrics", response_model=AgentMetricsResponse)
@handle_api_errors
async def get_agent_metrics(
    request: Request,
    storage: TaskStorage = Depends(get_storage)
):
    """
    Obtener métricas completas del agente y worker.
    
    Returns:
        AgentMetricsResponse: Métricas del worker, estado del agente y estadísticas de tareas
        
    Raises:
        HTTPException: Si el worker manager no está disponible o hay errores al obtener métricas
    """
    try:
        # Validar que el worker manager esté disponible
        if not hasattr(request.app.state, "worker_manager"):
            logger.warning("Intento de obtener métricas sin worker manager inicializado")
            raise HTTPException(
                status_code=503,
                detail="Worker manager no está disponible. El servicio puede no estar completamente inicializado."
            )
        
        worker_manager = request.app.state.worker_manager
        
        # Obtener métricas del worker con validación
        try:
            worker_metrics_raw = worker_manager.get_metrics()
            if not worker_metrics_raw or not isinstance(worker_metrics_raw, dict):
                logger.warning("Worker metrics retornó datos inválidos")
                worker_metrics_raw = {}
        except Exception as e:
            logger.error(f"Error al obtener métricas del worker: {e}", exc_info=True)
            worker_metrics_raw = {
                "tasks_processed": 0,
                "tasks_succeeded": 0,
                "tasks_failed": 0,
                "last_task_time": None,
                "average_task_duration": 0.0,
                "circuit_state": "unknown",
                "consecutive_failures": 0,
                "is_running": False
            }
        
        # Obtener estado del agente
        try:
            agent_state_raw = await storage.get_agent_state()
        except Exception as e:
            logger.error(f"Error al obtener estado del agente: {e}", exc_info=True)
            agent_state_raw = {
                "is_running": False,
                "current_task_id": None,
                "last_activity": None,
                "metadata": {}
            }
        
        # Obtener estadísticas adicionales de tareas
        task_statistics = None
        try:
            # Obtener conteos de tareas por estado (límite alto para estadísticas)
            all_tasks = await storage.get_tasks(status=None, limit=10000)
            if all_tasks:
                task_stats = {
                    "total": len(all_tasks),
                    "by_status": {}
                }
                
                for task in all_tasks:
                    status = task.get("status", "unknown")
                    task_stats["by_status"][status] = task_stats["by_status"].get(status, 0) + 1
                
                # Calcular porcentajes
                if task_stats["total"] > 0:
                    task_stats["by_status_percentage"] = {
                        status: round((count / task_stats["total"]) * 100, 2)
                        for status, count in task_stats["by_status"].items()
                    }
                
                task_statistics = task_stats
        except Exception as e:
            logger.warning(f"No se pudieron obtener estadísticas de tareas: {e}")
            # No fallar si no se pueden obtener estadísticas
        
        # Construir respuesta estructurada
        worker_metrics = WorkerMetricsResponse(
            tasks_processed=worker_metrics_raw.get("tasks_processed", 0),
            tasks_succeeded=worker_metrics_raw.get("tasks_succeeded", 0),
            tasks_failed=worker_metrics_raw.get("tasks_failed", 0),
            last_task_time=worker_metrics_raw.get("last_task_time"),
            average_task_duration=worker_metrics_raw.get("average_task_duration", 0.0),
            circuit_state=worker_metrics_raw.get("circuit_state", "unknown"),
            consecutive_failures=worker_metrics_raw.get("consecutive_failures", 0),
            is_running=worker_metrics_raw.get("is_running", False)
        )
        
        agent_state = AgentStatusResponse(
            is_running=agent_state_raw.get("is_running", False),
            current_task_id=agent_state_raw.get("current_task_id"),
            last_activity=agent_state_raw.get("last_activity"),
            metadata=agent_state_raw.get("metadata", {})
        )
        
        logger.debug(f"Métricas obtenidas exitosamente: {worker_metrics.tasks_processed} tareas procesadas")
        
        return AgentMetricsResponse(
            worker_metrics=worker_metrics,
            agent_state=agent_state,
            task_statistics=task_statistics,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inesperado al obtener métricas: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener métricas: {str(e)}"
        )

