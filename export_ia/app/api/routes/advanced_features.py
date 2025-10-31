"""
Advanced Features API Routes - Rutas API para funcionalidades avanzadas
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from ..security.enhanced_security import EnhancedSecurity, SecurityLevel, ThreatType
from ..automation.workflow_automation import WorkflowAutomation, TriggerType, ActionType, WorkflowStatus
from ..data.advanced_data_manager import AdvancedDataManager, DataType, StorageType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/advanced", tags=["Advanced Features"])

# Instancias globales (en producción usar inyección de dependencias)
security_manager = EnhancedSecurity(SecurityLevel.HIGH)
automation_manager = WorkflowAutomation()
data_manager = AdvancedDataManager()


# Modelos Pydantic
class SecurityEventRequest(BaseModel):
    threat_type: str
    severity: str
    source_ip: str
    endpoint: str
    details: Dict[str, Any] = Field(default_factory=dict)


class WorkflowCreateRequest(BaseModel):
    name: str
    description: str
    triggers: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    created_by: str
    tags: List[str] = Field(default_factory=list)


class WorkflowExecuteRequest(BaseModel):
    workflow_id: str
    triggered_by: str = "manual"
    input_data: Dict[str, Any] = Field(default_factory=dict)


class DataStoreRequest(BaseModel):
    key: str
    value: Any
    data_type: str = "text"
    storage_type: str = "memory"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    expires_in_hours: Optional[int] = None


class DataSearchRequest(BaseModel):
    query: str
    data_type: Optional[str] = None
    storage_type: Optional[str] = None
    limit: int = 100


# Rutas de Seguridad
@router.post("/security/events")
async def log_security_event(request: SecurityEventRequest):
    """Registrar evento de seguridad."""
    try:
        await security_manager._log_security_event(
            threat_type=ThreatType(request.threat_type),
            severity=request.severity,
            source_ip=request.source_ip,
            endpoint=request.endpoint,
            details=request.details
        )
        
        return {
            "success": True,
            "message": "Evento de seguridad registrado",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al registrar evento de seguridad: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/security/events")
async def get_security_events(
    threat_type: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100
):
    """Obtener eventos de seguridad."""
    try:
        events = await security_manager.get_security_events(
            threat_type=ThreatType(threat_type) if threat_type else None,
            severity=severity,
            limit=limit
        )
        
        return {
            "events": events,
            "count": len(events),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener eventos de seguridad: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/security/block-ip")
async def block_ip(ip_address: str, duration_minutes: int = 60):
    """Bloquear dirección IP."""
    try:
        await security_manager.block_ip(ip_address, duration_minutes)
        
        return {
            "success": True,
            "message": f"IP {ip_address} bloqueada por {duration_minutes} minutos",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al bloquear IP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/security/stats")
async def get_security_stats():
    """Obtener estadísticas de seguridad."""
    try:
        stats = await security_manager.get_security_stats()
        
        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener estadísticas de seguridad: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/security/sessions")
async def create_session(
    user_id: str,
    ip_address: str,
    user_agent: str,
    permissions: List[str] = None
):
    """Crear sesión de usuario."""
    try:
        session_id = await security_manager.create_session(
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            permissions=permissions or []
        )
        
        return {
            "session_id": session_id,
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al crear sesión: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/security/sessions/{session_id}")
async def validate_session(session_id: str):
    """Validar sesión de usuario."""
    try:
        session = await security_manager.validate_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Sesión no encontrada o expirada")
        
        return {
            "session": {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "ip_address": session.ip_address,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "permissions": session.permissions
            },
            "valid": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al validar sesión: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Automatización
@router.post("/automation/workflows")
async def create_workflow(request: WorkflowCreateRequest):
    """Crear flujo de trabajo."""
    try:
        workflow_id = await automation_manager.create_workflow(
            name=request.name,
            description=request.description,
            triggers=request.triggers,
            actions=request.actions,
            created_by=request.created_by,
            tags=request.tags
        )
        
        return {
            "workflow_id": workflow_id,
            "success": True,
            "message": "Flujo de trabajo creado exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al crear flujo de trabajo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/automation/workflows/{workflow_id}/activate")
async def activate_workflow(workflow_id: str):
    """Activar flujo de trabajo."""
    try:
        success = await automation_manager.activate_workflow(workflow_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Flujo de trabajo no encontrado")
        
        return {
            "success": True,
            "message": "Flujo de trabajo activado",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al activar flujo de trabajo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/automation/workflows/{workflow_id}/pause")
async def pause_workflow(workflow_id: str):
    """Pausar flujo de trabajo."""
    try:
        success = await automation_manager.pause_workflow(workflow_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Flujo de trabajo no encontrado o no activo")
        
        return {
            "success": True,
            "message": "Flujo de trabajo pausado",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al pausar flujo de trabajo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/automation/workflows/execute")
async def execute_workflow(request: WorkflowExecuteRequest):
    """Ejecutar flujo de trabajo."""
    try:
        execution_id = await automation_manager.execute_workflow(
            workflow_id=request.workflow_id,
            triggered_by=request.triggered_by,
            input_data=request.input_data
        )
        
        return {
            "execution_id": execution_id,
            "success": True,
            "message": "Ejecución de flujo de trabajo iniciada",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al ejecutar flujo de trabajo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/automation/workflows/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Obtener flujo de trabajo."""
    try:
        workflow = await automation_manager.get_workflow(workflow_id)
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Flujo de trabajo no encontrado")
        
        return {
            "workflow": workflow,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener flujo de trabajo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/automation/executions/{execution_id}")
async def get_execution(execution_id: str):
    """Obtener ejecución de flujo de trabajo."""
    try:
        execution = await automation_manager.get_execution(execution_id)
        
        if not execution:
            raise HTTPException(status_code=404, detail="Ejecución no encontrada")
        
        return {
            "execution": execution,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener ejecución: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/automation/stats")
async def get_automation_stats():
    """Obtener estadísticas de automatización."""
    try:
        stats = await automation_manager.get_automation_stats()
        
        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener estadísticas de automatización: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Gestión de Datos
@router.post("/data/store")
async def store_data(request: DataStoreRequest):
    """Almacenar datos."""
    try:
        expires_in = timedelta(hours=request.expires_in_hours) if request.expires_in_hours else None
        
        record_id = await data_manager.store(
            key=request.key,
            value=request.value,
            data_type=DataType(request.data_type),
            storage_type=StorageType(request.storage_type),
            metadata=request.metadata,
            expires_in=expires_in
        )
        
        return {
            "record_id": record_id,
            "success": True,
            "message": "Datos almacenados exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al almacenar datos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/retrieve/{key}")
async def retrieve_data(key: str, storage_type: Optional[str] = None):
    """Recuperar datos."""
    try:
        value = await data_manager.retrieve(
            key=key,
            storage_type=StorageType(storage_type) if storage_type else None
        )
        
        if value is None:
            raise HTTPException(status_code=404, detail="Datos no encontrados")
        
        return {
            "key": key,
            "value": value,
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al recuperar datos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/data/delete/{key}")
async def delete_data(key: str, storage_type: Optional[str] = None):
    """Eliminar datos."""
    try:
        success = await data_manager.delete(
            key=key,
            storage_type=StorageType(storage_type) if storage_type else None
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Datos no encontrados")
        
        return {
            "success": True,
            "message": "Datos eliminados exitosamente",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al eliminar datos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/exists/{key}")
async def check_data_exists(key: str, storage_type: Optional[str] = None):
    """Verificar si existen datos."""
    try:
        exists = await data_manager.exists(
            key=key,
            storage_type=StorageType(storage_type) if storage_type else None
        )
        
        return {
            "key": key,
            "exists": exists,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al verificar existencia de datos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/list")
async def list_data_keys(
    pattern: Optional[str] = None,
    storage_type: Optional[str] = None,
    limit: int = 100
):
    """Listar claves de datos."""
    try:
        keys = await data_manager.list_keys(
            pattern=pattern,
            storage_type=StorageType(storage_type) if storage_type else None,
            limit=limit
        )
        
        return {
            "keys": keys,
            "count": len(keys),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al listar claves: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data/search")
async def search_data(request: DataSearchRequest):
    """Buscar datos."""
    try:
        results = await data_manager.search(
            query=request.query,
            data_type=DataType(request.data_type) if request.data_type else None,
            storage_type=StorageType(request.storage_type) if request.storage_type else None,
            limit=request.limit
        )
        
        return {
            "results": results,
            "count": len(results),
            "query": request.query,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al buscar datos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/stats")
async def get_data_stats():
    """Obtener estadísticas de datos."""
    try:
        stats = await data_manager.get_data_stats()
        
        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error al obtener estadísticas de datos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rutas de Health Check
@router.get("/health/security")
async def security_health_check():
    """Verificar salud del sistema de seguridad."""
    try:
        health = await security_manager.health_check()
        
        return {
            "health": health,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en health check de seguridad: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/automation")
async def automation_health_check():
    """Verificar salud del sistema de automatización."""
    try:
        health = await automation_manager.health_check()
        
        return {
            "health": health,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en health check de automatización: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/data")
async def data_health_check():
    """Verificar salud del gestor de datos."""
    try:
        health = await data_manager.health_check()
        
        return {
            "health": health,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en health check de datos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/all")
async def all_health_checks():
    """Verificar salud de todos los sistemas avanzados."""
    try:
        security_health = await security_manager.health_check()
        automation_health = await automation_manager.health_check()
        data_health = await data_manager.health_check()
        
        overall_status = "healthy"
        if (security_health.get("status") != "healthy" or 
            automation_health.get("status") != "healthy" or 
            data_health.get("status") != "healthy"):
            overall_status = "degraded"
        
        return {
            "overall_status": overall_status,
            "systems": {
                "security": security_health,
                "automation": automation_health,
                "data": data_health
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en health check general: {e}")
        raise HTTPException(status_code=500, detail=str(e))




