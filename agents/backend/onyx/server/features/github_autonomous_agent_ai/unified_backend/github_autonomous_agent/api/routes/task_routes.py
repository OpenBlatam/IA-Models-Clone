"""
Task Routes - Rutas para gestión de tareas.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, List

from api.utils import handle_api_errors, validate_github_token
from api.validators import validate_task_id, validate_instruction, validate_repository
from api.dependencies import (
    get_create_task_use_case,
    get_get_task_use_case,
    get_list_tasks_use_case,
    get_storage
)
from api.schemas import CreateTaskRequest, TaskResponse
from application.use_cases.task_use_cases import (
    CreateTaskUseCase,
    GetTaskUseCase,
    ListTasksUseCase
)
from core.storage import TaskStorage
from core.constants import TaskStatus, SuccessMessages
from config.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/", response_model=TaskResponse)
@handle_api_errors
async def create_task(
    request: CreateTaskRequest,
    use_case: CreateTaskUseCase = Depends(get_create_task_use_case)
):
    """
    Crear una nueva tarea.
    
    Args:
        request: Datos de la tarea a crear
        
    Returns:
        TaskResponse: Información de la tarea creada
        
    Raises:
        HTTPException: Si la validación falla o hay un error al crear la tarea
    """
    validate_github_token()
    
    # Validar parámetros
    validated_owner, validated_repo = validate_repository(
        request.repository_owner,
        request.repository_name
    )
    validated_instruction = validate_instruction(request.instruction)
    
    logger.info(f"Creando tarea para repositorio {validated_owner}/{validated_repo}")
    
    task = await use_case.execute(
        repository_owner=validated_owner,
        repository_name=validated_repo,
        instruction=validated_instruction,
        metadata=request.metadata
    )

    logger.info(f"Tarea {task.get('id')} creada exitosamente")
    
    # Publicar evento y broadcast WebSocket
    try:
        from core.events import publish_task_event, EventType
        await publish_task_event(EventType.TASK_CREATED, task, source="task_routes")
    except Exception as e:
        logger.debug(f"Event publishing failed (non-critical): {e}")
    
    return TaskResponse(**task)


@router.get("/{task_id}", response_model=TaskResponse)
@handle_api_errors
async def get_task(
    task_id: str,
    use_case: GetTaskUseCase = Depends(get_get_task_use_case)
):
    """
    Obtener una tarea por ID.
    
    Args:
        task_id: ID de la tarea
        
    Returns:
        TaskResponse: Información de la tarea
        
    Raises:
        HTTPException: Si la tarea no existe o el ID es inválido
    """
    validated_task_id = validate_task_id(task_id)
    
    logger.debug(f"Obteniendo tarea {validated_task_id}")
    task = await use_case.execute(validated_task_id)
    
    if not task:
        logger.warning(f"Tarea {validated_task_id} no encontrada")
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    
    return TaskResponse(**task)


@router.get("/", response_model=List[TaskResponse])
@handle_api_errors
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 100,
    use_case: ListTasksUseCase = Depends(get_list_tasks_use_case)
):
    """
    Listar tareas con filtros opcionales.
    
    Args:
        status: Estado de las tareas a filtrar (opcional)
        limit: Número máximo de tareas a retornar (default: 100, max: 1000)
        
    Returns:
        List[TaskResponse]: Lista de tareas
        
    Raises:
        HTTPException: Si el estado es inválido o el límite está fuera de rango
    """
    # Validar status si se proporciona
    if status and status not in [TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
        raise HTTPException(
            status_code=400,
            detail=f"Estado inválido: {status}. Estados válidos: {', '.join([TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED])}"
        )
    
    # Validar y limitar el límite
    if limit < 1:
        limit = 1
    elif limit > 1000:
        limit = 1000
    
    logger.debug(f"Listando tareas: status={status}, limit={limit}")
    tasks = await use_case.execute(status=status, limit=limit)
    
    logger.debug(f"Se encontraron {len(tasks)} tareas")
    return [TaskResponse(**task) for task in tasks]


@router.delete("/{task_id}")
@handle_api_errors
async def delete_task(
    task_id: str,
    storage: TaskStorage = Depends(get_storage)
):
    """
    Eliminar una tarea.
    
    Args:
        task_id: ID de la tarea a eliminar
        
    Returns:
        dict: Mensaje de confirmación
        
    Raises:
        HTTPException: Si la tarea no existe o el ID es inválido
    """
    validated_task_id = validate_task_id(task_id)
    
    logger.info(f"Eliminando tarea {validated_task_id}")
    
    # Verificar que la tarea existe
    task = await storage.get_task(validated_task_id)
    if not task:
        logger.warning(f"Intento de eliminar tarea inexistente: {validated_task_id}")
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    
    # Eliminar usando el método del storage
    deleted = await storage.delete_task(validated_task_id)
    
    if not deleted:
        logger.error(f"No se pudo eliminar la tarea {validated_task_id}")
        raise HTTPException(status_code=500, detail="Error al eliminar la tarea")
    
    logger.info(f"Tarea {validated_task_id} eliminada exitosamente")
    return {"message": SuccessMessages.TASK_DELETED}

