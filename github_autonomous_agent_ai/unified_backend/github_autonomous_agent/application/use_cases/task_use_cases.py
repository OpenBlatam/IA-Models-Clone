"""
Task Use Cases

Use cases for task management with improved validation and error handling.
"""

from typing import Dict, Any, Optional, List

from config.logging_config import get_logger
from core.storage import TaskStorage
from core.task_processor import TaskProcessor
from core.exceptions import TaskProcessingError, InstructionParseError
from core.constants import TaskStatus

logger = get_logger(__name__)


class CreateTaskUseCase:
    """Use case for creating a new task."""
    
    def __init__(self, task_processor: TaskProcessor):
        """
        Initialize use case.
        
        Args:
            task_processor: Task processor instance
        """
        self.task_processor = task_processor
    
    async def execute(
        self,
        repository_owner: str,
        repository_name: str,
        instruction: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new task with improved validation.
        
        Args:
            repository_owner: Repository owner
            repository_name: Repository name
            instruction: Instruction to execute
            metadata: Optional metadata
        
        Returns:
            Task dictionary
        
        Raises:
            TaskProcessingError: If task creation fails
            InstructionParseError: If instruction is invalid
            ValueError: If repository information is invalid
        """
        # Validar repositorio
        if not repository_owner or not repository_name:
            raise ValueError("repository_owner y repository_name son requeridos")
            
        validated_owner = repository_owner.strip()
        validated_repo = repository_name.strip()
        
        # Validar instrucción
        if not instruction or not instruction.strip():
            raise InstructionParseError("La instrucción no puede estar vacía")
            
        validated_instruction = instruction.strip()
        
        logger.info(
            f"Creando tarea para repositorio {validated_owner}/{validated_repo} "
            f"(instrucción: {len(validated_instruction)} caracteres)"
        )
        
        try:
            task = await self.task_processor.process_instruction(
                repository_owner=validated_owner,
                repository_name=validated_repo,
                instruction=validated_instruction,
                metadata=metadata
            )
            logger.info(f"✅ Tarea creada exitosamente: {task['id']}")
            return task
        except (TaskProcessingError, InstructionParseError):
            # Re-raise task processing errors as-is
            raise
        except Exception as e:
            logger.error(
                f"Error inesperado al crear tarea para {validated_owner}/{validated_repo}: {e}",
                exc_info=True
            )
            raise TaskProcessingError(
                message="Failed to create task",
                details={
                    "repository_owner": validated_owner,
                    "repository_name": validated_repo,
                    "instruction_length": len(validated_instruction)
                },
                original_error=e
            ) from e


class GetTaskUseCase:
    """Use case for getting a task by ID."""
    
    def __init__(self, storage: TaskStorage):
        """
        Initialize use case.
        
        Args:
            storage: Task storage instance
        """
        self.storage = storage
    
    async def execute(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a task by ID with improved validation.
        
        Args:
            task_id: Task ID
        
        Returns:
            Task dictionary or None if not found
        
        Raises:
            TaskProcessingError: If there's an error retrieving the task
            ValueError: If task_id is invalid
        """
        # Validar task_id
        if not task_id or not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("Task ID no puede estar vacío")
        
        task_id = task_id.strip()
        
        # Validar formato UUID básico
        if len(task_id) != 36:  # UUID v4 length
            logger.warning(f"Task ID con formato inválido: {task_id}")
            raise ValueError(f"Task ID tiene formato inválido: {task_id}")
        
        logger.debug(f"Obteniendo tarea: {task_id}")
        
        try:
            task = await self.storage.get_task(task_id)
            if task:
                logger.debug(f"✅ Tarea obtenida exitosamente: {task_id} (status: {task.get('status')})")
            else:
                logger.warning(f"⚠️  Tarea no encontrada: {task_id}")
            return task
        except TaskProcessingError:
            # Re-raise task processing errors as-is
            raise
        except Exception as e:
            logger.error(f"Error inesperado al obtener tarea {task_id}: {e}", exc_info=True)
            raise TaskProcessingError(
                message="Failed to get task",
                task_id=task_id,
                original_error=e
            ) from e


class ListTasksUseCase:
    """Use case for listing tasks."""
    
    def __init__(self, storage: TaskStorage):
        """
        Initialize use case.
        
        Args:
            storage: Task storage instance
        """
        self.storage = storage
    
    async def execute(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List tasks with improved validation.
        
        Args:
            status: Optional status filter (must be a valid TaskStatus)
            limit: Maximum number of tasks to return (1-1000)
        
        Returns:
            List of task dictionaries
        
        Raises:
            TaskProcessingError: If there's an error listing tasks
            ValueError: If status or limit is invalid
        """
        # Validar status si se proporciona
        if status:
            if not TaskStatus.is_valid(status):
                logger.warning(f"Estado inválido proporcionado: {status}")
                raise ValueError(
                    f"Estado inválido: {status}. "
                    f"Estados válidos: {', '.join(TaskStatus.ALL_STATES)}"
                )
        
        # Validar y limitar limit
        if not isinstance(limit, int) or limit < 1:
            limit = 1
        elif limit > 1000:
            logger.warning(f"Límite muy alto proporcionado: {limit}, limitando a 1000")
            limit = 1000
        
        logger.debug(f"Listando tareas: status={status}, limit={limit}")
        
        try:
            tasks = await self.storage.get_tasks(status=status, limit=limit)
            logger.info(f"✅ {len(tasks)} tareas obtenidas (status={status}, limit={limit})")
            return tasks
        except TaskProcessingError:
            # Re-raise task processing errors as-is
            raise
        except Exception as e:
            logger.error(
                f"Error inesperado al listar tareas (status={status}, limit={limit}): {e}",
                exc_info=True
            )
            raise TaskProcessingError(
                message="Failed to list tasks",
                details={
                    "status": status,
                    "limit": limit
                },
                original_error=e
            ) from e

