"""
Utilidades de Testing y Helpers.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import uuid
import random
import string

from config.logging_config import get_logger

logger = get_logger(__name__)


def generate_task_id() -> str:
    """Generar ID de tarea único."""
    return f"task-{uuid.uuid4().hex[:12]}"


def generate_repository() -> str:
    """Generar nombre de repositorio aleatorio."""
    owner = ''.join(random.choices(string.ascii_lowercase, k=random.randint(5, 10)))
    repo = ''.join(random.choices(string.ascii_lowercase, k=random.randint(5, 15)))
    return f"{owner}/{repo}"


def generate_instruction() -> str:
    """Generar instrucción aleatoria."""
    instructions = [
        "Create a new file with hello world",
        "Update the README with project description",
        "Add error handling to the main function",
        "Refactor the code to use async/await",
        "Add unit tests for the service layer",
        "Fix the bug in the authentication module",
        "Optimize the database queries",
        "Add logging to track errors",
        "Implement caching for API responses",
        "Update dependencies to latest versions"
    ]
    return random.choice(instructions)


def create_mock_task(
    task_id: Optional[str] = None,
    repository: Optional[str] = None,
    instruction: Optional[str] = None,
    status: str = "pending",
    **kwargs
) -> Dict[str, Any]:
    """
    Crear tarea mock.
    
    Args:
        task_id: ID de tarea (opcional)
        repository: Repositorio (opcional)
        instruction: Instrucción (opcional)
        status: Status (opcional)
        **kwargs: Campos adicionales
        
    Returns:
        Diccionario con datos de tarea
    """
    return {
        "id": task_id or generate_task_id(),
        "repository_owner": (repository or generate_repository()).split("/")[0],
        "repository_name": (repository or generate_repository()).split("/")[1],
        "instruction": instruction or generate_instruction(),
        "status": status,
        "created_at": datetime.now().isoformat(),
        "started_at": None if status == "pending" else (datetime.now() - timedelta(minutes=5)).isoformat(),
        "result": None,
        "error": None,
        **kwargs
    }


def create_mock_tasks(count: int, **kwargs) -> List[Dict[str, Any]]:
    """
    Crear múltiples tareas mock.
    
    Args:
        count: Número de tareas
        **kwargs: Parámetros para create_mock_task
        
    Returns:
        Lista de tareas
    """
    return [create_mock_task(**kwargs) for _ in range(count)]


def create_mock_llm_response(
    content: Optional[str] = None,
    model: str = "gpt-4",
    **kwargs
) -> Dict[str, Any]:
    """
    Crear respuesta mock de LLM.
    
    Args:
        content: Contenido de respuesta (opcional)
        model: Modelo usado (opcional)
        **kwargs: Campos adicionales
        
    Returns:
        Diccionario con respuesta de LLM
    """
    default_content = """Here's the plan:
1. Create the file
2. Add the code
3. Test the implementation

Code:
```python
def hello():
    print("Hello, World!")
```"""
    
    return {
        "content": content or default_content,
        "model": model,
        "usage": {
            "prompt_tokens": random.randint(100, 500),
            "completion_tokens": random.randint(50, 300),
            "total_tokens": random.randint(150, 800)
        },
        "created_at": datetime.now().isoformat(),
        **kwargs
    }


def assert_task_structure(task: Dict[str, Any]) -> bool:
    """
    Verificar estructura de tarea.
    
    Args:
        task: Tarea a verificar
        
    Returns:
        True si la estructura es válida
        
    Raises:
        AssertionError: Si la estructura es inválida
    """
    required_fields = ["id", "repository_owner", "repository_name", "instruction", "status"]
    
    for field in required_fields:
        assert field in task, f"Campo requerido '{field}' no encontrado"
    
    assert task["status"] in ["pending", "running", "completed", "failed", "cancelled"], \
        f"Status inválido: {task['status']}"
    
    return True


def assert_llm_response_structure(response: Dict[str, Any]) -> bool:
    """
    Verificar estructura de respuesta de LLM.
    
    Args:
        response: Respuesta a verificar
        
    Returns:
        True si la estructura es válida
        
    Raises:
        AssertionError: Si la estructura es inválida
    """
    required_fields = ["content", "model"]
    
    for field in required_fields:
        assert field in response, f"Campo requerido '{field}' no encontrado"
    
    return True


def wait_for_condition(
    condition: callable,
    timeout: float = 5.0,
    interval: float = 0.1,
    error_message: Optional[str] = None
) -> bool:
    """
    Esperar hasta que una condición se cumpla.
    
    Args:
        condition: Función que retorna True cuando se cumple
        timeout: Tiempo máximo en segundos
        interval: Intervalo entre verificaciones
        error_message: Mensaje de error si falla
        
    Returns:
        True si la condición se cumplió
        
    Raises:
        TimeoutError: Si la condición no se cumple en el tiempo límite
    """
    import time
    start = time.time()
    
    while time.time() - start < timeout:
        if condition():
            return True
        time.sleep(interval)
    
    raise TimeoutError(
        error_message or f"Condición no se cumplió después de {timeout}s"
    )


class MockStorage:
    """Storage mock para testing."""
    
    def __init__(self):
        """Inicializar storage mock."""
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.repositories: Dict[str, Dict[str, Any]] = {}
    
    async def create_task(self, **kwargs) -> Dict[str, Any]:
        """Crear tarea."""
        task = create_mock_task(**kwargs)
        self.tasks[task["id"]] = task
        return task
    
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtener tarea."""
        return self.tasks.get(task_id)
    
    async def list_tasks(self, **kwargs) -> List[Dict[str, Any]]:
        """Listar tareas."""
        return list(self.tasks.values())
    
    async def update_task(self, task_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Actualizar tarea."""
        if task_id in self.tasks:
            self.tasks[task_id].update(kwargs)
            return self.tasks[task_id]
        return None
    
    async def delete_task(self, task_id: str) -> bool:
        """Eliminar tarea."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True
        return False
    
    def clear(self) -> None:
        """Limpiar storage."""
        self.tasks.clear()
        self.repositories.clear()



