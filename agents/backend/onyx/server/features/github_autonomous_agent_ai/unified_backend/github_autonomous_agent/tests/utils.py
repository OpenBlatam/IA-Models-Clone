"""
Utilidades para tests.
"""

import json
from typing import Dict, Any, Optional
from datetime import datetime


def create_task_dict(
    task_id: str = "test-task",
    owner: str = "test",
    repo: str = "test-repo",
    instruction: str = "create file: test.py",
    status: str = "pending",
    **kwargs
) -> Dict[str, Any]:
    """
    Crear diccionario de tarea para tests.
    
    Args:
        task_id: ID de la tarea
        owner: Propietario del repositorio
        repo: Nombre del repositorio
        instruction: Instrucción
        status: Estado de la tarea
        **kwargs: Campos adicionales
        
    Returns:
        Diccionario de tarea
    """
    task = {
        "id": task_id,
        "repository_owner": owner,
        "repository_name": repo,
        "instruction": instruction,
        "status": status,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "metadata": {}
    }
    task.update(kwargs)
    return task


def create_repository_info(
    owner: str = "test",
    name: str = "test-repo",
    **kwargs
) -> Dict[str, Any]:
    """
    Crear información de repositorio para tests.
    
    Args:
        owner: Propietario
        name: Nombre del repositorio
        **kwargs: Campos adicionales
        
    Returns:
        Diccionario con información del repositorio
    """
    repo_info = {
        "name": name,
        "full_name": f"{owner}/{name}",
        "description": "Test repository",
        "url": f"https://github.com/{owner}/{name}",
        "default_branch": "main",
        "language": "Python",
        "stars": 0,
        "forks": 0,
        "is_private": False,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    repo_info.update(kwargs)
    return repo_info


def assert_task_structure(task: Dict[str, Any]) -> None:
    """
    Verificar que una tarea tenga la estructura correcta.
    
    Args:
        task: Diccionario de tarea
        
    Raises:
        AssertionError: Si la estructura es incorrecta
    """
    required_fields = [
        "id", "repository_owner", "repository_name",
        "instruction", "status", "created_at", "updated_at"
    ]
    
    for field in required_fields:
        assert field in task, f"Campo requerido '{field}' no encontrado en tarea"
    
    assert task["status"] in ["pending", "running", "completed", "failed"], \
        f"Estado inválido: {task['status']}"


def assert_repository_info_structure(repo_info: Dict[str, Any]) -> None:
    """
    Verificar que la información del repositorio tenga la estructura correcta.
    
    Args:
        repo_info: Diccionario con información del repositorio
        
    Raises:
        AssertionError: Si la estructura es incorrecta
    """
    required_fields = ["name", "full_name", "url", "default_branch"]
    
    for field in required_fields:
        assert field in repo_info, f"Campo requerido '{field}' no encontrado"


def assert_api_response(response, status_code: int = 200) -> Dict[str, Any]:
    """
    Verificar respuesta de API y retornar JSON.
    
    Args:
        response: Respuesta de TestClient
        status_code: Código de estado esperado
        
    Returns:
        JSON de la respuesta
        
    Raises:
        AssertionError: Si el código de estado no coincide
    """
    assert response.status_code == status_code, \
        f"Expected status {status_code}, got {response.status_code}: {response.text}"
    return response.json()


def create_llm_response(
    content: str = "Test response",
    model: str = "openai/gpt-4o-mini",
    error: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Crear respuesta LLM para tests.
    
    Args:
        content: Contenido de la respuesta
        model: Modelo usado
        error: Error si hay
        **kwargs: Campos adicionales
        
    Returns:
        Diccionario de respuesta LLM
    """
    response = {
        "model": model,
        "content": content,
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        },
        "finish_reason": "stop",
        "latency_ms": 100.0
    }
    if error:
        response["error"] = error
    response.update(kwargs)
    return response



