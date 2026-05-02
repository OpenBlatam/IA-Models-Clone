"""
Helpers avanzados para testing.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import status
from httpx import Response


def assert_response_success(response: Response, expected_status: int = status.HTTP_200_OK) -> Dict[str, Any]:
    """
    Verificar que la respuesta sea exitosa.
    
    Args:
        response: Response de httpx
        expected_status: Status code esperado
        
    Returns:
        JSON de la respuesta
        
    Raises:
        AssertionError: Si la respuesta no es exitosa
    """
    assert response.status_code == expected_status, \
        f"Expected status {expected_status}, got {response.status_code}: {response.text}"
    return response.json()


def assert_response_error(response: Response, expected_status: int = status.HTTP_400_BAD_REQUEST) -> Dict[str, Any]:
    """
    Verificar que la respuesta sea un error.
    
    Args:
        response: Response de httpx
        expected_status: Status code esperado
        
    Returns:
        JSON de la respuesta
        
    Raises:
        AssertionError: Si la respuesta no es un error
    """
    assert response.status_code == expected_status, \
        f"Expected error status {expected_status}, got {response.status_code}: {response.text}"
    data = response.json()
    assert "error" in data or "detail" in data, "Response should contain error or detail"
    return data


def create_task_payload(
    owner: str = "test",
    repo: str = "test-repo",
    instruction: str = "create file: test.py",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Crear payload para crear tarea.
    
    Args:
        owner: Propietario del repositorio
        repo: Nombre del repositorio
        instruction: Instrucción
        metadata: Metadata adicional
        
    Returns:
        Payload de request
    """
    payload = {
        "repository_owner": owner,
        "repository_name": repo,
        "instruction": instruction
    }
    if metadata:
        payload["metadata"] = metadata
    return payload


def create_batch_tasks_payload(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Crear payload para batch create.
    
    Args:
        tasks: Lista de tareas
        
    Returns:
        Payload de request
    """
    return {"tasks": tasks}


def assert_task_structure(task: Dict[str, Any], required_fields: Optional[List[str]] = None) -> None:
    """
    Verificar estructura de tarea.
    
    Args:
        task: Diccionario de tarea
        required_fields: Campos requeridos (opcional)
        
    Raises:
        AssertionError: Si la estructura es incorrecta
    """
    if required_fields is None:
        required_fields = [
            "id", "repository_owner", "repository_name",
            "instruction", "status", "created_at", "updated_at"
        ]
    
    for field in required_fields:
        assert field in task, f"Campo requerido '{field}' no encontrado"
    
    assert task["status"] in ["pending", "running", "completed", "failed", "cancelled"], \
        f"Estado inválido: {task['status']}"


def assert_batch_response(response: Dict[str, Any]) -> None:
    """
    Verificar estructura de respuesta batch.
    
    Args:
        response: Respuesta batch
        
    Raises:
        AssertionError: Si la estructura es incorrecta
    """
    assert "total" in response
    assert "successful" in response
    assert "failed" in response
    assert "tasks" in response
    assert "errors" in response
    assert response["total"] == response["successful"] + response["failed"]


def wait_for_task_status(
    client,
    task_id: str,
    expected_status: str,
    max_wait: int = 10,
    interval: float = 0.5
) -> Dict[str, Any]:
    """
    Esperar hasta que una tarea tenga un estado específico.
    
    Args:
        client: Cliente de test
        task_id: ID de la tarea
        expected_status: Estado esperado
        max_wait: Tiempo máximo de espera (segundos)
        interval: Intervalo entre verificaciones (segundos)
        
    Returns:
        Tarea con el estado esperado
        
    Raises:
        TimeoutError: Si no se alcanza el estado en el tiempo máximo
    """
    import time
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        response = client.get(f"/api/v1/tasks/{task_id}")
        if response.status_code == 200:
            task = response.json()
            if task.get("status") == expected_status:
                return task
        time.sleep(interval)
    
    raise TimeoutError(f"Task {task_id} did not reach status {expected_status} in {max_wait}s")


def create_auth_headers(token: str = "test-token") -> Dict[str, str]:
    """
    Crear headers de autenticación.
    
    Args:
        token: Token de autenticación
        
    Returns:
        Headers
    """
    return {"Authorization": f"Bearer {token}"}


def assert_rate_limit_headers(response: Response) -> None:
    """
    Verificar headers de rate limit.
    
    Args:
        response: Response de httpx
        
    Raises:
        AssertionError: Si los headers no están presentes
    """
    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Remaining" in response.headers


def create_llm_request(
    prompt: str = "Test prompt",
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Crear request para LLM.
    
    Args:
        prompt: Prompt principal
        system_prompt: System prompt (opcional)
        model: Modelo a usar (opcional)
        temperature: Temperature (opcional)
        
    Returns:
        Payload de request
    """
    payload = {"prompt": prompt, "temperature": temperature}
    if system_prompt:
        payload["system_prompt"] = system_prompt
    if model:
        payload["model"] = model
    return payload



