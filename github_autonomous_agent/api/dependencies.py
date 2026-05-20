"""
Dependencias compartidas para las rutas de la API.

Usa el sistema de DI para obtener servicios de manera consistente.
"""

from typing import Annotated
from fastapi import Depends, HTTPException
from core.github_client import GitHubClient
from core.storage import TaskStorage
from core.task_processor import TaskProcessor
from core.worker import WorkerManager
from application.use_cases.task_use_cases import (
    CreateTaskUseCase,
    GetTaskUseCase,
    ListTasksUseCase
)
from application.use_cases.github_use_cases import (
    GetRepositoryInfoUseCase,
    CloneRepositoryUseCase
)
from config.di_setup import get_service


def get_storage() -> TaskStorage:
    """
    Dependency para obtener instancia de TaskStorage desde DI container.
    
    Returns:
        Instancia de TaskStorage
    """
    return get_service("storage")


def get_github_client() -> GitHubClient:
    """
    Dependency para obtener instancia de GitHubClient desde DI container.
    
    Returns:
        Instancia de GitHubClient
        
    Raises:
        HTTPException: Si el token de GitHub no está configurado
    """
    try:
        return get_service("github_client")
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"GitHub token no configurado: {str(e)}"
        )


def get_task_processor() -> TaskProcessor:
    """
    Dependency para obtener instancia de TaskProcessor desde DI container.
    
    Returns:
        Instancia de TaskProcessor
    """
    return get_service("task_processor")


def get_worker_manager() -> WorkerManager:
    """
    Dependency para obtener instancia de WorkerManager desde DI container.
    
    Returns:
        Instancia de WorkerManager
    """
    return get_service("worker_manager")


# Use Cases Dependencies
def get_create_task_use_case() -> CreateTaskUseCase:
    """Get CreateTaskUseCase from DI container."""
    return get_service("create_task_use_case")


def get_get_task_use_case() -> GetTaskUseCase:
    """Get GetTaskUseCase from DI container."""
    return get_service("get_task_use_case")


def get_list_tasks_use_case() -> ListTasksUseCase:
    """Get ListTasksUseCase from DI container."""
    return get_service("list_tasks_use_case")


def get_get_repository_info_use_case() -> GetRepositoryInfoUseCase:
    """Get GetRepositoryInfoUseCase from DI container."""
    return get_service("get_repository_info_use_case")


def get_clone_repository_use_case() -> CloneRepositoryUseCase:
    """Get CloneRepositoryUseCase from DI container."""
    return get_service("clone_repository_use_case")

