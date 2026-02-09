"""
Service Helpers - Helpers para usar servicios fácilmente
========================================================

Funciones helper que simplifican el uso de los servicios.
"""

from typing import Optional, Dict, Any
from ..services.project_service import ProjectService
from ..services.generation_service import GenerationService
from ..infrastructure.dependencies import (
    get_project_service,
    get_generation_service
)


def get_project_service_simple() -> ProjectService:
    """
    Obtiene project service de forma simple.
    
    Returns:
        ProjectService configurado
    
    Example:
        ```python
        from helpers import get_project_service_simple
        
        service = get_project_service_simple()
        project = await service.get_project("project-id")
        ```
    """
    return get_project_service()


def get_generation_service_simple() -> GenerationService:
    """
    Obtiene generation service de forma simple.
    
    Returns:
        GenerationService configurado
    
    Example:
        ```python
        from helpers import get_generation_service_simple
        
        service = get_generation_service_simple()
        result = await service.generate_project("description")
        ```
    """
    return get_generation_service()


async def create_project_simple(
    description: str,
    project_name: Optional[str] = None,
    author: str = "Blatam Academy"
) -> Dict[str, Any]:
    """
    Crea un proyecto de forma simple.
    
    Args:
        description: Descripción del proyecto
        project_name: Nombre del proyecto (opcional)
        author: Autor del proyecto
    
    Returns:
        Información del proyecto creado
    
    Example:
        ```python
        from helpers import create_project_simple
        
        result = await create_project_simple(
            description="Un chat con IA",
            project_name="chat_ai"
        )
        print(result["project_id"])
        ```
    """
    service = get_project_service_simple()
    return await service.create_project(
        description=description,
        project_name=project_name,
        author=author
    )


async def generate_project_simple(
    description: str,
    project_name: Optional[str] = None,
    author: str = "Blatam Academy",
    async_generation: bool = True
) -> Dict[str, Any]:
    """
    Genera un proyecto de forma simple.
    
    Args:
        description: Descripción del proyecto
        project_name: Nombre del proyecto (opcional)
        author: Autor del proyecto
        async_generation: Si usar generación asíncrona
    
    Returns:
        Información de la generación
    
    Example:
        ```python
        from helpers import generate_project_simple
        
        result = await generate_project_simple(
            description="Un analizador de imágenes",
            project_name="image_analyzer"
        )
        print(result["project_id"] or result["task_id"])
        ```
    """
    service = get_generation_service_simple()
    return await service.generate_project(
        description=description,
        project_name=project_name,
        author=author,
        async_generation=async_generation
    )















