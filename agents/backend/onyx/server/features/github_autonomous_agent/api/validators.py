"""
Validadores para endpoints de la API.
"""

from fastapi import HTTPException
from typing import Optional
from core.validators import RepositoryValidator, InstructionValidator
from core.exceptions import InstructionParseError


def validate_repository(owner: str, repo: str) -> tuple[str, str]:
    """
    Validar información de repositorio.
    
    Args:
        owner: Propietario del repositorio
        repo: Nombre del repositorio
        
    Returns:
        Tupla con (owner, repo) validados
        
    Raises:
        HTTPException: Si la validación falla
    """
    try:
        validator = RepositoryValidator(owner=owner, repo=repo)
        return validator.owner, validator.repo
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def validate_instruction(instruction: str) -> str:
    """
    Validar instrucción.
    
    Args:
        instruction: Instrucción a validar
        
    Returns:
        Instrucción validada
        
    Raises:
        HTTPException: Si la validación falla
    """
    try:
        return InstructionValidator.validate_instruction(instruction)
    except InstructionParseError as e:
        raise HTTPException(status_code=400, detail=str(e))


def validate_task_id(task_id: str) -> str:
    """
    Validar ID de tarea.
    
    Args:
        task_id: ID de la tarea
        
    Returns:
        ID validado
        
    Raises:
        HTTPException: Si el ID es inválido
    """
    if not task_id:
        raise HTTPException(status_code=400, detail="El ID de la tarea no puede estar vacío")
    
    if len(task_id) != 36:  # UUID v4 length
        raise HTTPException(status_code=400, detail="El ID de la tarea tiene formato inválido")
    
    return task_id




