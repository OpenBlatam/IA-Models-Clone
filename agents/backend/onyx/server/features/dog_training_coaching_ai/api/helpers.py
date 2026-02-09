"""
API Helpers
===========
Utilidades para endpoints de API.
"""

from functools import wraps
from typing import Callable, Any, Optional, List
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from ...core.exceptions import ServiceException, ValidationException
from ...utils.validators import (
    validate_dog_breed,
    validate_dog_age,
    validate_dog_size,
    validate_training_goals,
    validate_experience_level
)


def handle_api_errors(func: Callable) -> Callable:
    """
    Decorador para manejar errores de API de forma consistente.
    
    Args:
        func: Función del endpoint
        
    Returns:
        Función decorada
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValidationException as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ServiceException as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
    return wrapper


def validate_request_fields(
    dog_breed: Optional[str] = None,
    dog_age: Optional[str] = None,
    dog_size: Optional[str] = None,
    training_goals: Optional[List[str]] = None,
    experience_level: Optional[str] = None
) -> None:
    """
    Validar campos comunes de request.
    
    Args:
        dog_breed: Raza del perro
        dog_age: Edad del perro
        dog_size: Tamaño del perro
        training_goals: Objetivos de entrenamiento
        experience_level: Nivel de experiencia
    """
    if dog_breed is not None:
        validate_dog_breed(dog_breed)
    if dog_age is not None:
        validate_dog_age(dog_age)
    if dog_size is not None:
        validate_dog_size(dog_size)
    if training_goals is not None:
        validate_training_goals(training_goals)
    if experience_level is not None:
        validate_experience_level(experience_level)


