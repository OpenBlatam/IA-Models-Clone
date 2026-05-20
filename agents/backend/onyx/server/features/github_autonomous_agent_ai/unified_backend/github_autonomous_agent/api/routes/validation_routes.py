"""
Validation Routes - Rutas para validación.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from api.utils import handle_api_errors
from core.services.validation_service import ValidationService, ValidationError
from config.logging_config import get_logger
from config.di_setup import get_service

router = APIRouter()
logger = get_logger(__name__)


class ValidateRequest(BaseModel):
    """Request para validar datos."""
    data: Dict[str, Any] = Field(..., description="Datos a validar")
    fields: Optional[List[str]] = Field(None, description="Campos específicos a validar")


class AddRuleRequest(BaseModel):
    """Request para agregar regla de validación."""
    field: str = Field(..., min_length=1)
    validator_type: str = Field(..., description="Tipo de validador: required, min_length, max_length, email, url, github_repo")
    message: str = Field(..., min_length=1)
    code: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


def get_validation_service() -> ValidationService:
    """Obtener servicio de validación."""
    try:
        return get_service("validation_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Validation service no disponible")


@router.post("/validate")
@handle_api_errors
async def validate_data(
    request: ValidateRequest,
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Validar datos.
    
    Args:
        request: Datos a validar
        
    Returns:
        Datos validados
    """
    try:
        validated = validation_service.validate(request.data, fields=request.fields)
        return {
            "valid": True,
            "data": validated
        }
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "valid": False,
                "error": e.message,
                "field": e.field,
                "code": e.code
            }
        )


@router.post("/validate/email")
@handle_api_errors
async def validate_email(
    email: str,
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Validar email.
    
    Args:
        email: Email a validar
        
    Returns:
        Resultado de validación
    """
    is_valid = validation_service.validate_email(email)
    return {
        "valid": is_valid,
        "email": email
    }


@router.post("/validate/url")
@handle_api_errors
async def validate_url(
    url: str,
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Validar URL.
    
    Args:
        url: URL a validar
        
    Returns:
        Resultado de validación
    """
    is_valid = validation_service.validate_url(url)
    return {
        "valid": is_valid,
        "url": url
    }


@router.post("/validate/github-repo")
@handle_api_errors
async def validate_github_repo(
    repo: str,
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Validar repositorio de GitHub.
    
    Args:
        repo: Repositorio a validar (formato: owner/repo)
        
    Returns:
        Resultado de validación
    """
    is_valid = validation_service.validate_github_repo(repo)
    return {
        "valid": is_valid,
        "repo": repo
    }


@router.get("/stats")
@handle_api_errors
async def get_validation_stats(
    validation_service: ValidationService = Depends(get_validation_service)
):
    """
    Obtener estadísticas de validación.
    
    Returns:
        Estadísticas
    """
    return validation_service.get_stats()



