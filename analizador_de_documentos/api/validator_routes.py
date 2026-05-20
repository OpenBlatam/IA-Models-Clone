"""
Rutas para Validación de Documentos
====================================

Endpoints para validar documentos según reglas personalizadas.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Form
from pydantic import BaseModel, Field

from ..core.document_validator import DocumentValidator, ValidationSeverity
from .routes import get_analyzer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/validation",
    tags=["Document Validation"]
)


class ValidateRequest(BaseModel):
    """Request para validación"""
    content: str = Field(..., description="Contenido del documento")
    rules: Optional[List[str]] = Field(None, description="Reglas específicas a aplicar")
    custom_params: Optional[Dict[str, Any]] = Field(None, description="Parámetros personalizados")


class AddRuleRequest(BaseModel):
    """Request para agregar regla"""
    name: str
    description: str
    severity: str
    error_message: str
    enabled: bool = True


# Instancia global del validador
_validator: Optional[DocumentValidator] = None


def get_validator() -> DocumentValidator:
    """Dependency para obtener validador"""
    global _validator
    if _validator is None:
        _validator = DocumentValidator()
    return _validator


@router.post("/validate")
async def validate_document(
    request: ValidateRequest,
    validator: DocumentValidator = Depends(get_validator)
):
    """Validar documento"""
    try:
        result = await validator.validate(
            request.content,
            request.rules,
            request.custom_params or {}
        )
        
        return {
            "is_valid": result.is_valid,
            "score": result.score,
            "errors": result.errors,
            "warnings": result.warnings,
            "info": result.info,
            "timestamp": result.timestamp
        }
    except Exception as e:
        logger.error(f"Error validando documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rules")
async def get_rules(
    validator: DocumentValidator = Depends(get_validator)
):
    """Obtener lista de reglas de validación"""
    return {"rules": validator.get_rules()}


@router.post("/rules")
async def add_rule(
    request: AddRuleRequest,
    validator: DocumentValidator = Depends(get_validator)
):
    """Agregar regla de validación personalizada"""
    try:
        severity = ValidationSeverity(request.severity)
        
        # Crear función validador simple
        def simple_validator(content: str) -> bool:
            # Esta es una función placeholder - en producción debería ser más compleja
            return True
        
        validator.add_rule(
            name=request.name,
            description=request.description,
            severity=severity,
            validator=simple_validator,
            error_message=request.error_message,
            enabled=request.enabled
        )
        
        return {"status": "rule_added", "name": request.name}
    except Exception as e:
        logger.error(f"Error agregando regla: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rules/{rule_name}/enable")
async def enable_rule(
    rule_name: str,
    validator: DocumentValidator = Depends(get_validator)
):
    """Habilitar regla"""
    validator.enable_rule(rule_name)
    return {"status": "enabled", "rule": rule_name}


@router.post("/rules/{rule_name}/disable")
async def disable_rule(
    rule_name: str,
    validator: DocumentValidator = Depends(get_validator)
):
    """Deshabilitar regla"""
    validator.disable_rule(rule_name)
    return {"status": "disabled", "rule": rule_name}
















