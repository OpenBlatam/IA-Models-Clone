"""
Advanced Validation endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.advanced_validation import AdvancedValidationService

router = APIRouter()
validation_service = AdvancedValidationService()


@router.post("/validate")
async def validate_data(
    data: Dict[str, Any],
    rules: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Validar datos según reglas"""
    try:
        result = validation_service.validate(data, rules)
        return {
            "valid": result.valid,
            "errors": [
                {
                    "field": e.field,
                    "rule": e.rule,
                    "message": e.message,
                }
                for e in result.errors
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-schema")
async def validate_schema(
    data: Dict[str, Any],
    schema: Dict[str, Any]
) -> Dict[str, Any]:
    """Validar datos contra schema"""
    try:
        result = validation_service.validate_schema(data, schema)
        return {
            "valid": result.valid,
            "errors": [
                {
                    "field": e.field,
                    "rule": e.rule,
                    "message": e.message,
                }
                for e in result.errors
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




