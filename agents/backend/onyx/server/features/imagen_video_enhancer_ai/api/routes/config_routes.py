"""
Configuration Routes
===================

API routes for configuration management.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ...utils.config_validator import ConfigValidator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["configuration"])


@router.post("/validate")
async def validate_configuration(config: Dict[str, Any]):
    """Validate configuration."""
    try:
        errors = ConfigValidator.validate_config(config)
        file_errors = ConfigValidator.validate_file_paths(config)
        env_errors = ConfigValidator.validate_environment()
        recommendations = ConfigValidator.get_config_recommendations(config)
        
        all_errors = errors + file_errors + env_errors
        
        return JSONResponse({
            "valid": len(all_errors) == 0,
            "errors": all_errors,
            "recommendations": recommendations
        })
    except Exception as e:
        logger.error(f"Error validating configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




