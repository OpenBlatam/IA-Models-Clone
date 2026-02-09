"""
Dynamic Configuration API for runtime configuration management.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

from .base_router import BaseRouter
from .exceptions import ValidationError
from ..config.maintenance_config import MaintenanceConfig
from ...utils.file_helpers import get_iso_timestamp

# Create base router instance
base = BaseRouter(
    prefix="/api/config",
    tags=["Configuration"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration."""
    section: str = Field(..., description="Configuration section: openrouter, nlp, ml, cache, rate_limiter")
    key: str = Field(..., description="Configuration key to update")
    value: Any = Field(..., description="New value")
    restart_required: bool = Field(False, description="Whether restart is required")


class ConfigValidationRequest(BaseModel):
    """Request to validate configuration."""
    config: Dict[str, Any] = Field(..., description="Configuration to validate")


@router.get("/current")
@base.timed_endpoint("get_current_config")
async def get_current_config(
    section: Optional[str] = Query(None, description="Configuration section"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get current configuration (sensitive values may be masked).
    """
    base.log_request("get_current_config", section=section)
    
    config = MaintenanceConfig()
    
    if section == "openrouter":
        return base.success({
            "base_url": config.openrouter.base_url,
            "default_model": config.openrouter.default_model,
            "temperature": config.openrouter.temperature,
            "max_tokens": config.openrouter.max_tokens,
            "timeout": config.openrouter.timeout,
            "api_key": "***masked***" if config.openrouter.api_key else None
        })
    elif section == "nlp":
        return base.success({
            "language": config.nlp.language,
            "use_spacy": config.nlp.use_spacy,
            "use_transformers": config.nlp.use_transformers
        })
    elif section == "ml":
        return base.success({
            "enable_predictive_maintenance": config.ml.enable_predictive_maintenance,
            "enable_anomaly_detection": config.ml.enable_anomaly_detection,
            "model_path": config.ml.model_path,
            "retrain_interval_days": config.ml.retrain_interval_days
        })
    else:
        return base.success({
            "openrouter": {
                "base_url": config.openrouter.base_url,
                "default_model": config.openrouter.default_model
            },
            "nlp": {
                "language": config.nlp.language,
                "use_spacy": config.nlp.use_spacy
            },
            "ml": {
                "enable_predictive_maintenance": config.ml.enable_predictive_maintenance,
                "enable_anomaly_detection": config.ml.enable_anomaly_detection
            }
        })


@router.post("/update")
@base.timed_endpoint("update_config")
async def update_config(
    request: ConfigUpdateRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Update configuration at runtime (if supported).
    
    Note: Some configuration changes may require restart.
    """
    base.log_request("update_config", section=request.section, key=request.key)
    
    # Validate section
    valid_sections = ["openrouter", "nlp", "ml", "cache", "rate_limiter"]
    if request.section not in valid_sections:
        raise ValidationError(f"Invalid section. Allowed: {', '.join(valid_sections)}")
    
    # Validate value based on key
    if request.section == "openrouter":
        if request.key == "temperature" and not (0 <= request.value <= 2):
            raise ValidationError("Temperature must be between 0 and 2")
        if request.key == "max_tokens" and request.value < 1:
            raise ValidationError("max_tokens must be at least 1")
    
    # In production, this would actually update the config
    # For now, return success with note about restart
    
    return base.success({
        "section": request.section,
        "key": request.key,
        "value": request.value,
        "updated_at": get_iso_timestamp(),
        "restart_required": request.restart_required,
        "message": "Configuration update accepted. Restart may be required for changes to take effect."
    })


@router.post("/validate")
@base.timed_endpoint("validate_config")
async def validate_config(
    request: ConfigValidationRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Validate a configuration without applying it.
    """
    base.log_request("validate_config")
    
    errors = []
    warnings = []
    
    config = request.config
    
    # Validate OpenRouter config
    if "openrouter" in config:
        or_config = config["openrouter"]
        if "api_key" in or_config and not or_config["api_key"]:
            errors.append("OpenRouter API key is required")
        if "temperature" in or_config:
            temp = or_config["temperature"]
            if not (0 <= temp <= 2):
                errors.append("Temperature must be between 0 and 2")
        if "max_tokens" in or_config:
            tokens = or_config["max_tokens"]
            if tokens < 1:
                errors.append("max_tokens must be at least 1")
    
    # Validate NLP config
    if "nlp" in config:
        nlp_config = config["nlp"]
        if "language" in nlp_config:
            valid_languages = ["es", "en", "fr", "de"]
            if nlp_config["language"] not in valid_languages:
                warnings.append(f"Language {nlp_config['language']} may not be fully supported")
    
    # Validate ML config
    if "ml" in config:
        ml_config = config["ml"]
        if "retrain_interval_days" in ml_config:
            days = ml_config["retrain_interval_days"]
            if days < 1:
                errors.append("retrain_interval_days must be at least 1")
            if days > 365:
                warnings.append("retrain_interval_days is very high (>365 days)")
    
    return base.success({
        "valid": not errors,
        "errors": errors,
        "warnings": warnings
    })


@router.get("/sections")
@base.timed_endpoint("get_config_sections")
async def get_config_sections(
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get available configuration sections and their keys.
    """
    base.log_request("get_config_sections")
    
    sections = {
        "openrouter": {
            "description": "OpenRouter API configuration",
            "keys": ["api_key", "base_url", "default_model", "temperature", "max_tokens", "timeout"],
            "restart_required": True
        },
        "nlp": {
            "description": "NLP processing configuration",
            "keys": ["language", "use_spacy", "use_transformers"],
            "restart_required": True
        },
        "ml": {
            "description": "Machine Learning configuration",
            "keys": ["enable_predictive_maintenance", "enable_anomaly_detection", "model_path", "retrain_interval_days"],
            "restart_required": True
        },
        "cache": {
            "description": "Cache configuration",
            "keys": ["enabled", "default_ttl", "max_size"],
            "restart_required": False
        },
        "rate_limiter": {
            "description": "Rate limiting configuration",
            "keys": ["enabled", "max_requests", "window_seconds"],
            "restart_required": False
        }
    }
    
    return base.success({
        "sections": sections,
        "total": len(sections)
    })




