"""
Dependencies
============

FastAPI dependencies for PDF variantes feature.
"""

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any, Generator
import logging
from pathlib import Path
import os
from functools import lru_cache

from .config import ConfigManager, PDFVariantesConfig
from .services import PDFVariantesService
from .models import validate_file_size

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)


@lru_cache()
def get_config_manager() -> ConfigManager:
    """Get configuration manager singleton."""
    return ConfigManager()


@lru_cache()
def get_config() -> PDFVariantesConfig:
    """Get application configuration."""
    config_manager = get_config_manager()
    return config_manager.get_config()


def get_pdf_service(
    config: PDFVariantesConfig = Depends(get_config)
) -> PDFVariantesService:
    """Get PDF variantes service instance."""
    upload_dir = Path(config.storage.upload_dir)
    return PDFVariantesService(upload_dir)


def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    config: PDFVariantesConfig = Depends(get_config)
) -> Optional[Dict[str, Any]]:
    """
    Get current user from request.
    
    Returns user info if authentication is enabled, otherwise returns None.
    """
    if not config.api.enable_authentication:
        return {"user_id": "anonymous", "role": "user"}
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # TODO: Implement proper JWT token validation
    # For now, return a mock user
    return {
        "user_id": "user_123",
        "role": "user",
        "permissions": ["read", "write"]
    }


def get_admin_user(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get admin user - requires admin role."""
    if not current_user or current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def validate_file_upload(
    file_size: int,
    config: PDFVariantesConfig = Depends(get_config)
) -> bool:
    """Validate file upload constraints."""
    max_size_bytes = config.limits.max_file_size_mb * 1024 * 1024
    
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum limit of {config.limits.max_file_size_mb}MB"
        )
    
    if file_size <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File cannot be empty"
        )
    
    return True


def check_feature_enabled(
    feature_name: str,
    config: PDFVariantesConfig = Depends(get_config)
) -> bool:
    """Check if a feature is enabled."""
    for feature in config.features:
        if feature.name == feature_name:
            if not feature.enabled:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Feature '{feature_name}' is currently disabled"
                )
            return True
    
    # Feature not found in config, assume it's disabled
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"Feature '{feature_name}' is not available"
    )


def get_rate_limit_info(
    request: Request,
    config: PDFVariantesConfig = Depends(get_config)
) -> Dict[str, Any]:
    """Get rate limiting information for the request."""
    client_ip = request.client.host if request.client else "unknown"
    
    return {
        "client_ip": client_ip,
        "rate_limit_per_minute": config.api.rate_limit_per_minute,
        "is_whitelisted": client_ip in config.api.cors_origins,
        "is_blacklisted": False  # TODO: Implement blacklist
    }


def validate_pdf_file_id(
    file_id: str,
    pdf_service: PDFVariantesService = Depends(get_pdf_service)
) -> str:
    """Validate PDF file ID and check if file exists."""
    if not file_id or len(file_id) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file ID format"
        )
    
    file_path = pdf_service.upload_handler.get_file_path(file_id)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PDF file not found"
        )
    
    return file_id


def get_request_context(
    request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get comprehensive request context."""
    return {
        "user": current_user,
        "client_ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", ""),
        "request_id": request.headers.get("x-request-id", ""),
        "timestamp": request.headers.get("x-timestamp", ""),
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers)
    }


def check_collaboration_permissions(
    file_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
    pdf_service: PDFVariantesService = Depends(get_pdf_service)
) -> Dict[str, Any]:
    """Check if user has collaboration permissions for a file."""
    # TODO: Implement proper permission checking
    # For now, return basic permissions
    return {
        "can_view": True,
        "can_edit": current_user and current_user.get("role") in ["user", "admin"],
        "can_delete": current_user and current_user.get("role") == "admin",
        "can_share": current_user and current_user.get("role") in ["user", "admin"]
    }


def get_processing_limits(
    config: PDFVariantesConfig = Depends(get_config)
) -> Dict[str, int]:
    """Get current processing limits."""
    return {
        "max_file_size_mb": config.limits.max_file_size_mb,
        "max_pages": config.limits.max_pages,
        "max_concurrent_processes": config.limits.max_concurrent_processes,
        "max_variants_per_document": config.limits.max_variants_per_document
    }


def validate_processing_request(
    file_id: str,
    processing_type: str,
    limits: Dict[str, int] = Depends(get_processing_limits),
    pdf_service: PDFVariantesService = Depends(get_pdf_service)
) -> Dict[str, Any]:
    """Validate processing request against limits."""
    # Check if file exists
    file_path = pdf_service.upload_handler.get_file_path(file_id)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PDF file not found"
        )
    
    # TODO: Implement more sophisticated limit checking
    # For now, return basic validation
    return {
        "file_id": file_id,
        "processing_type": processing_type,
        "limits": limits,
        "can_process": True
    }


def get_ai_config(
    config: PDFVariantesConfig = Depends(get_config)
) -> Dict[str, Any]:
    """Get AI processing configuration."""
    if not config.ai.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI processing is currently disabled"
        )
    
    return {
        "enabled": config.ai.enabled,
        "provider": config.ai.provider,
        "model": config.ai.model,
        "max_tokens": config.ai.max_tokens,
        "temperature": config.ai.temperature,
        "has_api_key": bool(config.ai.api_key)
    }


def get_storage_info(
    config: PDFVariantesConfig = Depends(get_config)
) -> Dict[str, Any]:
    """Get storage configuration and status."""
    upload_dir = Path(config.storage.upload_dir)
    
    # Calculate storage usage
    total_size = 0
    file_count = 0
    
    if upload_dir.exists():
        for file_path in upload_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
    
    return {
        "upload_dir": str(upload_dir),
        "max_storage_gb": config.storage.max_storage_gb,
        "current_usage_gb": round(total_size / (1024**3), 2),
        "file_count": file_count,
        "retention_days": config.storage.retention_days,
        "compression_enabled": config.storage.enable_compression,
        "backup_enabled": config.storage.backup_enabled
    }
