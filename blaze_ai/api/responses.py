"""
Response utilities for the Blaze AI API.

This module provides utility functions for creating standardized API responses
and handling common response patterns.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from datetime import datetime
from fastapi.responses import ORJSONResponse
from .schemas import BaseResponse, ErrorResponse, SuccessResponse


def create_success_response(
    message: str = "Success",
    data: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized success response.
    
    Args:
        message: Success message
        data: Response data
        metadata: Additional metadata
        
    Returns:
        Standardized success response dictionary
    """
    response = {
        "success": True,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data or {},
        "metadata": metadata or {}
    }
    return response


def create_error_response(
    message: str,
    error_code: str = "UNKNOWN_ERROR",
    error_type: str = "GeneralError",
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    status_code: int = 500
) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        message: Error message
        error_code: Error code for categorization
        error_type: Type of error
        details: Additional error details
        request_id: Request ID for tracking
        status_code: HTTP status code
        
    Returns:
        Standardized error response dictionary
    """
    response = {
        "success": False,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
        "error_code": error_code,
        "error_type": error_type,
        "details": details or {},
        "request_id": request_id,
        "status_code": status_code
    }
    return response


def create_orjson_response(
    data: Dict[str, Any],
    status_code: int = 200,
    headers: Optional[Dict[str, str]] = None
) -> ORJSONResponse:
    """
    Create a FastAPI ORJSONResponse with standardized formatting.
    
    Args:
        data: Response data
        status_code: HTTP status code
        headers: Additional headers
        
    Returns:
        ORJSONResponse instance
    """
    return ORJSONResponse(
        content=data,
        status_code=status_code,
        headers=headers or {}
    )


def create_health_response(
    status: str,
    components: Dict[str, Dict[str, Any]],
    uptime: float,
    version: str
) -> Dict[str, Any]:
    """
    Create a health check response.
    
    Args:
        status: Overall system status
        components: Health status of individual components
        uptime: System uptime in seconds
        version: System version
        
    Returns:
        Health response dictionary
    """
    return create_success_response(
        message=f"System status: {status}",
        data={
            "status": status,
            "components": components,
            "uptime": uptime,
            "version": version
        }
    )


def create_metrics_response(
    system_metrics: Dict[str, Any],
    engine_metrics: Dict[str, Any],
    service_metrics: Dict[str, Any],
    performance_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a metrics response.
    
    Args:
        system_metrics: System-level metrics
        engine_metrics: Engine-specific metrics
        service_metrics: Service-specific metrics
        performance_metrics: Performance metrics
        
    Returns:
        Metrics response dictionary
    """
    return create_success_response(
        message="Metrics retrieved successfully",
        data={
            "system_metrics": system_metrics,
            "engine_metrics": engine_metrics,
            "service_metrics": service_metrics,
            "performance_metrics": performance_metrics
        }
    )


def create_batch_response(
    results: list,
    successful_count: int,
    failed_count: int,
    total_time: float
) -> Dict[str, Any]:
    """
    Create a batch processing response.
    
    Args:
        results: Results for each request
        successful_count: Number of successful requests
        failed_count: Number of failed requests
        total_time: Total processing time
        
    Returns:
        Batch response dictionary
    """
    return create_success_response(
        message=f"Batch processing completed: {successful_count} successful, {failed_count} failed",
        data={
            "results": results,
            "successful_count": successful_count,
            "failed_count": failed_count,
            "total_time": total_time
        }
    )


# Legacy aliases for backward compatibility
SuccessResponse = create_success_response
ErrorResponse = create_error_response


__all__ = [
    "create_success_response",
    "create_error_response", 
    "create_orjson_response",
    "create_health_response",
    "create_metrics_response",
    "create_batch_response",
    "SuccessResponse",
    "ErrorResponse"
]


