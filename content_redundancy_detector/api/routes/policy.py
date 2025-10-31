"""
Policy Guardrails API Routes
Provides endpoints for policy validation and management
"""

from fastapi import APIRouter, Request, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List

try:
    from policy_guard import PolicyGuard, default_policy_guard, PolicyViolation
    POLICY_GUARD_AVAILABLE = True
except ImportError:
    POLICY_GUARD_AVAILABLE = False
    PolicyGuard = None
    default_policy_guard = None
    PolicyViolation = None

from utils.response_helpers import create_success_response, create_error_response, get_request_id

router = APIRouter(prefix="/policies", tags=["Policy Guardrails"])


class ContentValidationRequest(BaseModel):
    """Request model for content validation"""
    content: str = Field(..., description="Content to validate")
    user_id: Optional[str] = Field(None, description="User identifier")


class BatchValidationRequest(BaseModel):
    """Request model for batch validation"""
    items: List[str] = Field(..., description="List of items to validate")
    user_id: Optional[str] = Field(None, description="User identifier")


@router.post("/validate", summary="Validate content against policies")
async def validate_content(request: Request, payload: ContentValidationRequest):
    """Validate content against all configured policies"""
    if not POLICY_GUARD_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Policy guard system not available"
        )
    
    request_id = get_request_id(request)
    
    try:
        is_valid, violations = default_policy_guard.validate_content(
            payload.content,
            payload.user_id
        )
        
        violations_data = []
        if violations:
            violations_data = [
                {
                    "type": v.type.value,
                    "severity": v.severity,
                    "message": v.message,
                    "details": v.details or {}
                }
                for v in violations
            ]
        
        return create_success_response(
            {
                "is_valid": is_valid,
                "violations": violations_data,
                "violation_count": len(violations)
            },
            request_id=request_id
        )
    except Exception as e:
        return create_error_response(
            message="Content validation failed",
            details=str(e),
            request_id=request_id
        )


@router.post("/validate-batch", summary="Validate batch of items")
async def validate_batch(request: Request, payload: BatchValidationRequest):
    """Validate a batch of items against policies"""
    if not POLICY_GUARD_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Policy guard system not available"
        )
    
    request_id = get_request_id(request)
    
    try:
        is_valid, violations = default_policy_guard.validate_batch(
            payload.items,
            payload.user_id
        )
        
        violations_data = []
        if violations:
            violations_data = [
                {
                    "type": v.type.value,
                    "severity": v.severity,
                    "message": v.message,
                    "details": v.details or {}
                }
                for v in violations
            ]
        
        return create_success_response(
            {
                "is_valid": is_valid,
                "violations": violations_data,
                "violation_count": len(violations),
                "items_validated": len(payload.items)
            },
            request_id=request_id
        )
    except Exception as e:
        return create_error_response(
            message="Batch validation failed",
            details=str(e),
            request_id=request_id
        )


@router.get("/summary", summary="Get policy summary")
async def get_policy_summary(request: Request):
    """Get summary of all configured policies"""
    if not POLICY_GUARD_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Policy guard system not available"
        )
    
    request_id = get_request_id(request)
    
    try:
        summary = default_policy_guard.get_policy_summary()
        return create_success_response(summary, request_id=request_id)
    except Exception as e:
        return create_error_response(
            message="Failed to get policy summary",
            details=str(e),
            request_id=request_id
        )


