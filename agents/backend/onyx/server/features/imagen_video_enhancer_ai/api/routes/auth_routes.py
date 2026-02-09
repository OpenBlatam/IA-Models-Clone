"""
Authentication Routes
====================

API routes for authentication and authorization.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from ..dependencies import get_auth_manager, verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/generate-key")
async def generate_api_key(
    name: str,
    permissions: List[str],
    expires_days: Optional[int] = None,
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Generate a new API key (requires admin permission)."""
    auth_manager = get_auth_manager()
    
    if api_key and not auth_manager.check_permission(api_key, "admin"):
        raise HTTPException(status_code=403, detail="Admin permission required")
    
    try:
        new_key = auth_manager.generate_key(
            name=name,
            permissions=permissions,
            expires_days=expires_days
        )
        return JSONResponse({
            "success": True,
            "api_key": new_key,
            "warning": "Store this key securely, it won't be shown again"
        })
    except Exception as e:
        logger.error(f"Error generating API key: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/keys")
async def list_api_keys(api_key: Optional[str] = Depends(verify_api_key)):
    """List API keys (requires admin permission)."""
    auth_manager = get_auth_manager()
    
    if api_key and not auth_manager.check_permission(api_key, "admin"):
        raise HTTPException(status_code=403, detail="Admin permission required")
    
    try:
        keys = auth_manager.list_keys()
        return JSONResponse(keys)
    except Exception as e:
        logger.error(f"Error listing API keys: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




