"""
Webhooks Router
"""

from typing import Dict, Any
from fastapi import APIRouter

router = APIRouter()


@router.post("/register", response_model=Dict[str, Any])
async def register_webhook() -> Dict[str, Any]:
    """Register webhook"""
    # TODO: Implement webhooks
    return {"success": True, "data": None}






