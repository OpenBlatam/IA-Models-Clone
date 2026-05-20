"""
Similarity Router - Text similarity endpoints
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends

from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/", response_model=Dict[str, Any])
async def check_similarity(
    text1: str,
    text2: str,
    threshold: float = 0.8
) -> Dict[str, Any]:
    """Check similarity between two texts"""
    # TODO: Implement with similarity service
    raise HTTPException(status_code=501, detail="Not yet implemented - use legacy endpoint")






