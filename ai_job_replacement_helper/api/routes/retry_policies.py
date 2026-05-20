"""
Retry Policies endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.retry_policies import RetryPoliciesService, RetryStrategy

router = APIRouter()
retry_service = RetryPoliciesService()


@router.post("/create")
async def create_policy(
    name: str,
    max_attempts: int = 3,
    strategy: str = "exponential",
    base_delay_seconds: float = 1.0,
    max_delay_seconds: float = 60.0
) -> Dict[str, Any]:
    """Crear política de reintento"""
    try:
        strategy_enum = RetryStrategy(strategy)
        policy = retry_service.create_policy(
            name, max_attempts, strategy_enum, base_delay_seconds, max_delay_seconds
        )
        return {
            "name": policy.name,
            "max_attempts": policy.max_attempts,
            "strategy": policy.strategy.value,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




