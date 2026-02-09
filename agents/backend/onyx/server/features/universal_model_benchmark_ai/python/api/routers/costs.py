"""
Costs Router - Endpoints for cost tracking.

This module provides REST API endpoints for tracking
and managing costs.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends

from ..models import ErrorResponse
from ..auth import verify_token
from core.cost_tracking import CostTracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/costs", tags=["costs"])

# Initialize manager (should be injected in production)
cost_tracker = CostTracker()


@router.get("", response_model=dict)
async def get_costs(
    model_name: Optional[str] = None,
    benchmark_name: Optional[str] = None,
    token: str = Depends(verify_token),
):
    """
    Get cost information.
    
    Args:
        model_name: Filter by model name (optional)
        benchmark_name: Filter by benchmark name (optional)
        token: Authentication token
    
    Returns:
        Dictionary with cost information
    """
    try:
        total = cost_tracker.get_total_cost(model_name, benchmark_name)
        breakdown = cost_tracker.get_cost_breakdown(model_name)
        budget_status = cost_tracker.get_budget_status()
        
        return {
            "total_cost": total,
            "breakdown": breakdown,
            "budget_status": budget_status,
        }
    except Exception as e:
        logger.exception("Error getting costs")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/budget", response_model=dict)
async def set_budget(
    budget: float,
    token: str = Depends(verify_token),
):
    """
    Set budget.
    
    Args:
        budget: Budget amount
        token: Authentication token
    
    Returns:
        Dictionary with budget status
    """
    try:
        cost_tracker.set_budget(budget)
        return {"budget": budget, "status": "set"}
    except Exception as e:
        logger.exception("Error setting budget")
        raise HTTPException(status_code=500, detail=str(e))












