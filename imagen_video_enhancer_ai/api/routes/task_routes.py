"""
Task Routes
===========

API routes for task management.
"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/task", tags=["tasks"])


from ..dependencies import get_agent


@router.get("/{task_id}/status")
async def get_task_status(task_id: str):
    """Get task status."""
    agent = get_agent()
    
    try:
        status = await agent.get_task_status(task_id)
        return create_success_response(data=status)
    except Exception as e:
        raise handle_route_error(e, "Error getting task status")


@router.get("/{task_id}/result")
async def get_task_result(task_id: str):
    """Get task result."""
    agent = get_agent()
    
    try:
        result = await agent.get_task_result(task_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Task result not found")
        return create_success_response(data=result)
    except HTTPException:
        raise
    except Exception as e:
        raise handle_route_error(e, "Error getting task result")

