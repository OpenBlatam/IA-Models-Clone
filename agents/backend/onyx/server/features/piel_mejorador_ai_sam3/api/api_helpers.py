"""
API Helpers for Piel Mejorador AI SAM3 API
"""

from typing import Dict, Any, Optional
from fastapi import HTTPException
from ..core.piel_mejorador_agent import PielMejoradorAgent


def require_agent(agent: Optional[PielMejoradorAgent]) -> PielMejoradorAgent:
    """
    Ensure agent is initialized, raise HTTPException if not.
    
    Args:
        agent: Optional agent instance
        
    Returns:
        Agent instance
        
    Raises:
        HTTPException: If agent is not initialized
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return agent


def create_task_response(task_id: str, status: str = "submitted") -> Dict[str, Any]:
    """
    Create standard task response dictionary.
    
    Args:
        task_id: Task ID
        status: Task status (default: "submitted")
        
    Returns:
        Standard response dictionary
    """
    return {"task_id": task_id, "status": status}


async def handle_task_operation(
    agent: Optional[PielMejoradorAgent],
    operation_name: str,
    operation: callable,
    *args,
    **kwargs
) -> Any:
    """
    Handle task operations with consistent error handling.
    
    Args:
        agent: Optional agent instance
        operation_name: Name of operation for error messages
        operation: Function to execute
        *args: Positional arguments for operation
        **kwargs: Keyword arguments for operation
        
    Returns:
        Result of operation
        
    Raises:
        HTTPException: If agent not initialized or operation fails
    """
    require_agent(agent)
    
    try:
        return await operation(*args, **kwargs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in {operation_name}: {str(e)}")




