"""
Distributed Router - Endpoints for distributed execution.

This module provides REST API endpoints for managing
distributed execution nodes and tasks.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends

from ..models import ErrorResponse
from ..auth import verify_token
from core.distributed import DistributedExecutor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["distributed"])

# Initialize manager (should be injected in production)
distributed_executor = DistributedExecutor()


@router.get("/nodes", response_model=dict)
async def list_nodes(
    token: str = Depends(verify_token),
):
    """
    List available nodes.
    
    Args:
        token: Authentication token
    
    Returns:
        Dictionary with nodes list
    """
    try:
        nodes = list(distributed_executor.nodes.values())
        return {"nodes": [n.to_dict() if hasattr(n, 'to_dict') else n for n in nodes]}
    except Exception as e:
        logger.exception("Error listing nodes")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nodes", response_model=dict)
async def register_node(
    node_id: str,
    host: str,
    port: int,
    token: str = Depends(verify_token),
):
    """
    Register a node.
    
    Args:
        node_id: Node identifier
        host: Node host
        port: Node port
        token: Authentication token
    
    Returns:
        Node dictionary
    """
    try:
        node = distributed_executor.register_node(node_id, host, port)
        return node.to_dict() if hasattr(node, 'to_dict') else node
    except Exception as e:
        logger.exception("Error registering node")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks", response_model=dict)
async def list_tasks(
    node_id: Optional[str] = None,
    token: str = Depends(verify_token),
):
    """
    List tasks.
    
    Args:
        node_id: Filter by node ID (optional)
        token: Authentication token
    
    Returns:
        Dictionary with tasks list
    """
    try:
        if node_id:
            tasks = distributed_executor.get_node_tasks(node_id)
        else:
            tasks = list(distributed_executor.tasks.values())
        return {"tasks": [t.to_dict() if hasattr(t, 'to_dict') else t for t in tasks]}
    except Exception as e:
        logger.exception("Error listing tasks")
        raise HTTPException(status_code=500, detail=str(e))












