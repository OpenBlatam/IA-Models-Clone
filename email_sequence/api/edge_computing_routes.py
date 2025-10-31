"""
Edge Computing Routes for Email Sequence System

This module provides API endpoints for edge computing capabilities
including distributed processing and edge AI inference.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from .schemas import ErrorResponse
from ..core.edge_computing_engine import (
    edge_computing_engine,
    EdgeNodeType,
    ProcessingPriority
)
from ..core.dependencies import get_current_user
from ..core.exceptions import EdgeComputingError

logger = logging.getLogger(__name__)

# Edge Computing router
edge_computing_router = APIRouter(
    prefix="/api/v1/edge-computing",
    tags=["Edge Computing"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)


@edge_computing_router.post("/nodes")
async def register_edge_node(
    node_id: str,
    node_type: EdgeNodeType,
    location: str,
    ip_address: str,
    port: int,
    capabilities: List[str],
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Register a new edge computing node.
    
    Args:
        node_id: Unique node identifier
        node_type: Type of edge node
        location: Geographic location
        ip_address: Node IP address
        port: Node port
        capabilities: List of node capabilities
        metadata: Additional node metadata
        
    Returns:
        Node registration result
    """
    try:
        node = await edge_computing_engine.register_edge_node(
            node_id=node_id,
            node_type=node_type,
            location=location,
            ip_address=ip_address,
            port=port,
            capabilities=capabilities,
            metadata=metadata
        )
        
        return {
            "status": "success",
            "node_id": node.node_id,
            "node_type": node.node_type.value,
            "location": node.location,
            "status": node.status.value,
            "message": "Edge node registered successfully"
        }
        
    except EdgeComputingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error registering edge node: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@edge_computing_router.get("/nodes")
async def list_edge_nodes():
    """
    List all registered edge computing nodes.
    
    Returns:
        List of edge nodes
    """
    try:
        nodes = []
        for node_id, node in edge_computing_engine.nodes.items():
            nodes.append({
                "node_id": node_id,
                "node_type": node.node_type.value,
                "location": node.location,
                "ip_address": node.ip_address,
                "port": node.port,
                "status": node.status.value,
                "capabilities": node.capabilities,
                "cpu_usage": node.cpu_usage,
                "memory_usage": node.memory_usage,
                "network_latency": node.network_latency,
                "last_heartbeat": node.last_heartbeat.isoformat(),
                "uptime": (datetime.utcnow() - node.created_at).total_seconds()
            })
        
        return {
            "status": "success",
            "nodes": nodes,
            "total_nodes": len(nodes)
        }
        
    except Exception as e:
        logger.error(f"Error listing edge nodes: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@edge_computing_router.get("/nodes/{node_id}")
async def get_edge_node_status(node_id: str):
    """
    Get edge node status and metrics.
    
    Args:
        node_id: Node ID
        
    Returns:
        Node status information
    """
    try:
        status = await edge_computing_engine.get_edge_node_status(node_id)
        return status
        
    except EdgeComputingError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting edge node status: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@edge_computing_router.post("/tasks/ai-inference")
async def submit_ai_inference_task(
    model_name: str,
    input_data: Dict[str, Any],
    priority: ProcessingPriority = ProcessingPriority.NORMAL
):
    """
    Submit AI inference task to edge nodes.
    
    Args:
        model_name: Name of the AI model
        input_data: Input data for inference
        priority: Task priority
        
    Returns:
        Task submission result
    """
    try:
        task_id = await edge_computing_engine.process_ai_inference(
            model_name=model_name,
            input_data=input_data,
            priority=priority
        )
        
        return {
            "status": "success",
            "task_id": task_id,
            "task_type": "ai_inference",
            "model_name": model_name,
            "priority": priority.value,
            "message": "AI inference task submitted successfully"
        }
        
    except EdgeComputingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error submitting AI inference task: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@edge_computing_router.post("/tasks/content-optimization")
async def submit_content_optimization_task(
    content: str,
    optimization_type: str,
    target_audience: Dict[str, Any],
    priority: ProcessingPriority = ProcessingPriority.NORMAL
):
    """
    Submit content optimization task to edge nodes.
    
    Args:
        content: Content to optimize
        optimization_type: Type of optimization
        target_audience: Target audience data
        priority: Task priority
        
    Returns:
        Task submission result
    """
    try:
        task_id = await edge_computing_engine.process_content_optimization(
            content=content,
            optimization_type=optimization_type,
            target_audience=target_audience,
            priority=priority
        )
        
        return {
            "status": "success",
            "task_id": task_id,
            "task_type": "content_optimization",
            "optimization_type": optimization_type,
            "priority": priority.value,
            "message": "Content optimization task submitted successfully"
        }
        
    except EdgeComputingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error submitting content optimization task: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@edge_computing_router.post("/tasks/real-time-analytics")
async def submit_real_time_analytics_task(
    analytics_data: Dict[str, Any],
    analysis_type: str,
    priority: ProcessingPriority = ProcessingPriority.HIGH
):
    """
    Submit real-time analytics task to edge nodes.
    
    Args:
        analytics_data: Analytics data to process
        analysis_type: Type of analysis
        priority: Task priority
        
    Returns:
        Task submission result
    """
    try:
        task_id = await edge_computing_engine.process_real_time_analytics(
            analytics_data=analytics_data,
            analysis_type=analysis_type,
            priority=priority
        )
        
        return {
            "status": "success",
            "task_id": task_id,
            "task_type": "real_time_analytics",
            "analysis_type": analysis_type,
            "priority": priority.value,
            "message": "Real-time analytics task submitted successfully"
        }
        
    except EdgeComputingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error submitting real-time analytics task: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@edge_computing_router.post("/tasks/generic")
async def submit_generic_task(
    task_type: str,
    data: Dict[str, Any],
    target_node_type: EdgeNodeType,
    priority: ProcessingPriority = ProcessingPriority.NORMAL,
    timeout_seconds: int = 300
):
    """
    Submit a generic task for edge processing.
    
    Args:
        task_type: Type of task to process
        data: Task data
        target_node_type: Target node type for processing
        priority: Task priority
        timeout_seconds: Task timeout in seconds
        
    Returns:
        Task submission result
    """
    try:
        task_id = await edge_computing_engine.submit_edge_task(
            task_type=task_type,
            data=data,
            target_node_type=target_node_type,
            priority=priority,
            timeout_seconds=timeout_seconds
        )
        
        return {
            "status": "success",
            "task_id": task_id,
            "task_type": task_type,
            "target_node_type": target_node_type.value,
            "priority": priority.value,
            "timeout_seconds": timeout_seconds,
            "message": "Generic task submitted successfully"
        }
        
    except EdgeComputingError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error submitting generic task: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@edge_computing_router.get("/tasks/{task_id}")
async def get_task_result(task_id: str):
    """
    Get the result of an edge task.
    
    Args:
        task_id: Task ID
        
    Returns:
        Task result
    """
    try:
        result = await edge_computing_engine.get_task_result(task_id)
        return result
        
    except EdgeComputingError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting task result: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@edge_computing_router.get("/tasks")
async def list_tasks():
    """
    List all tasks (active and completed).
    
    Returns:
        List of tasks
    """
    try:
        active_tasks = []
        for task_id, task in edge_computing_engine.active_tasks.items():
            active_tasks.append({
                "task_id": task_id,
                "task_type": task.task_type,
                "priority": task.priority.value,
                "target_node_type": task.target_node_type.value,
                "status": "active",
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None
            })
        
        completed_tasks = []
        for task_id, task in edge_computing_engine.completed_tasks.items():
            completed_tasks.append({
                "task_id": task_id,
                "task_type": task.task_type,
                "priority": task.priority.value,
                "target_node_type": task.target_node_type.value,
                "status": "completed",
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "processing_time": (
                    (task.completed_at - task.started_at).total_seconds()
                    if task.started_at and task.completed_at else None
                )
            })
        
        return {
            "status": "success",
            "active_tasks": active_tasks,
            "completed_tasks": completed_tasks,
            "total_active": len(active_tasks),
            "total_completed": len(completed_tasks)
        }
        
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@edge_computing_router.get("/stats")
async def get_edge_computing_stats():
    """
    Get edge computing engine statistics.
    
    Returns:
        Engine statistics
    """
    try:
        stats = await edge_computing_engine.get_edge_computing_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting edge computing stats: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@edge_computing_router.delete("/nodes/{node_id}")
async def unregister_edge_node(node_id: str):
    """
    Unregister an edge computing node.
    
    Args:
        node_id: Node ID to unregister
        
    Returns:
        Unregistration result
    """
    try:
        if node_id not in edge_computing_engine.nodes:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Edge node not found")
        
        # Remove node
        del edge_computing_engine.nodes[node_id]
        
        # Remove from cache
        await edge_computing_engine.cache_manager.delete(f"edge_node:{node_id}")
        
        return {
            "status": "success",
            "node_id": node_id,
            "message": "Edge node unregistered successfully"
        }
        
    except Exception as e:
        logger.error(f"Error unregistering edge node: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@edge_computing_router.post("/nodes/{node_id}/heartbeat")
async def update_node_heartbeat(node_id: str, metrics: Dict[str, Any]):
    """
    Update edge node heartbeat and metrics.
    
    Args:
        node_id: Node ID
        metrics: Node metrics (CPU, memory, etc.)
        
    Returns:
        Heartbeat update result
    """
    try:
        if node_id not in edge_computing_engine.nodes:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Edge node not found")
        
        node = edge_computing_engine.nodes[node_id]
        
        # Update metrics
        node.cpu_usage = metrics.get("cpu_usage", node.cpu_usage)
        node.memory_usage = metrics.get("memory_usage", node.memory_usage)
        node.network_latency = metrics.get("network_latency", node.network_latency)
        node.last_heartbeat = datetime.utcnow()
        
        # Update status if provided
        if "status" in metrics:
            from ..core.edge_computing_engine import EdgeNodeStatus
            node.status = EdgeNodeStatus(metrics["status"])
        
        return {
            "status": "success",
            "node_id": node_id,
            "message": "Node heartbeat updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error updating node heartbeat: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


# Error handlers for edge computing routes
@edge_computing_router.exception_handler(EdgeComputingError)
async def edge_computing_error_handler(request, exc):
    """Handle edge computing errors"""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=f"Edge computing error: {exc.message}",
            error_code="EDGE_COMPUTING_ERROR"
        ).dict()
    )






























