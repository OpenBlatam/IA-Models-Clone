"""
Edge Computing API Endpoints
============================

REST API endpoints for edge computing integration,
distributed processing, and edge analytics.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, WebSocket, WebSocketDisconnect
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from ..services.edge_computing_service import (
    EdgeComputingService, EdgeNodeType, EdgeNodeStatus, ProcessingType, DataType,
    EdgeNode, EdgeTask, EdgeData, EdgeAnalytics
)
from ..middleware.auth_middleware import get_current_user, require_permission, User

# Create API router
router = APIRouter(prefix="/edge", tags=["Edge Computing"])

# Pydantic models
class EdgeNodeRegistrationRequest(BaseModel):
    name: str = Field(..., description="Node name")
    node_type: str = Field(..., description="Node type")
    location: Dict[str, float] = Field(..., description="Node location")
    capabilities: List[str] = Field(default_factory=list, description="Node capabilities")
    resources: Dict[str, Any] = Field(default_factory=dict, description="Node resources")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Node configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Node metadata")

class EdgeTaskRequest(BaseModel):
    node_id: str = Field(..., description="Target node ID")
    task_type: str = Field(..., description="Task type")
    data_type: str = Field(..., description="Data type")
    data: Any = Field(..., description="Task data")
    priority: int = Field(1, description="Task priority")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Task parameters")

class EdgeAnalyticsRequest(BaseModel):
    node_id: str = Field(..., description="Target node ID")
    analytics_type: str = Field(..., description="Analytics type")
    data: Any = Field(..., description="Data to analyze")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Analytics parameters")

class EdgeDataQueryRequest(BaseModel):
    node_id: str = Field(..., description="Node ID")
    data_type: Optional[str] = Field(None, description="Data type filter")
    limit: int = Field(100, description="Maximum number of records")

# Global edge computing service instance
edge_service = None

def get_edge_service() -> EdgeComputingService:
    """Get global edge computing service instance."""
    global edge_service
    if edge_service is None:
        edge_service = EdgeComputingService({
            "edge": {
                "max_nodes": 1000,
                "max_tasks_per_node": 100,
                "processing_timeout": 300,
                "data_retention_days": 30,
                "encryption_enabled": True
            }
        })
    return edge_service

# API Endpoints

@router.post("/initialize", response_model=Dict[str, str])
async def initialize_edge_service(
    current_user: User = Depends(require_permission("edge:manage"))
):
    """Initialize the edge computing service."""
    
    edge_service = get_edge_service()
    
    try:
        await edge_service.initialize()
        return {"message": "Edge Computing Service initialized successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize edge service: {str(e)}")

@router.get("/status", response_model=Dict[str, Any])
async def get_edge_status(
    current_user: User = Depends(require_permission("edge:view"))
):
    """Get edge computing service status."""
    
    edge_service = get_edge_service()
    
    try:
        status = await edge_service.get_service_status()
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get edge status: {str(e)}")

@router.post("/nodes/register", response_model=Dict[str, str])
async def register_edge_node(
    request: EdgeNodeRegistrationRequest,
    current_user: User = Depends(require_permission("edge:manage"))
):
    """Register a new edge node."""
    
    edge_service = get_edge_service()
    
    try:
        # Convert string to enum
        node_type = EdgeNodeType(request.node_type)
        
        # Create edge node
        node = EdgeNode(
            node_id="",  # Will be generated
            name=request.name,
            node_type=node_type,
            status=EdgeNodeStatus.ONLINE,
            location=request.location,
            capabilities=request.capabilities,
            resources=request.resources,
            configuration=request.configuration,
            last_seen=datetime.utcnow(),
            metadata=request.metadata
        )
        
        # Register node
        node_id = await edge_service.register_edge_node(node)
        
        return {
            "message": "Edge node registered successfully",
            "node_id": node_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register edge node: {str(e)}")

@router.delete("/nodes/{node_id}", response_model=Dict[str, str])
async def unregister_edge_node(
    node_id: str,
    current_user: User = Depends(require_permission("edge:manage"))
):
    """Unregister an edge node."""
    
    edge_service = get_edge_service()
    
    try:
        success = await edge_service.unregister_edge_node(node_id)
        
        if success:
            return {"message": f"Edge node {node_id} unregistered successfully"}
        else:
            raise HTTPException(status_code=404, detail="Edge node not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unregister edge node: {str(e)}")

@router.get("/nodes", response_model=List[Dict[str, Any]])
async def get_edge_nodes(
    node_type: Optional[str] = Query(None, description="Filter by node type"),
    status: Optional[str] = Query(None, description="Filter by node status"),
    current_user: User = Depends(require_permission("edge:view"))
):
    """Get edge nodes."""
    
    edge_service = get_edge_service()
    
    try:
        # Convert string to enum if provided
        node_type_enum = EdgeNodeType(node_type) if node_type else None
        
        # Get nodes
        nodes = await edge_service.get_edge_nodes(node_type_enum)
        
        # Filter by status if provided
        if status:
            status_enum = EdgeNodeStatus(status)
            nodes = [n for n in nodes if n.status == status_enum]
        
        result = []
        for node in nodes:
            node_dict = {
                "node_id": node.node_id,
                "name": node.name,
                "node_type": node.node_type.value,
                "status": node.status.value,
                "location": node.location,
                "capabilities": node.capabilities,
                "resources": node.resources,
                "configuration": node.configuration,
                "last_seen": node.last_seen.isoformat(),
                "metadata": node.metadata
            }
            result.append(node_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get edge nodes: {str(e)}")

@router.get("/nodes/{node_id}", response_model=Dict[str, Any])
async def get_edge_node(
    node_id: str,
    current_user: User = Depends(require_permission("edge:view"))
):
    """Get specific edge node."""
    
    edge_service = get_edge_service()
    
    try:
        node = await edge_service.get_edge_node(node_id)
        
        if not node:
            raise HTTPException(status_code=404, detail="Edge node not found")
        
        return {
            "node_id": node.node_id,
            "name": node.name,
            "node_type": node.node_type.value,
            "status": node.status.value,
            "location": node.location,
            "capabilities": node.capabilities,
            "resources": node.resources,
            "configuration": node.configuration,
            "last_seen": node.last_seen.isoformat(),
            "metadata": node.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get edge node: {str(e)}")

@router.post("/tasks/submit", response_model=Dict[str, Any])
async def submit_edge_task(
    request: EdgeTaskRequest,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(require_permission("edge:execute"))
):
    """Submit task to edge node."""
    
    edge_service = get_edge_service()
    
    try:
        # Convert string to enum
        task_type = ProcessingType(request.task_type)
        data_type = DataType(request.data_type)
        
        if background_tasks:
            # Execute in background
            background_tasks.add_task(
                edge_service.submit_edge_task,
                request.node_id,
                task_type,
                data_type,
                request.data,
                request.priority,
                request.parameters
            )
            return {
                "message": "Edge task submitted in background",
                "node_id": request.node_id,
                "task_type": request.task_type
            }
        else:
            # Execute synchronously
            task_id = await edge_service.submit_edge_task(
                node_id=request.node_id,
                task_type=task_type,
                data_type=data_type,
                data=request.data,
                priority=request.priority,
                parameters=request.parameters
            )
            
            return {
                "task_id": task_id,
                "node_id": request.node_id,
                "task_type": request.task_type,
                "data_type": request.data_type,
                "priority": request.priority,
                "status": "submitted"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit edge task: {str(e)}")

@router.get("/tasks", response_model=List[Dict[str, Any]])
async def get_edge_tasks(
    node_id: Optional[str] = Query(None, description="Filter by node ID"),
    status: Optional[str] = Query(None, description="Filter by task status"),
    limit: int = Query(100, description="Maximum number of tasks"),
    current_user: User = Depends(require_permission("edge:view"))
):
    """Get edge tasks."""
    
    edge_service = get_edge_service()
    
    try:
        tasks = await edge_service.get_edge_tasks(node_id)
        
        # Filter by status if provided
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        # Limit results
        limited_tasks = tasks[-limit:] if limit else tasks
        
        result = []
        for task in limited_tasks:
            task_dict = {
                "task_id": task.task_id,
                "node_id": task.node_id,
                "task_type": task.task_type.value,
                "data_type": task.data_type.value,
                "priority": task.priority,
                "status": task.status,
                "result": task.result,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "execution_time": task.execution_time,
                "parameters": task.parameters
            }
            result.append(task_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get edge tasks: {str(e)}")

@router.get("/tasks/{task_id}", response_model=Dict[str, Any])
async def get_edge_task(
    task_id: str,
    current_user: User = Depends(require_permission("edge:view"))
):
    """Get specific edge task."""
    
    edge_service = get_edge_service()
    
    try:
        task = await edge_service.get_edge_task(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="Edge task not found")
        
        return {
            "task_id": task.task_id,
            "node_id": task.node_id,
            "task_type": task.task_type.value,
            "data_type": task.data_type.value,
            "priority": task.priority,
            "data": task.data,
            "status": task.status,
            "result": task.result,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "execution_time": task.execution_time,
            "parameters": task.parameters
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get edge task: {str(e)}")

@router.get("/data", response_model=List[Dict[str, Any]])
async def get_edge_data(
    node_id: str = Query(..., description="Node ID"),
    data_type: Optional[str] = Query(None, description="Filter by data type"),
    limit: int = Query(100, description="Maximum number of records"),
    current_user: User = Depends(require_permission("edge:view"))
):
    """Get edge data."""
    
    edge_service = get_edge_service()
    
    try:
        # Convert string to enum if provided
        data_type_enum = DataType(data_type) if data_type else None
        
        # Get edge data
        data = await edge_service.get_edge_data(node_id, data_type_enum, limit)
        
        result = []
        for data_point in data:
            data_dict = {
                "data_id": data_point.data_id,
                "node_id": data_point.node_id,
                "data_type": data_point.data_type.value,
                "data": data_point.data,
                "size": data_point.size,
                "timestamp": data_point.timestamp.isoformat(),
                "quality": data_point.quality,
                "metadata": data_point.metadata
            }
            result.append(data_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get edge data: {str(e)}")

@router.post("/analytics/run", response_model=Dict[str, Any])
async def run_edge_analytics(
    request: EdgeAnalyticsRequest,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(require_permission("edge:execute"))
):
    """Run analytics on edge node."""
    
    edge_service = get_edge_service()
    
    try:
        if background_tasks:
            # Execute in background
            background_tasks.add_task(
                edge_service.run_edge_analytics,
                request.node_id,
                request.analytics_type,
                request.data,
                request.parameters
            )
            return {
                "message": "Edge analytics started in background",
                "node_id": request.node_id,
                "analytics_type": request.analytics_type
            }
        else:
            # Execute synchronously
            analytics = await edge_service.run_edge_analytics(
                node_id=request.node_id,
                analytics_type=request.analytics_type,
                data=request.data,
                parameters=request.parameters
            )
            
            return {
                "analytics_id": analytics.analytics_id,
                "node_id": analytics.node_id,
                "analytics_type": analytics.analytics_type,
                "result": analytics.result,
                "confidence": analytics.confidence,
                "processing_time": analytics.processing_time,
                "timestamp": analytics.timestamp.isoformat()
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run edge analytics: {str(e)}")

@router.get("/analytics", response_model=List[Dict[str, Any]])
async def get_edge_analytics(
    node_id: Optional[str] = Query(None, description="Filter by node ID"),
    limit: int = Query(100, description="Maximum number of records"),
    current_user: User = Depends(require_permission("edge:view"))
):
    """Get edge analytics results."""
    
    edge_service = get_edge_service()
    
    try:
        analytics = list(edge_service.edge_analytics.values())
        
        # Filter by node ID if provided
        if node_id:
            analytics = [a for a in analytics if a.node_id == node_id]
        
        # Limit results
        limited_analytics = analytics[-limit:] if limit else analytics
        
        result = []
        for analytics_result in limited_analytics:
            analytics_dict = {
                "analytics_id": analytics_result.analytics_id,
                "node_id": analytics_result.node_id,
                "analytics_type": analytics_result.analytics_type,
                "result": analytics_result.result,
                "confidence": analytics_result.confidence,
                "processing_time": analytics_result.processing_time,
                "timestamp": analytics_result.timestamp.isoformat()
            }
            result.append(analytics_dict)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get edge analytics: {str(e)}")

@router.get("/analytics/{analytics_id}", response_model=Dict[str, Any])
async def get_edge_analytics_result(
    analytics_id: str,
    current_user: User = Depends(require_permission("edge:view"))
):
    """Get specific edge analytics result."""
    
    edge_service = get_edge_service()
    
    try:
        analytics = edge_service.edge_analytics.get(analytics_id)
        
        if not analytics:
            raise HTTPException(status_code=404, detail="Edge analytics result not found")
        
        return {
            "analytics_id": analytics.analytics_id,
            "node_id": analytics.node_id,
            "analytics_type": analytics.analytics_type,
            "result": analytics.result,
            "confidence": analytics.confidence,
            "processing_time": analytics.processing_time,
            "timestamp": analytics.timestamp.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get edge analytics result: {str(e)}")

@router.get("/analytics/types", response_model=List[Dict[str, Any]])
async def get_analytics_types(
    current_user: User = Depends(require_permission("edge:view"))
):
    """Get available analytics types."""
    
    try:
        analytics_types = [
            {
                "type": "real_time",
                "name": "Real-Time Analytics",
                "description": "Real-time data processing and analysis",
                "use_cases": ["monitoring", "alerting", "live_dashboards"]
            },
            {
                "type": "batch",
                "name": "Batch Analytics",
                "description": "Batch processing of historical data",
                "use_cases": ["reporting", "trend_analysis", "data_mining"]
            },
            {
                "type": "stream",
                "name": "Stream Analytics",
                "description": "Continuous stream processing",
                "use_cases": ["event_processing", "pattern_detection", "anomaly_detection"]
            },
            {
                "type": "ml_inference",
                "name": "ML Inference",
                "description": "Machine learning model inference",
                "use_cases": ["prediction", "classification", "recommendation"]
            },
            {
                "type": "data_fusion",
                "name": "Data Fusion",
                "description": "Multi-source data integration and fusion",
                "use_cases": ["sensor_fusion", "data_integration", "correlation_analysis"]
            },
            {
                "type": "filtering",
                "name": "Data Filtering",
                "description": "Data filtering and preprocessing",
                "use_cases": ["noise_reduction", "data_cleaning", "feature_extraction"]
            }
        ]
        
        return analytics_types
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics types: {str(e)}")

@router.get("/analytics", response_model=Dict[str, Any])
async def get_edge_analytics_overview(
    current_user: User = Depends(require_permission("edge:view"))
):
    """Get edge analytics overview."""
    
    edge_service = get_edge_service()
    
    try:
        # Get service status
        status = await edge_service.get_service_status()
        
        # Get analytics
        analytics = list(edge_service.edge_analytics.values())
        
        # Calculate analytics overview
        overview = {
            "total_analytics": len(analytics),
            "analytics_by_type": {},
            "analytics_by_node": {},
            "average_confidence": sum(a.confidence for a in analytics) / max(len(analytics), 1),
            "average_processing_time": sum(a.processing_time for a in analytics) / max(len(analytics), 1),
            "total_nodes": status.get("total_nodes", 0),
            "online_nodes": status.get("online_nodes", 0),
            "total_tasks": status.get("total_tasks", 0),
            "completed_tasks": status.get("completed_tasks", 0),
            "total_data_points": status.get("total_data_points", 0),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Calculate analytics by type
        for analytics_result in analytics:
            analytics_type = analytics_result.analytics_type
            if analytics_type not in overview["analytics_by_type"]:
                overview["analytics_by_type"][analytics_type] = 0
            overview["analytics_by_type"][analytics_type] += 1
            
        # Calculate analytics by node
        for analytics_result in analytics:
            node_id = analytics_result.node_id
            if node_id not in overview["analytics_by_node"]:
                overview["analytics_by_node"][node_id] = 0
            overview["analytics_by_node"][node_id] += 1
        
        return overview
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get edge analytics overview: {str(e)}")

@router.websocket("/ws/{node_id}")
async def websocket_endpoint(websocket: WebSocket, node_id: str):
    """WebSocket endpoint for real-time edge node communication."""
    
    edge_service = get_edge_service()
    
    try:
        await websocket.accept()
        
        # Send welcome message
        await websocket.send_json({
            "type": "connection",
            "message": f"Connected to edge node {node_id}",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_json()
                
                # Handle different message types
                if data.get("type") == "task":
                    # Process task
                    task_type = ProcessingType(data.get("task_type", "real_time"))
                    data_type = DataType(data.get("data_type", "sensor"))
                    
                    task_id = await edge_service.submit_edge_task(
                        node_id=node_id,
                        task_type=task_type,
                        data_type=data_type,
                        data=data.get("data", {}),
                        parameters=data.get("parameters", {})
                    )
                    
                    # Send response
                    await websocket.send_json({
                        "type": "task_response",
                        "task_id": task_id,
                        "status": "submitted",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                elif data.get("type") == "analytics":
                    # Process analytics
                    analytics = await edge_service.run_edge_analytics(
                        node_id=node_id,
                        analytics_type=data.get("analytics_type", "general"),
                        data=data.get("data", {}),
                        parameters=data.get("parameters", {})
                    )
                    
                    # Send response
                    await websocket.send_json({
                        "type": "analytics_response",
                        "analytics_id": analytics.analytics_id,
                        "result": analytics.result,
                        "confidence": analytics.confidence,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                elif data.get("type") == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}")
    finally:
        # Clean up connection
        pass

@router.get("/health", response_model=Dict[str, Any])
async def edge_health_check():
    """Edge computing service health check."""
    
    edge_service = get_edge_service()
    
    try:
        # Check if service is initialized
        initialized = hasattr(edge_service, 'edge_network') and edge_service.edge_network is not None
        
        # Get service status
        status = await edge_service.get_service_status()
        
        return {
            "status": "healthy" if initialized else "initializing",
            "initialized": initialized,
            "total_nodes": status.get("total_nodes", 0),
            "online_nodes": status.get("online_nodes", 0),
            "total_tasks": status.get("total_tasks", 0),
            "completed_tasks": status.get("completed_tasks", 0),
            "queue_size": status.get("queue_size", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }




























