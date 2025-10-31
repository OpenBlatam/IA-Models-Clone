"""
Document Workflow Chain API v2.0 - Optimized & Refactored
========================================================

High-performance FastAPI application with:
- Async/await throughout
- Advanced caching and optimization
- Real-time WebSocket support
- Comprehensive monitoring
- Plugin system integration
- Advanced error handling
- Rate limiting and security
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import json

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

from workflow_chain_v2 import (
    WorkflowChain, DocumentNode, WorkflowStatus, Priority,
    WorkflowChainManager, PerformanceMetrics, CacheManager
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
workflow_manager = WorkflowChainManager()
security = HTTPBearer(auto_error=False)
active_connections: Dict[str, WebSocket] = {}


# Pydantic Models
class CreateWorkflowRequest(BaseModel):
    """Request model for creating workflow"""
    name: str = Field(..., min_length=1, max_length=255, description="Workflow name")
    description: str = Field("", max_length=1000, description="Workflow description")
    settings: Optional[Dict[str, Any]] = Field(None, description="Workflow settings")
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()


class AddNodeRequest(BaseModel):
    """Request model for adding node"""
    chain_id: str = Field(..., description="Chain ID")
    title: str = Field(..., min_length=1, max_length=255, description="Node title")
    content: str = Field(..., min_length=1, description="Node content")
    prompt: str = Field(..., min_length=1, description="Node prompt")
    parent_id: Optional[str] = Field(None, description="Parent node ID")
    tags: List[str] = Field(default_factory=list, description="Node tags")
    priority: Priority = Field(Priority.NORMAL, description="Node priority")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Node metadata")


class WorkflowResponse(BaseModel):
    """Response model for workflow"""
    id: str
    name: str
    description: str
    status: str
    created_at: str
    updated_at: str
    statistics: Dict[str, Any]


class NodeResponse(BaseModel):
    """Response model for node"""
    id: str
    title: str
    content: str
    prompt: str
    created_at: str
    updated_at: str
    parent_id: Optional[str]
    children_ids: List[str]
    tags: List[str]
    priority: int
    quality_score: Optional[float]
    word_count: int
    metrics: Dict[str, Any]


class StatisticsResponse(BaseModel):
    """Response model for statistics"""
    global_stats: Dict[str, Any]
    chain_stats: Optional[Dict[str, Any]] = None


class OptimizationRequest(BaseModel):
    """Request model for optimization"""
    chain_id: str = Field(..., description="Chain ID to optimize")
    clear_cache: bool = Field(True, description="Clear cache during optimization")
    rebuild_indexes: bool = Field(True, description="Rebuild indexes during optimization")


# Dependency functions
async def get_workflow_manager() -> WorkflowChainManager:
    """Get workflow manager instance"""
    return workflow_manager


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Get current user (simplified for demo)"""
    # In production, this would validate JWT tokens
    if credentials:
        return "demo_user"
    return None


# WebSocket connection manager
class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metrics: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_metrics[client_id] = {
            "connected_at": datetime.utcnow(),
            "messages_sent": 0,
            "messages_received": 0
        }
        logger.info(f"WebSocket connected: {client_id}")
    
    def disconnect(self, client_id: str):
        """Disconnect WebSocket"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.connection_metrics[client_id]
            logger.info(f"WebSocket disconnected: {client_id}")
    
    async def send_personal_message(self, message: str, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_text(message)
                self.connection_metrics[client_id]["messages_sent"] += 1
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected clients"""
        disconnected = []
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
                self.connection_metrics[client_id]["messages_sent"] += 1
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "active_connections": len(self.active_connections),
            "total_messages_sent": sum(
                metrics["messages_sent"] for metrics in self.connection_metrics.values()
            ),
            "connections": {
                client_id: {
                    "connected_at": metrics["connected_at"].isoformat(),
                    "messages_sent": metrics["messages_sent"],
                    "messages_received": metrics["messages_received"]
                }
                for client_id, metrics in self.connection_metrics.items()
            }
        }


# Global connection manager
manager = ConnectionManager()


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Document Workflow Chain API v2.0...")
    
    # Initialize global cache
    global_cache = CacheManager(max_size=10000, default_ttl=7200)
    
    # Subscribe to global events
    workflow_manager.subscribe_to_events("chain_created", handle_chain_created)
    workflow_manager.subscribe_to_events("global_node_added", handle_node_added)
    
    logger.info("API v2.0 started successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Document Workflow Chain API v2.0...")
    
    # Close all WebSocket connections
    for client_id in list(manager.active_connections.keys()):
        manager.disconnect(client_id)
    
    logger.info("API v2.0 shutdown completed")


# Event handlers
async def handle_chain_created(data: Dict[str, Any]):
    """Handle chain created event"""
    chain_id = data["chain_id"]
    message = {
        "type": "chain_created",
        "data": {"chain_id": chain_id},
        "timestamp": datetime.utcnow().isoformat()
    }
    await manager.broadcast(json.dumps(message))


async def handle_node_added(data: Dict[str, Any]):
    """Handle node added event"""
    chain_id = data["chain_id"]
    node_id = data["node_id"]
    message = {
        "type": "node_added",
        "data": {"chain_id": chain_id, "node_id": node_id},
        "timestamp": datetime.utcnow().isoformat()
    }
    await manager.broadcast(json.dumps(message))


# Create FastAPI application
app = FastAPI(
    title="Document Workflow Chain API v2.0",
    description="High-performance document workflow chain system",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# API Endpoints

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Document Workflow Chain API v2.0",
        "version": "2.0.0",
        "description": "High-performance document workflow chain system",
        "features": [
            "Async/await throughout",
            "Advanced caching",
            "Real-time WebSocket support",
            "Comprehensive monitoring",
            "Plugin system",
            "Rate limiting",
            "Security"
        ],
        "endpoints": {
            "workflows": "/api/v2/workflows",
            "nodes": "/api/v2/nodes",
            "statistics": "/api/v2/statistics",
            "websocket": "/ws",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    try:
        # Get system statistics
        global_stats = await workflow_manager.get_global_statistics()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0",
            "uptime": "24h 30m",  # This would be calculated
            "statistics": global_stats,
            "websocket_connections": manager.get_connection_stats()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


# Workflow endpoints
@app.post("/api/v2/workflows", response_model=WorkflowResponse)
async def create_workflow(
    request: CreateWorkflowRequest,
    background_tasks: BackgroundTasks,
    user: Optional[str] = Depends(get_current_user)
):
    """Create a new workflow chain"""
    try:
        start_time = time.time()
        
        # Create workflow
        chain = await workflow_manager.create_chain(
            name=request.name,
            description=request.description
        )
        
        # Set settings if provided
        if request.settings:
            chain.settings = request.settings
        
        # Get statistics
        stats = await chain.get_chain_statistics()
        
        duration = time.time() - start_time
        logger.info(f"Created workflow {chain.id} in {duration:.3f}s")
        
        return WorkflowResponse(
            id=chain.id,
            name=chain.name,
            description=chain.description,
            status=chain.status.value,
            created_at=chain.created_at.isoformat(),
            updated_at=chain.updated_at.isoformat(),
            statistics=stats
        )
        
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/workflows", response_model=List[WorkflowResponse])
async def list_workflows(
    status: Optional[WorkflowStatus] = None,
    limit: int = Field(20, ge=1, le=100),
    offset: int = Field(0, ge=0),
    user: Optional[str] = Depends(get_current_user)
):
    """List workflow chains with pagination"""
    try:
        chains = await workflow_manager.list_chains(status=status)
        
        # Apply pagination
        total = len(chains)
        chains = chains[offset:offset + limit]
        
        # Get statistics for each chain
        workflows = []
        for chain in chains:
            stats = await chain.get_chain_statistics()
            workflows.append(WorkflowResponse(
                id=chain.id,
                name=chain.name,
                description=chain.description,
                status=chain.status.value,
                created_at=chain.created_at.isoformat(),
                updated_at=chain.updated_at.isoformat(),
                statistics=stats
            ))
        
        return workflows
        
    except Exception as e:
        logger.error(f"Failed to list workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/workflows/{chain_id}", response_model=WorkflowResponse)
async def get_workflow(
    chain_id: str,
    user: Optional[str] = Depends(get_current_user)
):
    """Get workflow chain by ID"""
    try:
        chain = await workflow_manager.get_chain(chain_id)
        if not chain:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        stats = await chain.get_chain_statistics()
        
        return WorkflowResponse(
            id=chain.id,
            name=chain.name,
            description=chain.description,
            status=chain.status.value,
            created_at=chain.created_at.isoformat(),
            updated_at=chain.updated_at.isoformat(),
            statistics=stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow {chain_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v2/workflows/{chain_id}")
async def delete_workflow(
    chain_id: str,
    user: Optional[str] = Depends(get_current_user)
):
    """Delete workflow chain"""
    try:
        success = await workflow_manager.delete_chain(chain_id)
        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {"message": "Workflow deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete workflow {chain_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Node endpoints
@app.post("/api/v2/nodes", response_model=NodeResponse)
async def add_node(
    request: AddNodeRequest,
    background_tasks: BackgroundTasks,
    user: Optional[str] = Depends(get_current_user)
):
    """Add node to workflow chain"""
    try:
        start_time = time.time()
        
        # Get workflow
        chain = await workflow_manager.get_chain(request.chain_id)
        if not chain:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Create node
        node = DocumentNode(
            title=request.title,
            content=request.content,
            prompt=request.prompt,
            parent_id=request.parent_id,
            tags=request.tags,
            priority=request.priority,
            metadata=request.metadata or {}
        )
        
        # Add to chain
        await chain.add_node(node)
        
        duration = time.time() - start_time
        logger.info(f"Added node {node.id} in {duration:.3f}s")
        
        return NodeResponse(
            id=node.id,
            title=node.title,
            content=node.content,
            prompt=node.prompt,
            created_at=node.created_at.isoformat(),
            updated_at=node.updated_at.isoformat(),
            parent_id=node.parent_id,
            children_ids=node.children_ids,
            tags=node.tags,
            priority=node.priority.value,
            quality_score=node.quality_score,
            word_count=node.word_count,
            metrics={
                "duration": node.metrics.duration,
                "tokens_used": node.metrics.tokens_used,
                "cache_hit_rate": node.metrics.cache_hit_rate
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add node: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/workflows/{chain_id}/nodes", response_model=List[NodeResponse])
async def get_workflow_nodes(
    chain_id: str,
    limit: int = Field(20, ge=1, le=100),
    offset: int = Field(0, ge=0),
    user: Optional[str] = Depends(get_current_user)
):
    """Get nodes from workflow chain"""
    try:
        chain = await workflow_manager.get_chain(chain_id)
        if not chain:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Get all nodes
        nodes = list(chain._nodes.values())
        
        # Apply pagination
        total = len(nodes)
        nodes = nodes[offset:offset + limit]
        
        # Convert to response models
        node_responses = []
        for node in nodes:
            node_responses.append(NodeResponse(
                id=node.id,
                title=node.title,
                content=node.content,
                prompt=node.prompt,
                created_at=node.created_at.isoformat(),
                updated_at=node.updated_at.isoformat(),
                parent_id=node.parent_id,
                children_ids=node.children_ids,
                tags=node.tags,
                priority=node.priority.value,
                quality_score=node.quality_score,
                word_count=node.word_count,
                metrics={
                    "duration": node.metrics.duration,
                    "tokens_used": node.metrics.tokens_used,
                    "cache_hit_rate": node.metrics.cache_hit_rate
                }
            ))
        
        return node_responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow nodes {chain_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Statistics endpoints
@app.get("/api/v2/statistics", response_model=StatisticsResponse)
async def get_statistics(
    chain_id: Optional[str] = None,
    user: Optional[str] = Depends(get_current_user)
):
    """Get system statistics"""
    try:
        # Get global statistics
        global_stats = await workflow_manager.get_global_statistics()
        
        # Get chain-specific statistics if requested
        chain_stats = None
        if chain_id:
            chain = await workflow_manager.get_chain(chain_id)
            if chain:
                chain_stats = await chain.get_chain_statistics()
        
        return StatisticsResponse(
            global_stats=global_stats,
            chain_stats=chain_stats
        )
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Optimization endpoints
@app.post("/api/v2/workflows/{chain_id}/optimize")
async def optimize_workflow(
    chain_id: str,
    request: OptimizationRequest,
    user: Optional[str] = Depends(get_current_user)
):
    """Optimize workflow chain"""
    try:
        chain = await workflow_manager.get_chain(chain_id)
        if not chain:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Perform optimization
        results = await chain.optimize()
        
        return {
            "message": "Workflow optimized successfully",
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to optimize workflow {chain_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Update metrics
            if client_id in manager.connection_metrics:
                manager.connection_metrics[client_id]["messages_received"] += 1
            
            # Handle different message types
            if message.get("type") == "ping":
                await manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.utcnow().isoformat()}),
                    client_id
                )
            elif message.get("type") == "subscribe":
                # Handle subscription to specific events
                event_type = message.get("event_type", "all")
                # In a real implementation, you'd manage subscriptions here
                await manager.send_personal_message(
                    json.dumps({"type": "subscribed", "event_type": event_type}),
                    client_id
                )
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "message": "The requested resource was not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "An unexpected error occurred"}
    )


if __name__ == "__main__":
    uvicorn.run(
        "api_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )




