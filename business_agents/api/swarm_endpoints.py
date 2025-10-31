"""
Swarm Intelligence API Endpoints
================================

API endpoints for swarm intelligence service.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ..services.swarm_intelligence_service import (
    SwarmIntelligenceService,
    Swarm,
    SwarmAgent,
    SwarmTask,
    SwarmCommunication,
    SwarmType,
    AgentBehavior,
    SwarmAlgorithm
)

logger = logging.getLogger(__name__)

# Create router
swarm_router = APIRouter(prefix="/swarm", tags=["Swarm Intelligence"])

# Pydantic models for request/response
class SwarmRequest(BaseModel):
    name: str
    swarm_type: SwarmType
    algorithm: SwarmAlgorithm
    objective_function: str
    constraints: Dict[str, Any] = {}
    parameters: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class SwarmAgentRequest(BaseModel):
    swarm_id: str
    position: List[float]
    velocity: List[float] = []
    behavior: AgentBehavior = AgentBehavior.EXPLORATION
    communication_range: float = 1.0
    metadata: Dict[str, Any] = {}

class OptimizationRequest(BaseModel):
    swarm_id: str
    max_iterations: int = 100
    convergence_threshold: float = 0.001

class SwarmResponse(BaseModel):
    swarm_id: str
    name: str
    swarm_type: str
    algorithm: str
    agents: List[str]
    global_best_position: List[float]
    global_best_fitness: float
    objective_function: str
    constraints: Dict[str, Any]
    parameters: Dict[str, Any]
    status: str
    created_at: datetime
    last_update: datetime
    metadata: Dict[str, Any]

class SwarmAgentResponse(BaseModel):
    agent_id: str
    swarm_id: str
    position: List[float]
    velocity: List[float]
    fitness: float
    best_position: List[float]
    best_fitness: float
    behavior: str
    neighbors: List[str]
    communication_range: float
    last_update: datetime
    metadata: Dict[str, Any]

class SwarmTaskResponse(BaseModel):
    task_id: str
    swarm_id: str
    task_type: str
    objective: str
    constraints: Dict[str, Any]
    parameters: Dict[str, Any]
    status: str
    progress: float
    result: Optional[Any]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    metadata: Dict[str, Any]

class ServiceStatusResponse(BaseModel):
    service_status: str
    total_swarms: int
    active_swarms: int
    total_agents: int
    total_tasks: int
    completed_tasks: int
    objective_functions: int
    optimization_algorithms: int
    communication_enabled: bool
    adaptation_enabled: bool
    learning_enabled: bool
    real_time_optimization: bool
    timestamp: str

# Dependency to get swarm intelligence service
async def get_swarm_service() -> SwarmIntelligenceService:
    """Get swarm intelligence service instance."""
    # This would be injected from your dependency injection system
    # For now, we'll create a mock instance
    from ..main import get_swarm_intelligence_service
    return await get_swarm_intelligence_service()

@swarm_router.post("/create", response_model=Dict[str, str])
async def create_swarm(
    request: SwarmRequest,
    swarm_service: SwarmIntelligenceService = Depends(get_swarm_service)
):
    """Create a new swarm."""
    try:
        swarm = Swarm(
            swarm_id="",
            name=request.name,
            swarm_type=request.swarm_type,
            algorithm=request.algorithm,
            agents=[],
            global_best_position=[],
            global_best_fitness=float('inf'),
            objective_function=request.objective_function,
            constraints=request.constraints,
            parameters=request.parameters,
            status="active",
            created_at=datetime.utcnow(),
            last_update=datetime.utcnow(),
            metadata=request.metadata
        )
        
        swarm_id = await swarm_service.create_swarm(swarm)
        
        return {"swarm_id": swarm_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Failed to create swarm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@swarm_router.post("/{swarm_id}/agents", response_model=Dict[str, str])
async def add_agent_to_swarm(
    swarm_id: str,
    request: SwarmAgentRequest,
    swarm_service: SwarmIntelligenceService = Depends(get_swarm_service)
):
    """Add agent to swarm."""
    try:
        agent = SwarmAgent(
            agent_id="",
            swarm_id=swarm_id,
            position=request.position,
            velocity=request.velocity or [0.0] * len(request.position),
            fitness=float('inf'),
            best_position=request.position.copy(),
            best_fitness=float('inf'),
            behavior=request.behavior,
            neighbors=[],
            communication_range=request.communication_range,
            last_update=datetime.utcnow(),
            metadata=request.metadata
        )
        
        agent_id = await swarm_service.add_agent_to_swarm(swarm_id, agent)
        
        return {"agent_id": agent_id, "status": "added"}
        
    except Exception as e:
        logger.error(f"Failed to add agent to swarm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@swarm_router.get("/{swarm_id}", response_model=SwarmResponse)
async def get_swarm(
    swarm_id: str,
    swarm_service: SwarmIntelligenceService = Depends(get_swarm_service)
):
    """Get swarm by ID."""
    try:
        swarm = await swarm_service.get_swarm(swarm_id)
        
        if not swarm:
            raise HTTPException(status_code=404, detail="Swarm not found")
            
        return SwarmResponse(
            swarm_id=swarm.swarm_id,
            name=swarm.name,
            swarm_type=swarm.swarm_type.value,
            algorithm=swarm.algorithm.value,
            agents=swarm.agents,
            global_best_position=swarm.global_best_position,
            global_best_fitness=swarm.global_best_fitness,
            objective_function=swarm.objective_function,
            constraints=swarm.constraints,
            parameters=swarm.parameters,
            status=swarm.status,
            created_at=swarm.created_at,
            last_update=swarm.last_update,
            metadata=swarm.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get swarm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@swarm_router.get("/", response_model=List[SwarmResponse])
async def list_swarms(
    swarm_type: Optional[SwarmType] = None,
    swarm_service: SwarmIntelligenceService = Depends(get_swarm_service)
):
    """List swarms."""
    try:
        swarms = await swarm_service.get_swarms(swarm_type)
        
        return [
            SwarmResponse(
                swarm_id=swarm.swarm_id,
                name=swarm.name,
                swarm_type=swarm.swarm_type.value,
                algorithm=swarm.algorithm.value,
                agents=swarm.agents,
                global_best_position=swarm.global_best_position,
                global_best_fitness=swarm.global_best_fitness,
                objective_function=swarm.objective_function,
                constraints=swarm.constraints,
                parameters=swarm.parameters,
                status=swarm.status,
                created_at=swarm.created_at,
                last_update=swarm.last_update,
                metadata=swarm.metadata
            )
            for swarm in swarms
        ]
        
    except Exception as e:
        logger.error(f"Failed to list swarms: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@swarm_router.get("/{swarm_id}/agents", response_model=List[SwarmAgentResponse])
async def get_swarm_agents(
    swarm_id: str,
    swarm_service: SwarmIntelligenceService = Depends(get_swarm_service)
):
    """Get swarm agents."""
    try:
        agents = await swarm_service.get_swarm_agents(swarm_id)
        
        return [
            SwarmAgentResponse(
                agent_id=agent.agent_id,
                swarm_id=agent.swarm_id,
                position=agent.position,
                velocity=agent.velocity,
                fitness=agent.fitness,
                best_position=agent.best_position,
                best_fitness=agent.best_fitness,
                behavior=agent.behavior.value,
                neighbors=agent.neighbors,
                communication_range=agent.communication_range,
                last_update=agent.last_update,
                metadata=agent.metadata
            )
            for agent in agents
        ]
        
    except Exception as e:
        logger.error(f"Failed to get swarm agents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@swarm_router.post("/{swarm_id}/optimize", response_model=SwarmTaskResponse)
async def run_optimization(
    swarm_id: str,
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    swarm_service: SwarmIntelligenceService = Depends(get_swarm_service)
):
    """Run swarm optimization."""
    try:
        task = await swarm_service.run_optimization(
            swarm_id=swarm_id,
            max_iterations=request.max_iterations
        )
        
        return SwarmTaskResponse(
            task_id=task.task_id,
            swarm_id=task.swarm_id,
            task_type=task.task_type,
            objective=task.objective,
            constraints=task.constraints,
            parameters=task.parameters,
            status=task.status,
            progress=task.progress,
            result=task.result,
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            metadata=task.metadata
        )
        
    except Exception as e:
        logger.error(f"Failed to run optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@swarm_router.get("/tasks/{task_id}", response_model=SwarmTaskResponse)
async def get_optimization_task(
    task_id: str,
    swarm_service: SwarmIntelligenceService = Depends(get_swarm_service)
):
    """Get optimization task."""
    try:
        if task_id not in swarm_service.swarm_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
            
        task = swarm_service.swarm_tasks[task_id]
        
        return SwarmTaskResponse(
            task_id=task.task_id,
            swarm_id=task.swarm_id,
            task_type=task.task_type,
            objective=task.objective,
            constraints=task.constraints,
            parameters=task.parameters,
            status=task.status,
            progress=task.progress,
            result=task.result,
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            metadata=task.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimization task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@swarm_router.get("/tasks", response_model=List[SwarmTaskResponse])
async def list_optimization_tasks(
    swarm_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    swarm_service: SwarmIntelligenceService = Depends(get_swarm_service)
):
    """List optimization tasks."""
    try:
        tasks = list(swarm_service.swarm_tasks.values())
        
        if swarm_id:
            tasks = [t for t in tasks if t.swarm_id == swarm_id]
            
        if status:
            tasks = [t for t in tasks if t.status == status]
            
        return [
            SwarmTaskResponse(
                task_id=task.task_id,
                swarm_id=task.swarm_id,
                task_type=task.task_type,
                objective=task.objective,
                constraints=task.constraints,
                parameters=task.parameters,
                status=task.status,
                progress=task.progress,
                result=task.result,
                created_at=task.created_at,
                started_at=task.started_at,
                completed_at=task.completed_at,
                metadata=task.metadata
            )
            for task in tasks[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list optimization tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@swarm_router.get("/status", response_model=ServiceStatusResponse)
async def get_service_status(
    swarm_service: SwarmIntelligenceService = Depends(get_swarm_service)
):
    """Get swarm intelligence service status."""
    try:
        status = await swarm_service.get_service_status()
        
        return ServiceStatusResponse(
            service_status=status["service_status"],
            total_swarms=status["total_swarms"],
            active_swarms=status["active_swarms"],
            total_agents=status["total_agents"],
            total_tasks=status["total_tasks"],
            completed_tasks=status["completed_tasks"],
            objective_functions=status["objective_functions"],
            optimization_algorithms=status["optimization_algorithms"],
            communication_enabled=status["communication_enabled"],
            adaptation_enabled=status["adaptation_enabled"],
            learning_enabled=status["learning_enabled"],
            real_time_optimization=status["real_time_optimization"],
            timestamp=status["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get service status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@swarm_router.get("/objective-functions", response_model=Dict[str, Any])
async def get_objective_functions(
    swarm_service: SwarmIntelligenceService = Depends(get_swarm_service)
):
    """Get available objective functions."""
    try:
        return swarm_service.objective_functions
        
    except Exception as e:
        logger.error(f"Failed to get objective functions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@swarm_router.get("/algorithms", response_model=Dict[str, Any])
async def get_optimization_algorithms(
    swarm_service: SwarmIntelligenceService = Depends(get_swarm_service)
):
    """Get available optimization algorithms."""
    try:
        return swarm_service.optimization_algorithms
        
    except Exception as e:
        logger.error(f"Failed to get optimization algorithms: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@swarm_router.get("/swarm-types", response_model=List[str])
async def get_swarm_types():
    """Get available swarm types."""
    return [st.value for st in SwarmType]

@swarm_router.get("/agent-behaviors", response_model=List[str])
async def get_agent_behaviors():
    """Get available agent behaviors."""
    return [ab.value for ab in AgentBehavior]

@swarm_router.get("/algorithms", response_model=List[str])
async def get_swarm_algorithms():
    """Get available swarm algorithms."""
    return [sa.value for sa in SwarmAlgorithm]

@swarm_router.post("/{swarm_id}/agents/{agent_id}/update")
async def update_agent_position(
    swarm_id: str,
    agent_id: str,
    position: List[float],
    swarm_service: SwarmIntelligenceService = Depends(get_swarm_service)
):
    """Update agent position."""
    try:
        if agent_id not in swarm_service.swarm_agents:
            raise HTTPException(status_code=404, detail="Agent not found")
            
        agent = swarm_service.swarm_agents[agent_id]
        agent.position = position
        agent.last_update = datetime.utcnow()
        
        return {"status": "updated", "agent_id": agent_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update agent position: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@swarm_router.delete("/{swarm_id}")
async def delete_swarm(
    swarm_id: str,
    swarm_service: SwarmIntelligenceService = Depends(get_swarm_service)
):
    """Delete a swarm."""
    try:
        if swarm_id not in swarm_service.swarms:
            raise HTTPException(status_code=404, detail="Swarm not found")
            
        # Remove agents
        swarm = swarm_service.swarms[swarm_id]
        for agent_id in swarm.agents:
            if agent_id in swarm_service.swarm_agents:
                del swarm_service.swarm_agents[agent_id]
                
        # Remove swarm
        del swarm_service.swarms[swarm_id]
        
        return {"status": "deleted", "swarm_id": swarm_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete swarm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@swarm_router.delete("/{swarm_id}/agents/{agent_id}")
async def remove_agent_from_swarm(
    swarm_id: str,
    agent_id: str,
    swarm_service: SwarmIntelligenceService = Depends(get_swarm_service)
):
    """Remove agent from swarm."""
    try:
        if swarm_id not in swarm_service.swarms:
            raise HTTPException(status_code=404, detail="Swarm not found")
            
        if agent_id not in swarm_service.swarm_agents:
            raise HTTPException(status_code=404, detail="Agent not found")
            
        # Remove from swarm
        swarm = swarm_service.swarms[swarm_id]
        if agent_id in swarm.agents:
            swarm.agents.remove(agent_id)
            
        # Remove agent
        del swarm_service.swarm_agents[agent_id]
        
        return {"status": "removed", "agent_id": agent_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove agent from swarm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

























