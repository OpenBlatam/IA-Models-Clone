"""
System API Router
================

FastAPI router for system management and monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
from pydantic import BaseModel
import logging

from ..business_agents import BusinessAgentManager
from ..services import HealthService, SystemInfoService, MetricsService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["System"])

# Dependency to get agent manager
def get_agent_manager() -> BusinessAgentManager:
    """Get the global agent manager instance."""
    from ..main import app
    return app.state.agent_manager

# Response Models
class SystemInfoResponse(BaseModel):
    name: str
    version: str
    description: str
    features: List[str]
    business_areas: List[str]
    endpoints: Dict[str, str]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    components: Dict[str, str]
    metrics: Dict[str, Any]

class MetricsResponse(BaseModel):
    agents: Dict[str, Any]
    workflows: Dict[str, Any]
    documents: Dict[str, Any]
    system: Dict[str, Any]

# Endpoints
@router.get("/info", response_model=SystemInfoResponse)
async def get_system_info():
    """Get system information."""
    
    try:
        return SystemInfoResponse(
            name="Business Agents System",
            version="1.0.0",
            description="Comprehensive agent system for all business areas with workflow management and document generation",
            features=[
                "Workflow Management",
                "Document Generation", 
                "Agent Coordination",
                "Real-time Execution",
                "Multi-business Area Support"
            ],
            business_areas=[
                "marketing",
                "sales",
                "operations", 
                "hr",
                "finance",
                "legal",
                "technical",
                "content",
                "customer_service",
                "product_development",
                "strategy",
                "compliance"
            ],
            endpoints={
                "health": "/health",
                "docs": "/docs",
                "api": "/business-agents"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get system info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")

@router.get("/health", response_model=HealthResponse)
async def get_system_health(
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Get system health status."""
    
    try:
        health_service = HealthService(agent_manager)
        health_status = await health_service.get_health_status()
        
        return HealthResponse(
            status=health_status["status"],
            timestamp=health_status["timestamp"],
            version=health_status.get("version", "1.0.0"),
            components=health_status.get("components", {}),
            metrics=health_status.get("metrics", {})
        )
        
    except Exception as e:
        logger.error(f"Failed to get system health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")

@router.get("/metrics", response_model=MetricsResponse)
async def get_system_metrics(
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Get system metrics."""
    
    try:
        metrics_service = MetricsService(agent_manager)
        metrics = await metrics_service.get_system_metrics()
        
        return MetricsResponse(
            agents=metrics.get("agents", {}),
            workflows=metrics.get("workflows", {}),
            documents=metrics.get("documents", {}),
            system=metrics.get("system", {})
        )
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")

@router.get("/status")
async def get_system_status(
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Get detailed system status."""
    
    try:
        # Get basic system info
        agents = agent_manager.list_agents()
        workflows = agent_manager.list_workflows()
        
        # Calculate status
        active_agents = len([a for a in agents if a.is_active])
        total_agents = len(agents)
        
        active_workflows = len([w for w in workflows if w.status.value == "active"])
        completed_workflows = len([w for w in workflows if w.status.value == "completed"])
        failed_workflows = len([w for w in workflows if w.status.value == "failed"])
        
        return {
            "status": "operational",
            "agents": {
                "total": total_agents,
                "active": active_agents,
                "inactive": total_agents - active_agents
            },
            "workflows": {
                "total": len(workflows),
                "active": active_workflows,
                "completed": completed_workflows,
                "failed": failed_workflows
            },
            "business_areas": {
                "total": len(agent_manager.get_business_areas()),
                "covered": len(set(a.business_area for a in agents))
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@router.get("/capabilities")
async def get_system_capabilities(
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Get all system capabilities."""
    
    try:
        agents = agent_manager.list_agents()
        capabilities = {}
        
        for agent in agents:
            if agent.business_area.value not in capabilities:
                capabilities[agent.business_area.value] = []
            
            for capability in agent.capabilities:
                capabilities[agent.business_area.value].append({
                    "name": capability.name,
                    "description": capability.description,
                    "agent": agent.name,
                    "estimated_duration": capability.estimated_duration,
                    "input_types": capability.input_types,
                    "output_types": capability.output_types
                })
        
        return {
            "total_capabilities": sum(len(caps) for caps in capabilities.values()),
            "business_areas": len(capabilities),
            "capabilities_by_area": capabilities
        }
        
    except Exception as e:
        logger.error(f"Failed to get system capabilities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system capabilities: {str(e)}")

@router.get("/templates")
async def get_system_templates(
    agent_manager: BusinessAgentManager = Depends(get_agent_manager)
):
    """Get all available system templates."""
    
    try:
        workflow_templates = agent_manager.get_workflow_templates()
        
        return {
            "workflow_templates": workflow_templates,
            "total_templates": sum(len(templates) for templates in workflow_templates.values()),
            "business_areas_with_templates": list(workflow_templates.keys())
        }
        
    except Exception as e:
        logger.error(f"Failed to get system templates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system templates: {str(e)}")

@router.post("/restart")
async def restart_system():
    """Restart the system (placeholder)."""
    
    try:
        # This would implement actual system restart logic
        return {"message": "System restart initiated", "status": "success"}
        
    except Exception as e:
        logger.error(f"Failed to restart system: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to restart system: {str(e)}")

@router.get("/logs")
async def get_system_logs(
    level: str = "INFO",
    limit: int = 100
):
    """Get system logs (placeholder)."""
    
    try:
        # This would implement actual log retrieval
        return {
            "message": "Log retrieval not implemented",
            "level": level,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Failed to get system logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system logs: {str(e)}")


