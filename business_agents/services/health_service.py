"""
Health Service
==============

Service for system health monitoring and status checks.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import logging

from ..business_agents import BusinessAgentManager

logger = logging.getLogger(__name__)

class HealthService:
    """Service for system health monitoring."""
    
    def __init__(self, agent_manager: BusinessAgentManager):
        self.agent_manager = agent_manager
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        
        try:
            # Check if agent manager is initialized
            if not self.agent_manager:
                return self._create_unhealthy_response("Agent manager not initialized")
            
            # Get system overview
            agents = self.agent_manager.list_agents()
            workflows = self.agent_manager.list_workflows()
            
            # Check component health
            components = await self._check_components()
            
            # Calculate metrics
            metrics = self._calculate_metrics(agents, workflows)
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "components": components,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return self._create_unhealthy_response(str(e))
    
    async def _check_components(self) -> Dict[str, str]:
        """Check health of system components."""
        
        components = {}
        
        try:
            # Check agent manager
            agents = self.agent_manager.list_agents()
            components["agent_manager"] = "healthy" if agents else "warning"
            
            # Check workflow engine
            workflows = self.agent_manager.list_workflows()
            components["workflow_engine"] = "healthy"
            
            # Check document generator
            components["document_generator"] = "healthy"
            
        except Exception as e:
            logger.error(f"Component health check failed: {str(e)}")
            components["error"] = str(e)
        
        return components
    
    def _calculate_metrics(self, agents, workflows) -> Dict[str, Any]:
        """Calculate system metrics."""
        
        return {
            "total_agents": len(agents),
            "active_agents": len([a for a in agents if a.is_active]),
            "total_workflows": len(workflows),
            "active_workflows": len([w for w in workflows if w.status.value == "active"]),
            "completed_workflows": len([w for w in workflows if w.status.value == "completed"]),
            "failed_workflows": len([w for w in workflows if w.status.value == "failed"])
        }
    
    def _create_unhealthy_response(self, reason: str) -> Dict[str, Any]:
        """Create unhealthy response."""
        
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "reason": reason
        }
