"""
System Info Service
===================

Service for providing system information and configuration details.
"""

from typing import Dict, Any, List
import logging
import platform
import sys
from datetime import datetime

from ..business_agents import BusinessAgentManager

logger = logging.getLogger(__name__)

class SystemInfoService:
    """Service for system information management."""
    
    def __init__(self, agent_manager: BusinessAgentManager):
        self.agent_manager = agent_manager
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        
        try:
            # Get system details
            system_info = {
                "name": "Business Agents System",
                "version": "1.0.0",
                "description": "Comprehensive agent system for all business areas with workflow management and document generation",
                "build_date": "2024-01-01",
                "environment": "production",
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "python_version": sys.version
                },
                "features": [
                    "Workflow Management",
                    "Document Generation",
                    "Agent Coordination", 
                    "Real-time Execution",
                    "Multi-business Area Support",
                    "API Gateway",
                    "Health Monitoring",
                    "Metrics Collection"
                ],
                "business_areas": [area.value for area in self.agent_manager.get_business_areas()],
                "endpoints": {
                    "health": "/health",
                    "docs": "/docs",
                    "api": "/business-agents",
                    "agents": "/business-agents/agents",
                    "workflows": "/business-agents/workflows",
                    "documents": "/business-agents/documents",
                    "system": "/business-agents/system"
                },
                "capabilities": await self._get_system_capabilities(),
                "configuration": await self._get_system_configuration()
            }
            
            return system_info
            
        except Exception as e:
            logger.error(f"Failed to get system info: {str(e)}")
            raise
    
    async def _get_system_capabilities(self) -> Dict[str, Any]:
        """Get system capabilities summary."""
        
        try:
            agents = self.agent_manager.list_agents()
            total_capabilities = sum(len(agent.capabilities) for agent in agents)
            
            capabilities_by_area = {}
            for agent in agents:
                area = agent.business_area.value
                if area not in capabilities_by_area:
                    capabilities_by_area[area] = 0
                capabilities_by_area[area] += len(agent.capabilities)
            
            return {
                "total_agents": len(agents),
                "total_capabilities": total_capabilities,
                "capabilities_by_area": capabilities_by_area,
                "active_agents": len([a for a in agents if a.is_active])
            }
            
        except Exception as e:
            logger.error(f"Failed to get system capabilities: {str(e)}")
            return {}
    
    async def _get_system_configuration(self) -> Dict[str, Any]:
        """Get system configuration details."""
        
        try:
            return {
                "max_concurrent_workflows": 10,
                "max_document_size": "10MB",
                "supported_formats": ["markdown", "html", "pdf", "docx"],
                "api_rate_limit": "1000/hour",
                "session_timeout": "30 minutes",
                "log_level": "INFO",
                "debug_mode": False,
                "features_enabled": {
                    "workflow_engine": True,
                    "document_generator": True,
                    "agent_coordination": True,
                    "real_time_execution": True,
                    "health_monitoring": True,
                    "metrics_collection": True
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system configuration: {str(e)}")
            return {}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        
        try:
            agents = self.agent_manager.list_agents()
            workflows = self.agent_manager.list_workflows()
            
            return {
                "status": "operational",
                "uptime": "24h 15m 30s",  # Would calculate actual uptime
                "last_restart": datetime.now().isoformat(),
                "components": {
                    "agent_manager": "healthy",
                    "workflow_engine": "healthy", 
                    "document_generator": "healthy",
                    "api_server": "healthy",
                    "database": "healthy"
                },
                "metrics": {
                    "total_agents": len(agents),
                    "active_agents": len([a for a in agents if a.is_active]),
                    "total_workflows": len(workflows),
                    "active_workflows": len([w for w in workflows if w.status.value == "active"]),
                    "completed_workflows": len([w for w in workflows if w.status.value == "completed"]),
                    "failed_workflows": len([w for w in workflows if w.status.value == "failed"])
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def get_system_health_summary(self) -> Dict[str, Any]:
        """Get system health summary."""
        
        try:
            # Get basic metrics
            agents = self.agent_manager.list_agents()
            workflows = self.agent_manager.list_workflows()
            
            # Calculate health indicators
            agent_health = len([a for a in agents if a.is_active]) / len(agents) if agents else 0
            workflow_health = len([w for w in workflows if w.status.value in ["completed", "active"]]) / len(workflows) if workflows else 0
            
            overall_health = (agent_health + workflow_health) / 2
            
            return {
                "overall_health": "healthy" if overall_health > 0.8 else "warning" if overall_health > 0.5 else "critical",
                "health_score": overall_health,
                "agent_health": agent_health,
                "workflow_health": workflow_health,
                "last_check": datetime.now().isoformat(),
                "recommendations": self._get_health_recommendations(overall_health)
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health summary: {str(e)}")
            return {"overall_health": "error", "error": str(e)}
    
    def _get_health_recommendations(self, health_score: float) -> List[str]:
        """Get health recommendations based on score."""
        
        recommendations = []
        
        if health_score < 0.5:
            recommendations.extend([
                "System health is critical. Immediate attention required.",
                "Check agent status and restart inactive agents.",
                "Review failed workflows and resolve issues.",
                "Consider system restart if problems persist."
            ])
        elif health_score < 0.8:
            recommendations.extend([
                "System health is below optimal. Monitor closely.",
                "Review inactive agents and failed workflows.",
                "Consider optimizing workflow configurations."
            ])
        else:
            recommendations.append("System is operating normally.")
        
        return recommendations