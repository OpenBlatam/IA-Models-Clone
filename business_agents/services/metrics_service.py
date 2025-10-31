"""
Metrics Service
===============

Service for collecting and providing system metrics and analytics.
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

from ..business_agents import BusinessAgentManager

logger = logging.getLogger(__name__)

class MetricsService:
    """Service for metrics collection and analysis."""
    
    def __init__(self, agent_manager: BusinessAgentManager):
        self.agent_manager = agent_manager
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        
        try:
            return {
                "agents": await self._get_agent_metrics(),
                "workflows": await self._get_workflow_metrics(),
                "documents": await self._get_document_metrics(),
                "system": await self._get_system_metrics(),
                "performance": await self._get_performance_metrics(),
                "usage": await self._get_usage_metrics()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {str(e)}")
            raise
    
    async def _get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent-related metrics."""
        
        try:
            agents = self.agent_manager.list_agents()
            
            # Calculate agent metrics
            total_agents = len(agents)
            active_agents = len([a for a in agents if a.is_active])
            inactive_agents = total_agents - active_agents
            
            # Capabilities metrics
            total_capabilities = sum(len(agent.capabilities) for agent in agents)
            avg_capabilities_per_agent = total_capabilities / total_agents if total_agents > 0 else 0
            
            # Business area distribution
            area_distribution = {}
            for agent in agents:
                area = agent.business_area.value
                area_distribution[area] = area_distribution.get(area, 0) + 1
            
            return {
                "total_agents": total_agents,
                "active_agents": active_agents,
                "inactive_agents": inactive_agents,
                "activation_rate": active_agents / total_agents if total_agents > 0 else 0,
                "total_capabilities": total_capabilities,
                "avg_capabilities_per_agent": avg_capabilities_per_agent,
                "area_distribution": area_distribution,
                "most_active_area": max(area_distribution.items(), key=lambda x: x[1])[0] if area_distribution else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get agent metrics: {str(e)}")
            return {}
    
    async def _get_workflow_metrics(self) -> Dict[str, Any]:
        """Get workflow-related metrics."""
        
        try:
            workflows = self.agent_manager.list_workflows()
            
            # Status distribution
            status_distribution = {}
            for workflow in workflows:
                status = workflow.status.value
                status_distribution[status] = status_distribution.get(status, 0) + 1
            
            # Business area distribution
            area_distribution = {}
            for workflow in workflows:
                area = workflow.business_area
                area_distribution[area] = area_distribution.get(area, 0) + 1
            
            # Calculate success rate
            completed = status_distribution.get("completed", 0)
            failed = status_distribution.get("failed", 0)
            total_executed = completed + failed
            success_rate = completed / total_executed if total_executed > 0 else 0
            
            return {
                "total_workflows": len(workflows),
                "status_distribution": status_distribution,
                "area_distribution": area_distribution,
                "success_rate": success_rate,
                "completion_rate": completed / len(workflows) if workflows else 0,
                "failure_rate": failed / len(workflows) if workflows else 0,
                "most_used_area": max(area_distribution.items(), key=lambda x: x[1])[0] if area_distribution else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get workflow metrics: {str(e)}")
            return {}
    
    async def _get_document_metrics(self) -> Dict[str, Any]:
        """Get document-related metrics."""
        
        try:
            # This would integrate with document storage
            # For now, return mock metrics
            return {
                "total_documents": 0,
                "documents_by_type": {},
                "documents_by_format": {},
                "total_size_bytes": 0,
                "avg_size_bytes": 0,
                "generation_success_rate": 1.0,
                "most_used_format": "markdown",
                "most_used_type": "strategy"
            }
            
        except Exception as e:
            logger.error(f"Failed to get document metrics: {str(e)}")
            return {}
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        
        try:
            return {
                "uptime_seconds": 86400,  # Would calculate actual uptime
                "memory_usage_mb": 512,   # Would get actual memory usage
                "cpu_usage_percent": 25,  # Would get actual CPU usage
                "disk_usage_percent": 45, # Would get actual disk usage
                "active_connections": 10, # Would get actual connections
                "error_count": 0,         # Would count actual errors
                "warning_count": 2,       # Would count actual warnings
                "last_restart": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {str(e)}")
            return {}
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance-related metrics."""
        
        try:
            return {
                "avg_response_time_ms": 150,
                "max_response_time_ms": 2000,
                "min_response_time_ms": 50,
                "requests_per_second": 10,
                "throughput_mb_per_second": 5.2,
                "error_rate_percent": 0.1,
                "availability_percent": 99.9,
                "last_24h_requests": 8640,
                "peak_concurrent_users": 25
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {str(e)}")
            return {}
    
    async def _get_usage_metrics(self) -> Dict[str, Any]:
        """Get usage-related metrics."""
        
        try:
            agents = self.agent_manager.list_agents()
            workflows = self.agent_manager.list_workflows()
            
            # Calculate usage patterns
            business_areas = self.agent_manager.get_business_areas()
            area_usage = {}
            
            for area in business_areas:
                area_agents = [a for a in agents if a.business_area == area]
                area_workflows = [w for w in workflows if w.business_area == area.value]
                area_usage[area.value] = {
                    "agents": len(area_agents),
                    "workflows": len(area_workflows),
                    "active_agents": len([a for a in area_agents if a.is_active])
                }
            
            return {
                "total_business_areas": len(business_areas),
                "areas_in_use": len([area for area in area_usage.values() if area["workflows"] > 0]),
                "area_usage": area_usage,
                "most_used_area": max(area_usage.items(), key=lambda x: x[1]["workflows"])[0] if area_usage else None,
                "least_used_area": min(area_usage.items(), key=lambda x: x[1]["workflows"])[0] if area_usage else None,
                "usage_trend": "increasing",  # Would calculate actual trend
                "peak_usage_hour": 14,        # Would calculate actual peak hour
                "avg_sessions_per_day": 50    # Would calculate actual sessions
            }
            
        except Exception as e:
            logger.error(f"Failed to get usage metrics: {str(e)}")
            return {}
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of key metrics."""
        
        try:
            metrics = await self.get_system_metrics()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_agents": metrics["agents"].get("total_agents", 0),
                    "active_agents": metrics["agents"].get("active_agents", 0),
                    "total_workflows": metrics["workflows"].get("total_workflows", 0),
                    "success_rate": metrics["workflows"].get("success_rate", 0),
                    "system_health": "healthy" if metrics["system"].get("error_count", 0) == 0 else "warning",
                    "performance": "good" if metrics["performance"].get("avg_response_time_ms", 0) < 200 else "needs_attention"
                },
                "alerts": await self._get_metric_alerts(metrics),
                "recommendations": await self._get_metric_recommendations(metrics)
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {str(e)}")
            return {"error": str(e)}
    
    async def _get_metric_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get alerts based on metrics."""
        
        alerts = []
        
        # Check for critical issues
        if metrics["system"].get("error_count", 0) > 10:
            alerts.append({
                "level": "critical",
                "message": "High error count detected",
                "metric": "error_count",
                "value": metrics["system"]["error_count"]
            })
        
        if metrics["workflows"].get("success_rate", 1) < 0.8:
            alerts.append({
                "level": "warning",
                "message": "Low workflow success rate",
                "metric": "success_rate",
                "value": metrics["workflows"]["success_rate"]
            })
        
        if metrics["performance"].get("avg_response_time_ms", 0) > 1000:
            alerts.append({
                "level": "warning",
                "message": "High response time detected",
                "metric": "avg_response_time_ms",
                "value": metrics["performance"]["avg_response_time_ms"]
            })
        
        return alerts
    
    async def _get_metric_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Get recommendations based on metrics."""
        
        recommendations = []
        
        # Agent recommendations
        if metrics["agents"].get("activation_rate", 1) < 0.8:
            recommendations.append("Consider activating more agents to improve coverage")
        
        # Workflow recommendations
        if metrics["workflows"].get("success_rate", 1) < 0.9:
            recommendations.append("Review failed workflows and optimize configurations")
        
        # Performance recommendations
        if metrics["performance"].get("avg_response_time_ms", 0) > 500:
            recommendations.append("Consider optimizing system performance")
        
        # Usage recommendations
        if metrics["usage"].get("areas_in_use", 0) < len(metrics["usage"].get("area_usage", {})):
            recommendations.append("Explore underutilized business areas")
        
        return recommendations