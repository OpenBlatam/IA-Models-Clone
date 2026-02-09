"""
Resource Optimizer
==================

Optimize resource usage to reduce costs.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ResourceRecommendation:
    """Resource optimization recommendation."""
    resource_type: str
    current_config: Dict[str, Any]
    recommended_config: Dict[str, Any]
    estimated_savings: float
    risk_level: str  # low, medium, high
    description: str


class ResourceOptimizer:
    """Resource optimizer for cost reduction."""
    
    def __init__(self):
        self._resources: Dict[str, Dict[str, Any]] = {}
        self._recommendations: List[ResourceRecommendation] = []
    
    def register_resource(
        self,
        resource_id: str,
        resource_type: str,
        config: Dict[str, Any],
        cost_per_hour: float
    ):
        """Register resource for optimization."""
        self._resources[resource_id] = {
            "type": resource_type,
            "config": config,
            "cost_per_hour": cost_per_hour
        }
        logger.debug(f"Registered resource: {resource_id}")
    
    def analyze_and_recommend(self) -> List[ResourceRecommendation]:
        """Analyze resources and generate recommendations."""
        recommendations = []
        
        for resource_id, resource in self._resources.items():
            # CPU optimization
            if resource["type"] == "compute":
                cpu = resource["config"].get("cpu", 0)
                memory = resource["config"].get("memory", 0)
                utilization = resource["config"].get("utilization", 0)
                
                if utilization < 0.3 and cpu > 2:
                    # Recommend downsizing
                    new_cpu = max(1, int(cpu * 0.5))
                    savings = (cpu - new_cpu) * 0.1  # Estimate
                    
                    recommendations.append(ResourceRecommendation(
                        resource_type="compute",
                        current_config={"cpu": cpu, "memory": memory},
                        recommended_config={"cpu": new_cpu, "memory": memory},
                        estimated_savings=savings,
                        risk_level="low",
                        description=f"CPU utilization is low ({utilization:.1%}), consider downsizing"
                    ))
            
            # Storage optimization
            elif resource["type"] == "storage":
                size_gb = resource["config"].get("size_gb", 0)
                used_gb = resource["config"].get("used_gb", 0)
                utilization = used_gb / size_gb if size_gb > 0 else 0
                
                if utilization < 0.5:
                    new_size = int(used_gb * 1.2)  # 20% headroom
                    savings = (size_gb - new_size) * 0.1  # Estimate
                    
                    recommendations.append(ResourceRecommendation(
                        resource_type="storage",
                        current_config={"size_gb": size_gb, "used_gb": used_gb},
                        recommended_config={"size_gb": new_size},
                        estimated_savings=savings,
                        risk_level="medium",
                        description=f"Storage utilization is low ({utilization:.1%}), consider resizing"
                    ))
        
        self._recommendations.extend(recommendations)
        return recommendations
    
    def get_recommendations(self, risk_level: Optional[str] = None) -> List[ResourceRecommendation]:
        """Get optimization recommendations."""
        recommendations = self._recommendations
        
        if risk_level:
            recommendations = [
                r for r in recommendations
                if r.risk_level == risk_level
            ]
        
        return recommendations
    
    def get_total_savings(self) -> float:
        """Get total estimated savings."""
        return sum(r.estimated_savings for r in self._recommendations)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "total_resources": len(self._resources),
            "recommendations": len(self._recommendations),
            "estimated_savings": self.get_total_savings(),
            "by_risk_level": {
                level: sum(1 for r in self._recommendations if r.risk_level == level)
                for level in ["low", "medium", "high"]
            }
        }















