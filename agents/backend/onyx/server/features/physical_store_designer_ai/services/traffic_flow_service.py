"""
Traffic Flow Service - Análisis de tráfico y flujo de clientes
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class TrafficFlowService:
    """Servicio para análisis de tráfico y flujo"""
    
    def __init__(self):
        self.traffic_data: Dict[str, List[Dict[str, Any]]] = {}
        self.heatmaps: Dict[str, Dict[str, Any]] = {}
        self.bottlenecks: Dict[str, List[Dict[str, Any]]] = {}
    
    def record_traffic_point(
        self,
        store_id: str,
        location: str,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Registrar punto de tráfico"""
        
        traffic_point = {
            "point_id": f"tp_{store_id}_{len(self.traffic_data.get(store_id, [])) + 1}",
            "store_id": store_id,
            "location": location,
            "timestamp": timestamp or datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        if store_id not in self.traffic_data:
            self.traffic_data[store_id] = []
        
        self.traffic_data[store_id].append(traffic_point)
        
        return traffic_point
    
    def analyze_traffic_flow(
        self,
        store_id: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Analizar flujo de tráfico"""
        
        store_traffic = self.traffic_data.get(store_id, [])
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        recent_traffic = [
            t for t in store_traffic
            if start_time <= datetime.fromisoformat(t["timestamp"]) <= end_time
        ]
        
        # Agrupar por ubicación
        location_counts = defaultdict(int)
        hourly_distribution = defaultdict(int)
        
        for point in recent_traffic:
            location_counts[point["location"]] += 1
            hour = datetime.fromisoformat(point["timestamp"]).hour
            hourly_distribution[hour] += 1
        
        # Identificar bottlenecks
        bottlenecks = self._identify_bottlenecks(location_counts, recent_traffic)
        
        analysis = {
            "store_id": store_id,
            "period_hours": hours,
            "total_traffic_points": len(recent_traffic),
            "location_distribution": dict(location_counts),
            "hourly_distribution": dict(hourly_distribution),
            "peak_hour": max(hourly_distribution.items(), key=lambda x: x[1])[0] if hourly_distribution else None,
            "bottlenecks": bottlenecks,
            "flow_efficiency": self._calculate_flow_efficiency(recent_traffic),
            "analyzed_at": datetime.now().isoformat()
        }
        
        return analysis
    
    def _identify_bottlenecks(
        self,
        location_counts: Dict[str, int],
        traffic_points: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identificar cuellos de botella"""
        bottlenecks = []
        
        avg_traffic = sum(location_counts.values()) / len(location_counts) if location_counts else 0
        
        for location, count in location_counts.items():
            if count > avg_traffic * 1.5:  # 50% más que el promedio
                bottlenecks.append({
                    "location": location,
                    "traffic_count": count,
                    "severity": "high" if count > avg_traffic * 2 else "medium",
                    "recommendation": f"Considerar optimizar flujo en {location}"
                })
        
        return bottlenecks
    
    def _calculate_flow_efficiency(
        self,
        traffic_points: List[Dict[str, Any]]
    ) -> float:
        """Calcular eficiencia del flujo"""
        if not traffic_points:
            return 0.0
        
        # Simplificado - en producción usar métricas más sofisticadas
        unique_locations = len(set(t["location"] for t in traffic_points))
        total_points = len(traffic_points)
        
        # Eficiencia basada en distribución
        efficiency = min(1.0, unique_locations / (total_points / 10))
        
        return round(efficiency, 2)
    
    def generate_flow_heatmap(
        self,
        store_id: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Generar heatmap de flujo"""
        
        analysis = self.analyze_traffic_flow(store_id, hours)
        
        heatmap = {
            "store_id": store_id,
            "period_hours": hours,
            "heatmap_data": {
                location: {
                    "intensity": count,
                    "level": "high" if count > 50 else "medium" if count > 20 else "low"
                }
                for location, count in analysis["location_distribution"].items()
            },
            "generated_at": datetime.now().isoformat()
        }
        
        self.heatmaps[store_id] = heatmap
        
        return heatmap
    
    def optimize_traffic_flow(
        self,
        store_id: str
    ) -> Dict[str, Any]:
        """Optimizar flujo de tráfico"""
        
        analysis = self.analyze_traffic_flow(store_id)
        bottlenecks = analysis["bottlenecks"]
        
        recommendations = []
        
        for bottleneck in bottlenecks:
            recommendations.append({
                "location": bottleneck["location"],
                "issue": f"Alto tráfico en {bottleneck['location']}",
                "recommendation": bottleneck["recommendation"],
                "priority": bottleneck["severity"]
            })
        
        return {
            "store_id": store_id,
            "current_efficiency": analysis["flow_efficiency"],
            "recommendations": recommendations,
            "optimized_at": datetime.now().isoformat()
        }




