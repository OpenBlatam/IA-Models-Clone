"""
Analytics - Sistema de estadísticas y analytics
===============================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class PrototypeAnalytics:
    """Sistema de analytics y estadísticas"""
    
    def __init__(self):
        self.stats = {
            "total_generated": 0,
            "by_product_type": defaultdict(int),
            "by_difficulty": defaultdict(int),
            "by_cost_range": defaultdict(int),
            "average_cost": 0,
            "average_build_time": 0,
            "most_used_materials": Counter(),
            "generation_times": [],
            "daily_stats": defaultdict(lambda: {"count": 0, "total_cost": 0})
        }
    
    def record_generation(self, prototype_data: Dict[str, Any], 
                         generation_time: float):
        """Registra la generación de un prototipo"""
        self.stats["total_generated"] += 1
        
        product_type = prototype_data.get("specifications", {}).get("tipo", "otro")
        self.stats["by_product_type"][product_type] += 1
        
        difficulty = prototype_data.get("difficulty_level", "Unknown")
        self.stats["by_difficulty"][difficulty] += 1
        
        cost = prototype_data.get("total_cost_estimate", 0)
        cost_range = self._get_cost_range(cost)
        self.stats["by_cost_range"][cost_range] += 1
        
        # Actualizar promedio de costo
        total_cost = self.stats["average_cost"] * (self.stats["total_generated"] - 1) + cost
        self.stats["average_cost"] = total_cost / self.stats["total_generated"]
        
        # Registrar materiales más usados
        materials = prototype_data.get("materials", [])
        for material in materials:
            self.stats["most_used_materials"][material.get("name", "Unknown")] += 1
        
        # Registrar tiempo de generación
        self.stats["generation_times"].append(generation_time)
        if len(self.stats["generation_times"]) > 1000:
            self.stats["generation_times"] = self.stats["generation_times"][-1000:]
        
        # Estadísticas diarias
        today = datetime.now().date().isoformat()
        self.stats["daily_stats"][today]["count"] += 1
        self.stats["daily_stats"][today]["total_cost"] += cost
    
    def _get_cost_range(self, cost: float) -> str:
        """Obtiene el rango de costo"""
        if cost < 50:
            return "Muy Bajo (<$50)"
        elif cost < 100:
            return "Bajo ($50-$100)"
        elif cost < 200:
            return "Medio ($100-$200)"
        elif cost < 500:
            return "Alto ($200-$500)"
        else:
            return "Muy Alto (>$500)"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas generales"""
        generation_times = self.stats["generation_times"]
        avg_generation_time = (
            sum(generation_times) / len(generation_times) 
            if generation_times else 0
        )
        
        return {
            "total_prototypes": self.stats["total_generated"],
            "by_product_type": dict(self.stats["by_product_type"]),
            "by_difficulty": dict(self.stats["by_difficulty"]),
            "by_cost_range": dict(self.stats["by_cost_range"]),
            "average_cost": round(self.stats["average_cost"], 2),
            "average_generation_time": round(avg_generation_time, 3),
            "most_used_materials": dict(self.stats["most_used_materials"].most_common(10)),
            "daily_stats": dict(self.stats["daily_stats"])
        }
    
    def get_trends(self, days: int = 7) -> Dict[str, Any]:
        """Obtiene tendencias de los últimos días"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        daily_counts = []
        daily_costs = []
        
        for i in range(days):
            date = start_date + timedelta(days=i)
            date_str = date.isoformat()
            stats = self.stats["daily_stats"].get(date_str, {"count": 0, "total_cost": 0})
            daily_counts.append(stats["count"])
            daily_costs.append(stats["total_cost"])
        
        return {
            "period_days": days,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily_counts": daily_counts,
            "daily_costs": daily_costs,
            "total_in_period": sum(daily_counts),
            "average_daily": sum(daily_counts) / days if days > 0 else 0,
            "trend": self._calculate_trend(daily_counts)
        }
    
    def _calculate_trend(self, values: List[int]) -> str:
        """Calcula la tendencia"""
        if len(values) < 2:
            return "insuficiente_datos"
        
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        if second_half > first_half * 1.1:
            return "creciente"
        elif second_half < first_half * 0.9:
            return "decreciente"
        else:
            return "estable"
    
    def get_product_type_insights(self, product_type: str) -> Dict[str, Any]:
        """Obtiene insights para un tipo de producto específico"""
        # Esto sería más útil con datos históricos reales
        return {
            "product_type": product_type,
            "total_generated": self.stats["by_product_type"].get(product_type, 0),
            "average_cost": self.stats["average_cost"],
            "common_materials": [
                {"name": mat, "count": count}
                for mat, count in self.stats["most_used_materials"].most_common(5)
            ]
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de rendimiento"""
        generation_times = self.stats["generation_times"]
        
        if not generation_times:
            return {
                "average_time": 0,
                "min_time": 0,
                "max_time": 0,
                "p95_time": 0
            }
        
        sorted_times = sorted(generation_times)
        p95_index = int(len(sorted_times) * 0.95)
        
        return {
            "average_time": round(sum(generation_times) / len(generation_times), 3),
            "min_time": round(min(generation_times), 3),
            "max_time": round(max(generation_times), 3),
            "p95_time": round(sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1], 3),
            "total_generations": len(generation_times)
        }




