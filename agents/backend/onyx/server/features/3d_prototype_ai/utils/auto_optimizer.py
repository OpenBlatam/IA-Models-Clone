"""
Auto Optimizer - Sistema de optimización automática
====================================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class AutoOptimizer:
    """Sistema de optimización automática"""
    
    def __init__(self):
        self.optimization_history: List[Dict[str, Any]] = []
        self.performance_baseline: Dict[str, float] = {}
        self.optimization_rules: Dict[str, callable] = {}
    
    def analyze_and_optimize(self, operation: str, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analiza y optimiza una operación"""
        baseline = self.performance_baseline.get(operation, {})
        
        optimizations = []
        
        # Optimización de tiempo de respuesta
        if "response_time" in current_metrics:
            current_time = current_metrics["response_time"]
            baseline_time = baseline.get("response_time", current_time)
            
            if current_time > baseline_time * 1.2:
                optimizations.append({
                    "type": "performance",
                    "issue": "Tiempo de respuesta alto",
                    "current": current_time,
                    "baseline": baseline_time,
                    "suggestion": "Considera usar caché o procesamiento asíncrono"
                })
        
        # Optimización de memoria
        if "memory_usage" in current_metrics:
            current_memory = current_metrics["memory_usage"]
            baseline_memory = baseline.get("memory_usage", current_memory)
            
            if current_memory > baseline_memory * 1.5:
                optimizations.append({
                    "type": "memory",
                    "issue": "Uso de memoria alto",
                    "current": current_memory,
                    "baseline": baseline_memory,
                    "suggestion": "Considera limpiar caché o reducir datos en memoria"
                })
        
        # Optimización de costo
        if "cost" in current_metrics:
            current_cost = current_metrics["cost"]
            baseline_cost = baseline.get("cost", current_cost)
            
            if current_cost > baseline_cost * 1.3:
                optimizations.append({
                    "type": "cost",
                    "issue": "Costo alto",
                    "current": current_cost,
                    "baseline": baseline_cost,
                    "suggestion": "Revisa materiales y busca alternativas más económicas"
                })
        
        if optimizations:
            self.optimization_history.append({
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "optimizations": optimizations
            })
        
        return {
            "operation": operation,
            "optimizations": optimizations,
            "optimization_count": len(optimizations)
        }
    
    def update_baseline(self, operation: str, metrics: Dict[str, float]):
        """Actualiza la línea base de rendimiento"""
        if operation not in self.performance_baseline:
            self.performance_baseline[operation] = {}
        
        for key, value in metrics.items():
            current_baseline = self.performance_baseline[operation].get(key, value)
            # Promedio móvil
            self.performance_baseline[operation][key] = (current_baseline * 0.7 + value * 0.3)
    
    def get_optimization_suggestions(self, operation: str) -> List[str]:
        """Obtiene sugerencias de optimización"""
        suggestions = []
        
        # Sugerencias genéricas
        suggestions.extend([
            "Usa caché para operaciones repetitivas",
            "Considera procesamiento asíncrono para operaciones largas",
            "Optimiza consultas a base de datos",
            "Usa compresión para datos grandes"
        ])
        
        # Sugerencias específicas por operación
        if "generate" in operation.lower():
            suggestions.extend([
                "Usa templates para prototipos comunes",
                "Cachea resultados de generación",
                "Procesa en lotes si generas múltiples prototipos"
            ])
        
        return suggestions
    
    def auto_optimize_cache(self, cache_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Optimiza automáticamente el caché"""
        optimizations = []
        
        hit_rate = cache_stats.get("hit_rate", 0)
        if hit_rate < 0.5:
            optimizations.append({
                "action": "increase_ttl",
                "reason": "Hit rate bajo",
                "suggestion": "Aumenta TTL del caché"
            })
        
        cache_size = cache_stats.get("size", 0)
        max_size = cache_stats.get("max_size", 1000)
        if cache_size > max_size * 0.9:
            optimizations.append({
                "action": "clear_old_entries",
                "reason": "Caché casi lleno",
                "suggestion": "Limpia entradas antiguas"
            })
        
        return {
            "optimizations": optimizations,
            "recommended_actions": [opt["action"] for opt in optimizations]
        }
    
    def get_optimization_history(self, operation: Optional[str] = None,
                                limit: int = 50) -> List[Dict[str, Any]]:
        """Obtiene historial de optimizaciones"""
        history = self.optimization_history
        
        if operation:
            history = [h for h in history if h["operation"] == operation]
        
        return history[-limit:]




