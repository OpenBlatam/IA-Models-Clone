"""
Routing Cost Optimizations
===========================

Optimizaciones para reducción de costos.
Incluye: Resource usage tracking, Cost analysis, Auto-scaling, etc.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)


class ResourceTracker:
    """Rastreador de uso de recursos."""
    
    def __init__(self):
        """Inicializar rastreador."""
        self.resource_usage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.cost_per_unit: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def track_resource(self, resource_type: str, amount: float, cost: Optional[float] = None):
        """
        Rastrear uso de recurso.
        
        Args:
            resource_type: Tipo de recurso
            amount: Cantidad usada
            cost: Costo (opcional)
        """
        with self.lock:
            self.resource_usage[resource_type].append({
                'amount': amount,
                'cost': cost,
                'timestamp': time.time()
            })
    
    def set_cost_per_unit(self, resource_type: str, cost: float):
        """Establecer costo por unidad."""
        self.cost_per_unit[resource_type] = cost
    
    def calculate_total_cost(self, window_seconds: Optional[float] = None) -> float:
        """
        Calcular costo total.
        
        Args:
            window_seconds: Ventana de tiempo (None = todo)
        
        Returns:
            Costo total
        """
        total_cost = 0.0
        current_time = time.time()
        
        with self.lock:
            for resource_type, usage_history in self.resource_usage.items():
                for entry in usage_history:
                    if window_seconds and (current_time - entry['timestamp']) > window_seconds:
                        continue
                    
                    if entry['cost'] is not None:
                        total_cost += entry['cost']
                    elif resource_type in self.cost_per_unit:
                        total_cost += entry['amount'] * self.cost_per_unit[resource_type]
        
        return total_cost
    
    def get_resource_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de recursos."""
        with self.lock:
            stats = {}
            for resource_type, usage_history in self.resource_usage.items():
                if usage_history:
                    amounts = [e['amount'] for e in usage_history]
                    stats[resource_type] = {
                        'total': sum(amounts),
                        'average': sum(amounts) / len(amounts),
                        'max': max(amounts),
                        'min': min(amounts),
                        'count': len(usage_history)
                    }
            return stats


class AutoScaler:
    """Auto-scaler basado en métricas."""
    
    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 10,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3
    ):
        """
        Inicializar auto-scaler.
        
        Args:
            min_instances: Mínimo de instancias
            max_instances: Máximo de instancias
            scale_up_threshold: Umbral para escalar hacia arriba
            scale_down_threshold: Umbral para escalar hacia abajo
        """
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.current_instances = min_instances
        self.metrics_history: deque = deque(maxlen=100)
        self.lock = threading.Lock()
    
    def record_metric(self, metric_name: str, value: float):
        """Registrar métrica."""
        with self.lock:
            self.metrics_history.append({
                'metric': metric_name,
                'value': value,
                'timestamp': time.time()
            })
    
    def should_scale_up(self) -> bool:
        """Verificar si se debe escalar hacia arriba."""
        with self.lock:
            if self.current_instances >= self.max_instances:
                return False
            
            if not self.metrics_history:
                return False
            
            # Calcular promedio de métricas recientes
            recent_metrics = [
                m['value'] for m in self.metrics_history
                if (time.time() - m['timestamp']) < 60  # Último minuto
            ]
            
            if recent_metrics:
                avg_value = sum(recent_metrics) / len(recent_metrics)
                return avg_value > self.scale_up_threshold
            
            return False
    
    def should_scale_down(self) -> bool:
        """Verificar si se debe escalar hacia abajo."""
        with self.lock:
            if self.current_instances <= self.min_instances:
                return False
            
            if not self.metrics_history:
                return False
            
            # Calcular promedio de métricas recientes
            recent_metrics = [
                m['value'] for m in self.metrics_history
                if (time.time() - m['timestamp']) < 60  # Último minuto
            ]
            
            if recent_metrics:
                avg_value = sum(recent_metrics) / len(recent_metrics)
                return avg_value < self.scale_down_threshold
            
            return False
    
    def scale(self) -> Optional[int]:
        """
        Escalar si es necesario.
        
        Returns:
            Nuevo número de instancias o None
        """
        if self.should_scale_up():
            with self.lock:
                self.current_instances += 1
                logger.info(f"Scaled up to {self.current_instances} instances")
                return self.current_instances
        
        if self.should_scale_down():
            with self.lock:
                self.current_instances -= 1
                logger.info(f"Scaled down to {self.current_instances} instances")
                return self.current_instances
        
        return None


class CostOptimizer:
    """Optimizador completo de costos."""
    
    def __init__(self):
        """Inicializar optimizador de costos."""
        self.resource_tracker = ResourceTracker()
        self.auto_scaler = AutoScaler()
    
    def track_cost(self, resource_type: str, amount: float, cost: Optional[float] = None):
        """Rastrear costo."""
        self.resource_tracker.track_resource(resource_type, amount, cost)
    
    def get_cost_analysis(self, window_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Obtener análisis de costos."""
        total_cost = self.resource_tracker.calculate_total_cost(window_seconds)
        resource_stats = self.resource_tracker.get_resource_statistics()
        
        return {
            'total_cost': total_cost,
            'resource_statistics': resource_stats,
            'current_instances': self.auto_scaler.current_instances,
            'cost_per_instance': total_cost / max(self.auto_scaler.current_instances, 1)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            'total_cost_24h': self.resource_tracker.calculate_total_cost(86400),
            'current_instances': self.auto_scaler.current_instances,
            'resource_stats': self.resource_tracker.get_resource_statistics()
        }

