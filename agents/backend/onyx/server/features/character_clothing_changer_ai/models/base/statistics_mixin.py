"""
Statistics Mixin
================
Mixin para agregar funcionalidad de estadísticas común
"""

from typing import Dict, Any, List, Optional
from collections import defaultdict
import time


class StatisticsMixin:
    """
    Mixin para agregar estadísticas comunes a cualquier clase
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._statistics = {
            'created_at': time.time(),
            'operations': defaultdict(int),
            'errors': defaultdict(int),
            'timings': []
        }
    
    def record_operation(self, operation_name: str, success: bool = True, duration: float = 0.0):
        """Registrar operación"""
        self._statistics['operations'][operation_name] += 1
        if not success:
            self._statistics['errors'][operation_name] += 1
        if duration > 0:
            self._statistics['timings'].append({
                'operation': operation_name,
                'duration': duration,
                'timestamp': time.time()
            })
            # Mantener solo últimos 1000 timings
            if len(self._statistics['timings']) > 1000:
                self._statistics['timings'] = self._statistics['timings'][-1000:]
    
    def get_operation_count(self, operation_name: Optional[str] = None) -> int:
        """Obtener conteo de operaciones"""
        if operation_name:
            return self._statistics['operations'].get(operation_name, 0)
        return sum(self._statistics['operations'].values())
    
    def get_error_count(self, operation_name: Optional[str] = None) -> int:
        """Obtener conteo de errores"""
        if operation_name:
            return self._statistics['errors'].get(operation_name, 0)
        return sum(self._statistics['errors'].values())
    
    def get_success_rate(self, operation_name: Optional[str] = None) -> float:
        """Obtener tasa de éxito"""
        total = self.get_operation_count(operation_name)
        if total == 0:
            return 0.0
        errors = self.get_error_count(operation_name)
        return (total - errors) / total
    
    def get_average_duration(self, operation_name: Optional[str] = None) -> float:
        """Obtener duración promedio"""
        timings = self._statistics['timings']
        if operation_name:
            timings = [t for t in timings if t['operation'] == operation_name]
        
        if not timings:
            return 0.0
        
        return sum(t['duration'] for t in timings) / len(timings)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtener estadísticas completas
        
        Debe ser sobrescrito por clases que lo usen para agregar estadísticas específicas
        """
        base_stats = {
            'created_at': self._statistics['created_at'],
            'uptime': time.time() - self._statistics['created_at'],
            'total_operations': self.get_operation_count(),
            'total_errors': self.get_error_count(),
            'success_rate': self.get_success_rate(),
            'operations': dict(self._statistics['operations']),
            'errors': dict(self._statistics['errors']),
            'average_duration': self.get_average_duration()
        }
        
        return base_stats
    
    def reset_statistics(self):
        """Resetear estadísticas"""
        self._statistics = {
            'created_at': time.time(),
            'operations': defaultdict(int),
            'errors': defaultdict(int),
            'timings': []
        }

