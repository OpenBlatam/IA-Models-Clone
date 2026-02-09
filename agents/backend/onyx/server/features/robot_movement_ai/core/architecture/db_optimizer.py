"""
Optimizaciones de base de datos para Robot Movement AI v2.0
Query optimization, connection pooling avanzado, y más
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import time


@dataclass
class QueryStats:
    """Estadísticas de query"""
    query: str
    execution_time: float
    timestamp: datetime
    parameters: Optional[Dict[str, Any]] = None


class DatabaseOptimizer:
    """Optimizador de base de datos"""
    
    def __init__(self):
        """Inicializar optimizador"""
        self.query_stats: List[QueryStats] = []
        self.slow_query_threshold: float = 1.0  # segundos
        self.max_stats: int = 1000
    
    def record_query(self, query: str, execution_time: float, parameters: Optional[Dict] = None):
        """Registrar estadísticas de query"""
        stats = QueryStats(
            query=query,
            execution_time=execution_time,
            timestamp=datetime.now(),
            parameters=parameters
        )
        
        self.query_stats.append(stats)
        
        # Mantener solo últimos N
        if len(self.query_stats) > self.max_stats:
            self.query_stats.pop(0)
        
        # Alertar si es lenta
        if execution_time > self.slow_query_threshold:
            from core.architecture.alerts import send_alert, AlertLevel
            import asyncio
            asyncio.create_task(send_alert(
                title="Slow Query Detected",
                message=f"Query took {execution_time:.2f}s: {query[:100]}",
                level=AlertLevel.WARNING
            ))
    
    def get_slow_queries(self, limit: int = 10) -> List[QueryStats]:
        """Obtener queries lentas"""
        slow = [q for q in self.query_stats if q.execution_time > self.slow_query_threshold]
        slow.sort(key=lambda x: x.execution_time, reverse=True)
        return slow[:limit]
    
    def get_query_analysis(self) -> Dict[str, Any]:
        """Obtener análisis de queries"""
        if not self.query_stats:
            return {}
        
        execution_times = [q.execution_time for q in self.query_stats]
        
        return {
            "total_queries": len(self.query_stats),
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "slow_queries_count": len([t for t in execution_times if t > self.slow_query_threshold]),
            "slow_query_threshold": self.slow_query_threshold
        }
    
    def optimize_query(self, query: str) -> str:
        """
        Optimizar query básica (ejemplo simplificado)
        
        Args:
            query: Query SQL
            
        Returns:
            Query optimizada
        """
        # Optimizaciones básicas
        optimized = query.strip()
        
        # Eliminar espacios múltiples
        import re
        optimized = re.sub(r'\s+', ' ', optimized)
        
        # Sugerencias básicas (en producción, usar analizador SQL real)
        if "SELECT *" in optimized.upper():
            # Sugerir columnas específicas
            pass
        
        return optimized


class ConnectionPoolMonitor:
    """Monitor de connection pool"""
    
    def __init__(self):
        """Inicializar monitor"""
        self.connection_stats: Dict[str, Any] = {}
    
    def record_connection(self, pool_name: str, action: str, duration: Optional[float] = None):
        """Registrar uso de conexión"""
        if pool_name not in self.connection_stats:
            self.connection_stats[pool_name] = {
                "total_connections": 0,
                "active_connections": 0,
                "max_connections": 0,
                "wait_times": []
            }
        
        stats = self.connection_stats[pool_name]
        
        if action == "acquire":
            stats["total_connections"] += 1
            stats["active_connections"] += 1
            if duration:
                stats["wait_times"].append(duration)
        elif action == "release":
            stats["active_connections"] = max(0, stats["active_connections"] - 1)
        
        stats["max_connections"] = max(stats["max_connections"], stats["active_connections"])
    
    def get_pool_stats(self, pool_name: str) -> Optional[Dict[str, Any]]:
        """Obtener estadísticas de pool"""
        if pool_name not in self.connection_stats:
            return None
        
        stats = self.connection_stats[pool_name]
        wait_times = stats["wait_times"]
        
        result = stats.copy()
        if wait_times:
            result["avg_wait_time"] = sum(wait_times) / len(wait_times)
            result["max_wait_time"] = max(wait_times)
        else:
            result["avg_wait_time"] = 0
            result["max_wait_time"] = 0
        
        return result


# Instancias globales
_db_optimizer: Optional[DatabaseOptimizer] = None
_pool_monitor: Optional[ConnectionPoolMonitor] = None


def get_db_optimizer() -> DatabaseOptimizer:
    """Obtener instancia global del optimizador"""
    global _db_optimizer
    if _db_optimizer is None:
        _db_optimizer = DatabaseOptimizer()
    return _db_optimizer


def get_pool_monitor() -> ConnectionPoolMonitor:
    """Obtener instancia global del monitor de pool"""
    global _pool_monitor
    if _pool_monitor is None:
        _pool_monitor = ConnectionPoolMonitor()
    return _pool_monitor




