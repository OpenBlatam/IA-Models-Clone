"""
Bulk Operations Enhanced - Mejoras Avanzadas
=============================================

Mejoras adicionales para optimización, resilencia y observabilidad
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Estrategias de optimización."""
    AGGRESSIVE = "aggressive"  # Máxima velocidad, más recursos
    BALANCED = "balanced"  # Balance entre velocidad y recursos
    CONSERVATIVE = "conservative"  # Menos recursos, más tiempo
    ADAPTIVE = "adaptive"  # Se adapta automáticamente


@dataclass
class PerformanceMetrics:
    """Métricas de performance."""
    operation_count: int = 0
    total_duration: float = 0.0
    success_count: int = 0
    error_count: int = 0
    avg_duration: float = 0.0
    p50_duration: float = 0.0
    p95_duration: float = 0.0
    p99_duration: float = 0.0
    throughput: float = 0.0  # operaciones por segundo
    error_rate: float = 0.0
    durations: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def update(self, duration: float, success: bool = True):
        """Actualizar métricas."""
        self.operation_count += 1
        self.total_duration += duration
        self.durations.append(duration)
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        self._calculate_stats()
    
    def _calculate_stats(self):
        """Calcular estadísticas."""
        if self.operation_count == 0:
            return
        
        self.avg_duration = self.total_duration / self.operation_count
        self.error_rate = self.error_count / self.operation_count
        
        if len(self.durations) > 0:
            sorted_durations = sorted(self.durations)
            self.p50_duration = sorted_durations[len(sorted_durations) // 2]
            if len(sorted_durations) >= 20:
                self.p95_duration = sorted_durations[int(len(sorted_durations) * 0.95)]
                self.p99_duration = sorted_durations[int(len(sorted_durations) * 0.99)]
        
        # Calcular throughput (últimas 100 operaciones)
        if len(self.durations) >= 2:
            time_window = sum(list(self.durations)[-100:])
            if time_window > 0:
                self.throughput = min(100, len(self.durations)) / time_window
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtener resumen de métricas."""
        return {
            "operation_count": self.operation_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "error_rate": round(self.error_rate, 4),
            "avg_duration_ms": round(self.avg_duration * 1000, 2),
            "p50_duration_ms": round(self.p50_duration * 1000, 2),
            "p95_duration_ms": round(self.p95_duration * 1000, 2),
            "p99_duration_ms": round(self.p99_duration * 1000, 2),
            "throughput_ops_per_sec": round(self.throughput, 2)
        }


class AdaptiveOptimizer:
    """Optimizador adaptivo que ajusta parámetros automáticamente."""
    
    def __init__(
        self,
        initial_workers: int = 10,
        min_workers: int = 1,
        max_workers: int = 100,
        target_p95_latency_ms: float = 500.0,
        target_throughput: float = 100.0
    ):
        self.initial_workers = initial_workers
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_p95_latency_ms = target_p95_latency_ms
        self.target_throughput = target_throughput
        
        self.current_workers = initial_workers
        self.metrics = PerformanceMetrics()
        self.optimization_history: List[Dict[str, Any]] = []
        self.last_optimization = datetime.now()
        self.optimization_interval = timedelta(seconds=30)
    
    def record_operation(self, duration: float, success: bool = True):
        """Registrar una operación."""
        self.metrics.update(duration, success)
    
    def should_optimize(self) -> bool:
        """Determinar si se debe optimizar."""
        now = datetime.now()
        if now - self.last_optimization < self.optimization_interval:
            return False
        
        # Optimizar si hay suficientes datos
        if self.metrics.operation_count < 50:
            return False
        
        return True
    
    def optimize(self) -> Dict[str, Any]:
        """Optimizar configuración."""
        if not self.should_optimize():
            return {"action": "no_action", "reason": "Not enough data or too soon"}
        
        summary = self.metrics.get_summary()
        p95_ms = summary["p95_duration_ms"]
        throughput = summary["throughput_ops_per_sec"]
        error_rate = summary["error_rate"]
        
        action = "maintain"
        new_workers = self.current_workers
        reason = ""
        
        # Si hay muchos errores, reducir workers
        if error_rate > 0.1:
            new_workers = max(
                self.min_workers,
                int(self.current_workers * 0.8)
            )
            action = "scale_down"
            reason = f"High error rate: {error_rate:.2%}"
        
        # Si la latencia es alta, aumentar workers
        elif p95_ms > self.target_p95_latency_ms * 1.2:
            new_workers = min(
                self.max_workers,
                int(self.current_workers * 1.2)
            )
            action = "scale_up"
            reason = f"High latency P95: {p95_ms:.2f}ms (target: {self.target_p95_latency_ms}ms)"
        
        # Si el throughput es bajo, aumentar workers
        elif throughput < self.target_throughput * 0.8:
            new_workers = min(
                self.max_workers,
                int(self.current_workers * 1.1)
            )
            action = "scale_up"
            reason = f"Low throughput: {throughput:.2f} ops/s (target: {self.target_throughput})"
        
        # Si la latencia es baja y el throughput es alto, reducir workers
        elif p95_ms < self.target_p95_latency_ms * 0.7 and throughput > self.target_throughput * 1.2:
            new_workers = max(
                self.min_workers,
                int(self.current_workers * 0.9)
            )
            action = "scale_down"
            reason = f"Good performance, can reduce workers (P95: {p95_ms:.2f}ms, throughput: {throughput:.2f} ops/s)"
        
        if new_workers != self.current_workers:
            self.current_workers = new_workers
            self.last_optimization = datetime.now()
            
            optimization_result = {
                "action": action,
                "previous_workers": self.current_workers if action == "scale_down" else new_workers,
                "new_workers": new_workers,
                "reason": reason,
                "metrics": summary,
                "timestamp": datetime.now().isoformat()
            }
            
            self.optimization_history.append(optimization_result)
            
            # Mantener solo últimos 100 optimizaciones
            if len(self.optimization_history) > 100:
                self.optimization_history.pop(0)
            
            return optimization_result
        
        return {"action": "maintain", "current_workers": self.current_workers, "reason": "Performance within targets"}
    
    def get_recommended_config(self) -> Dict[str, Any]:
        """Obtener configuración recomendada."""
        summary = self.metrics.get_summary()
        
        return {
            "workers": self.current_workers,
            "batch_size": self._calculate_optimal_batch_size(),
            "timeout": self._calculate_optimal_timeout(),
            "metrics": summary,
            "recommendations": self._generate_recommendations()
        }
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calcular tamaño de batch óptimo."""
        base_batch_size = 100
        
        # Ajustar basado en throughput
        if self.metrics.throughput > 200:
            return min(base_batch_size * 2, 500)
        elif self.metrics.throughput < 50:
            return max(base_batch_size // 2, 10)
        
        return base_batch_size
    
    def _calculate_optimal_timeout(self) -> float:
        """Calcular timeout óptimo."""
        if self.metrics.p99_duration > 0:
            # Timeout = 3x P99
            return min(self.metrics.p99_duration * 3, 300.0)  # Max 5 minutos
        
        return 60.0  # Default 1 minuto
    
    def _generate_recommendations(self) -> List[str]:
        """Generar recomendaciones basadas en métricas."""
        recommendations = []
        summary = self.metrics.get_summary()
        
        if summary["error_rate"] > 0.05:
            recommendations.append("Considerar reducir carga o aumentar timeouts")
        
        if summary["p95_duration_ms"] > self.target_p95_latency_ms * 2:
            recommendations.append("Latencia alta - considerar aumentar workers o optimizar operaciones")
        
        if summary["throughput_ops_per_sec"] < self.target_throughput * 0.5:
            recommendations.append("Throughput bajo - considerar aumentar paralelismo")
        
        if summary["error_rate"] < 0.01 and summary["p95_duration_ms"] < self.target_p95_latency_ms * 0.5:
            recommendations.append("Performance excelente - considerar reducir recursos para optimizar costos")
        
        return recommendations


class IntelligentCache:
    """Cache inteligente con estrategias avanzadas."""
    
    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: int = 3600,
        strategy: str = "lru"
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        
        self.cache: Dict[str, Tuple[Any, float, float, int]] = {}  # key -> (value, expiry, access_time, access_count)
        self.access_patterns: Dict[str, deque] = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del cache."""
        now = time.time()
        
        if key not in self.cache:
            self.miss_count += 1
            return None
        
        value, expiry, last_access, access_count = self.cache[key]
        
        # Verificar expiración
        if now > expiry:
            del self.cache[key]
            self.miss_count += 1
            return None
        
        # Actualizar acceso
        self.cache[key] = (value, expiry, now, access_count + 1)
        self.hit_count += 1
        
        # Registrar patrón de acceso
        if key not in self.access_patterns:
            self.access_patterns[key] = deque(maxlen=100)
        self.access_patterns[key].append(now)
        
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Establecer valor en cache."""
        now = time.time()
        ttl = ttl or self.default_ttl
        
        # Si el cache está lleno, evict según estrategia
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict()
        
        self.cache[key] = (value, now + ttl, now, 0)
    
    def _evict(self):
        """Evict entradas según estrategia."""
        if self.strategy == "lru":
            # Least Recently Used
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k][2]  # access_time
            )
        elif self.strategy == "lfu":
            # Least Frequently Used
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k][3]  # access_count
            )
        else:
            # FIFO - First key
            oldest_key = next(iter(self.cache.keys()))
        
        del self.cache[oldest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 4),
            "strategy": self.strategy
        }
    
    def clear(self):
        """Limpiar cache."""
        self.cache.clear()
        self.access_patterns.clear()
        self.hit_count = 0
        self.miss_count = 0


class BulkOperationEnhancer:
    """Enhancer para operaciones bulk con mejoras avanzadas."""
    
    def __init__(
        self,
        optimizer: Optional[AdaptiveOptimizer] = None,
        cache: Optional[IntelligentCache] = None
    ):
        self.optimizer = optimizer or AdaptiveOptimizer()
        self.cache = cache or IntelligentCache()
        self.active_operations: Dict[str, Dict[str, Any]] = {}
    
    async def execute_with_optimization(
        self,
        operation_id: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Ejecutar operación con optimización."""
        start_time = time.time()
        
        try:
            # Verificar cache primero
            cache_key = f"{operation_id}:{hash(str(args) + str(kwargs))}"
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                logger.debug(f"Cache hit for operation {operation_id}")
                return cached_result
            
            # Ejecutar operación
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            
            duration = time.time() - start_time
            
            # Registrar métricas
            self.optimizer.record_operation(duration, success=True)
            
            # Cachear resultado
            self.cache.set(cache_key, result, ttl=300)  # 5 minutos
            
            return result
        
        except Exception as e:
            duration = time.time() - start_time
            self.optimizer.record_operation(duration, success=False)
            logger.error(f"Error in operation {operation_id}: {e}")
            raise
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Obtener reporte de performance."""
        return {
            "optimizer": {
                "current_workers": self.optimizer.current_workers,
                "recommended_config": self.optimizer.get_recommended_config(),
                "optimization_history": self.optimizer.optimization_history[-10:]
            },
            "cache": self.cache.get_stats(),
            "metrics": self.optimizer.metrics.get_summary()
        }
    
    def optimize_now(self) -> Dict[str, Any]:
        """Forzar optimización inmediata."""
        return self.optimizer.optimize()


# Funciones de utilidad para mejoras

def calculate_optimal_chunk_size(
    total_items: int,
    target_chunk_time_ms: float = 100.0,
    avg_item_time_ms: float = 1.0
) -> int:
    """Calcular tamaño óptimo de chunk."""
    items_per_chunk = target_chunk_time_ms / avg_item_time_ms
    return max(1, min(int(items_per_chunk), total_items // 10, 1000))


def estimate_completion_time(
    processed: int,
    total: int,
    elapsed_time: float
) -> float:
    """Estimar tiempo de completación."""
    if processed == 0:
        return 0.0
    
    rate = processed / elapsed_time
    remaining = total - processed
    
    return remaining / rate if rate > 0 else 0.0


def adaptive_retry_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0
) -> float:
    """Calcular delay adaptivo para retry."""
    delay = base_delay * (exponential_base ** attempt)
    return min(delay, max_delay)


def batch_processor_with_progress(
    items: List[Any],
    processor: Callable,
    batch_size: Optional[int] = None,
    max_workers: Optional[int] = None,
    on_progress: Optional[Callable[[int, int], None]] = None
) -> List[Any]:
    """Procesar items en batches con callback de progreso."""
    if batch_size is None:
        batch_size = calculate_optimal_chunk_size(len(items))
    
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = processor(batch)
        results.extend(batch_results)
        
        if on_progress:
            processed = min(i + batch_size, len(items))
            on_progress(processed, len(items))
    
    return results


# Exportar componentes principales
__all__ = [
    "AdaptiveOptimizer",
    "PerformanceMetrics",
    "IntelligentCache",
    "BulkOperationEnhancer",
    "OptimizationStrategy",
    "calculate_optimal_chunk_size",
    "estimate_completion_time",
    "adaptive_retry_delay",
    "batch_processor_with_progress"
]
















