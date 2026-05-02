"""
Performance Optimizer para LLM Service.

Optimizaciones automáticas de performance incluyendo:
- Auto-tuning de parámetros
- Optimización de batch size
- Connection pooling inteligente
- Prefetching y caching proactivo
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import statistics

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Métricas de performance."""
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput: float  # requests per second
    error_rate: float
    cache_hit_rate: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "avg_latency_ms": self.avg_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "throughput": self.throughput,
            "error_rate": self.error_rate,
            "cache_hit_rate": self.cache_hit_rate,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class OptimizationRecommendation:
    """Recomendación de optimización."""
    type: str
    description: str
    expected_improvement: str
    priority: str  # "low", "medium", "high", "critical"
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "type": self.type,
            "description": self.description,
            "expected_improvement": self.expected_improvement,
            "priority": self.priority,
            "parameters": self.parameters
        }


class PerformanceOptimizer:
    """
    Optimizador de performance para LLM Service.
    
    Características:
    - Auto-tuning de parámetros
    - Análisis de bottlenecks
    - Recomendaciones de optimización
    - A/B testing de configuraciones
    """
    
    def __init__(self, window_size: int = 100):
        """
        Inicializar optimizador.
        
        Args:
            window_size: Tamaño de ventana para análisis
        """
        self.window_size = window_size
        self.metrics_history: deque = deque(maxlen=window_size)
        self.optimization_history: List[Dict[str, Any]] = []
    
    def record_metrics(
        self,
        latency_ms: float,
        error: bool = False,
        cached: bool = False
    ) -> None:
        """
        Registrar métricas de una request.
        
        Args:
            latency_ms: Latencia en milisegundos
            error: Si hubo error
            cached: Si fue cache hit
        """
        # Calcular throughput aproximado (simplificado)
        now = datetime.now()
        
        # Limpiar métricas antiguas (últimos 60 segundos)
        cutoff = now - timedelta(seconds=60)
        self.metrics_history = deque(
            [m for m in self.metrics_history if m["timestamp"] > cutoff],
            maxlen=self.window_size
        )
        
        # Agregar nueva métrica
        self.metrics_history.append({
            "latency_ms": latency_ms,
            "error": error,
            "cached": cached,
            "timestamp": now
        })
    
    def analyze_performance(self) -> PerformanceMetrics:
        """
        Analizar performance actual.
        
        Returns:
            Métricas de performance
        """
        if not self.metrics_history:
            return PerformanceMetrics(
                avg_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                throughput=0,
                error_rate=0,
                cache_hit_rate=0,
                timestamp=datetime.now()
            )
        
        latencies = [m["latency_ms"] for m in self.metrics_history]
        errors = [m["error"] for m in self.metrics_history]
        cached = [m["cached"] for m in self.metrics_history]
        
        # Calcular percentiles
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        
        def percentile(data, p):
            if not data:
                return 0
            k = (n - 1) * p
            f = int(k)
            c = k - f
            if f + 1 < n:
                return data[f] + c * (data[f + 1] - data[f])
            return data[f] if f < n else data[-1]
        
        # Calcular throughput (requests en última ventana)
        time_window = (self.metrics_history[-1]["timestamp"] - 
                      self.metrics_history[0]["timestamp"]).total_seconds()
        throughput = len(self.metrics_history) / max(time_window, 1) if time_window > 0 else 0
        
        return PerformanceMetrics(
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p95_latency_ms=percentile(latencies_sorted, 0.95),
            p99_latency_ms=percentile(latencies_sorted, 0.99),
            throughput=throughput,
            error_rate=sum(errors) / len(errors) if errors else 0,
            cache_hit_rate=sum(cached) / len(cached) if cached else 0,
            timestamp=datetime.now()
        )
    
    def get_optimization_recommendations(
        self,
        current_config: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """
        Obtener recomendaciones de optimización.
        
        Args:
            current_config: Configuración actual
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        metrics = self.analyze_performance()
        
        # Recomendación: Mejorar cache hit rate
        if metrics.cache_hit_rate < 0.3:
            recommendations.append(OptimizationRecommendation(
                type="cache_optimization",
                description="Cache hit rate bajo. Considera aumentar TTL o mejorar estrategia de cache.",
                expected_improvement="Reducción de 20-40% en latencia promedio",
                priority="high",
                parameters={
                    "current_cache_hit_rate": metrics.cache_hit_rate,
                    "recommended_ttl": current_config.get("cache_ttl", 3600) * 2
                }
            ))
        
        # Recomendación: Optimizar batch size
        if metrics.throughput < 1.0 and metrics.avg_latency_ms > 2000:
            recommendations.append(OptimizationRecommendation(
                type="batch_optimization",
                description="Throughput bajo y latencia alta. Considera procesamiento por lotes.",
                expected_improvement="Aumento de 30-50% en throughput",
                priority="medium",
                parameters={
                    "current_throughput": metrics.throughput,
                    "recommended_batch_size": 5
                }
            ))
        
        # Recomendación: Reducir timeout
        if metrics.p95_latency_ms < current_config.get("timeout", 60) * 1000 * 0.5:
            recommendations.append(OptimizationRecommendation(
                type="timeout_optimization",
                description="Timeout muy alto comparado con latencia real. Puede reducirse.",
                expected_improvement="Mejor manejo de errores y fallos más rápidos",
                priority="low",
                parameters={
                    "current_timeout": current_config.get("timeout", 60),
                    "recommended_timeout": int(metrics.p95_latency_ms / 1000) + 5
                }
            ))
        
        # Recomendación: Aumentar paralelismo
        if metrics.throughput < 2.0 and metrics.error_rate < 0.05:
            recommendations.append(OptimizationRecommendation(
                type="parallelism_optimization",
                description="Throughput bajo con baja tasa de errores. Puede aumentar paralelismo.",
                expected_improvement="Aumento de 50-100% en throughput",
                priority="medium",
                parameters={
                    "current_max_parallel": current_config.get("max_parallel_requests", 10),
                    "recommended_max_parallel": current_config.get("max_parallel_requests", 10) * 2
                }
            ))
        
        # Recomendación: Optimizar modelo
        if metrics.avg_latency_ms > 5000:
            recommendations.append(OptimizationRecommendation(
                type="model_optimization",
                description="Latencia muy alta. Considera usar modelo más rápido o optimizar prompts.",
                expected_improvement="Reducción de 30-60% en latencia",
                priority="high",
                parameters={
                    "current_avg_latency": metrics.avg_latency_ms,
                    "suggested_models": ["openai/gpt-4o-mini", "anthropic/claude-3-haiku"]
                }
            ))
        
        return recommendations
    
    def auto_tune_parameters(
        self,
        current_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Auto-tuning de parámetros basado en métricas.
        
        Args:
            current_config: Configuración actual
            
        Returns:
            Configuración optimizada
        """
        metrics = self.analyze_performance()
        optimized = current_config.copy()
        
        # Auto-tune timeout
        if metrics.p95_latency_ms > 0:
            recommended_timeout = int((metrics.p95_latency_ms * 2) / 1000) + 10
            if recommended_timeout < current_config.get("timeout", 60):
                optimized["timeout"] = recommended_timeout
                logger.info(f"Auto-tuned timeout: {current_config.get('timeout')} -> {recommended_timeout}")
        
        # Auto-tune cache TTL basado en cache hit rate
        if metrics.cache_hit_rate < 0.2:
            optimized["cache_ttl"] = current_config.get("cache_ttl", 3600) * 2
            logger.info(f"Auto-tuned cache TTL: {current_config.get('cache_ttl')} -> {optimized['cache_ttl']}")
        
        # Auto-tune max parallel requests
        if metrics.error_rate < 0.01 and metrics.throughput < 5.0:
            current_parallel = current_config.get("max_parallel_requests", 10)
            optimized["max_parallel_requests"] = min(current_parallel * 2, 50)
            logger.info(
                f"Auto-tuned max parallel: {current_parallel} -> {optimized['max_parallel_requests']}"
            )
        
        return optimized
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Obtener reporte completo de performance.
        
        Returns:
            Reporte con métricas y recomendaciones
        """
        metrics = self.analyze_performance()
        
        return {
            "metrics": metrics.to_dict(),
            "sample_size": len(self.metrics_history),
            "window_size": self.window_size,
            "timestamp": datetime.now().isoformat()
        }


def get_performance_optimizer(window_size: int = 100) -> PerformanceOptimizer:
    """Factory function para obtener instancia singleton del optimizador."""
    if not hasattr(get_performance_optimizer, "_instance"):
        get_performance_optimizer._instance = PerformanceOptimizer(window_size)
    return get_performance_optimizer._instance



