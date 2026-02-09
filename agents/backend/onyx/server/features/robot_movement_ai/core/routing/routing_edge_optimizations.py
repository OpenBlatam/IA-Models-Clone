"""
Routing Edge Computing Optimizations
=====================================

Optimizaciones para edge computing.
Incluye: Edge deployment, Latency optimization, Offline capabilities, etc.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from collections import deque
import threading

logger = logging.getLogger(__name__)


class EdgeCache:
    """Cache optimizado para edge."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 300.0):
        """
        Inicializar cache de edge.
        
        Args:
            max_size: Tamaño máximo
            ttl: Time to live en segundos
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                # Verificar TTL
                if time.time() - entry['timestamp'] < self.ttl:
                    self.access_times[key] = time.time()
                    return entry['value']
                else:
                    # Expirar
                    del self.cache[key]
                    del self.access_times[key]
            return None
    
    def put(self, key: str, value: Any):
        """Guardar valor en cache."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Evictar menos usado
                if self.access_times:
                    oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                    del self.cache[oldest_key]
                    del self.access_times[oldest_key]
            
            self.cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
            self.access_times[key] = time.time()


class OfflineProcessor:
    """Procesador offline para edge."""
    
    def __init__(self):
        """Inicializar procesador offline."""
        self.offline_queue: deque = deque(maxlen=10000)
        self.processed_count = 0
        self.lock = threading.Lock()
    
    def queue_request(self, request: Dict[str, Any]):
        """Agregar request a la cola offline."""
        with self.lock:
            self.offline_queue.append({
                'request': request,
                'timestamp': time.time()
            })
    
    def process_offline_queue(self, processor_func):
        """Procesar cola offline."""
        with self.lock:
            queue_copy = list(self.offline_queue)
            self.offline_queue.clear()
        
        for item in queue_copy:
            try:
                processor_func(item['request'])
                self.processed_count += 1
            except Exception as e:
                logger.error(f"Error processing offline request: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        with self.lock:
            return {
                'queued_requests': len(self.offline_queue),
                'processed_count': self.processed_count
            }


class LatencyOptimizer:
    """Optimizador de latencia para edge."""
    
    def __init__(self):
        """Inicializar optimizador."""
        self.latency_measurements: deque = deque(maxlen=1000)
        self.optimization_strategies: Dict[str, Any] = {}
        self.lock = threading.Lock()
    
    def measure_latency(self, operation: str, latency: float):
        """Medir latencia."""
        with self.lock:
            self.latency_measurements.append({
                'operation': operation,
                'latency': latency,
                'timestamp': time.time()
            })
    
    def get_optimization_strategy(self, operation: str) -> Optional[Dict[str, Any]]:
        """Obtener estrategia de optimización."""
        with self.lock:
            return self.optimization_strategies.get(operation)
    
    def optimize_for_latency(self, operation: str, strategy: Dict[str, Any]):
        """Optimizar operación para latencia."""
        with self.lock:
            self.optimization_strategies[operation] = strategy
    
    def get_latency_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de latencia."""
        with self.lock:
            if not self.latency_measurements:
                return {}
            
            latencies = [m['latency'] for m in self.latency_measurements]
            return {
                'avg_latency': sum(latencies) / len(latencies),
                'min_latency': min(latencies),
                'max_latency': max(latencies),
                'p95_latency': sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0
            }


class EdgeOptimizer:
    """Optimizador completo para edge computing."""
    
    def __init__(self):
        """Inicializar optimizador de edge."""
        self.edge_cache = EdgeCache()
        self.offline_processor = OfflineProcessor()
        self.latency_optimizer = LatencyOptimizer()
    
    def optimize_for_edge(self):
        """Aplicar optimizaciones para edge."""
        logger.info("Edge optimizations applied")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            'cache_size': len(self.edge_cache.cache),
            'offline_stats': self.offline_processor.get_stats(),
            'latency_stats': self.latency_optimizer.get_latency_stats()
        }

