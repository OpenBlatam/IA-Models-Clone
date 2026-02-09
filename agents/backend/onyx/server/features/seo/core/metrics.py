from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import time
import psutil
import tracemalloc
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from loguru import logger
import asyncio
from collections import defaultdict, deque
import statistics
from .interfaces import PerformanceTracker
        import orjson
from typing import Any, List, Dict, Optional
import logging
"""
Métricas y Performance Tracking ultra-optimizado para el servicio SEO.
Implementación con tracking en tiempo real y análisis de rendimiento.
"""




@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento detalladas."""
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    max_processing_time: float = 0.0
    total_memory_usage: float = 0.0
    avg_memory_usage: float = 0.0
    cache_hit_rate: float = 0.0
    compression_ratio: float = 0.0
    network_latency: float = 0.0
    elements_extracted: int = 0
    last_updated: float = field(default_factory=time.time)


class PerformanceTracker:
    """Tracker de rendimiento ultra-optimizado."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        self.timers = {}
        self.metrics = defaultdict(float)
        self.historical_data = defaultdict(lambda: deque(maxlen=1000))
        self.start_time = time.time()
        self.process = psutil.Process()
        
        # Iniciar tracemalloc si está habilitado
        if self.config.get('enable_tracemalloc', True):
            tracemalloc.start()
    
    def start_timer(self, name: str):
        """Inicia un timer."""
        self.timers[name] = time.perf_counter()
    
    def end_timer(self, name: str) -> float:
        """Termina un timer y retorna la duración."""
        if name not in self.timers:
            return 0.0
        
        duration = time.perf_counter() - self.timers[name]
        self.metrics[f"{name}_time"] = duration
        self.historical_data[f"{name}_times"].append(duration)
        
        del self.timers[name]
        return duration
    
    def record_metric(self, name: str, value: float):
        """Registra una métrica."""
        self.metrics[name] = value
        self.historical_data[name].append(value)
    
    def record_request(self, success: bool, processing_time: float, memory_usage: float):
        """Registra métricas de una request."""
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        self.metrics['total_processing_time'] += processing_time
        self.metrics['total_memory_usage'] += memory_usage
        
        # Actualizar estadísticas
        self._update_statistics()
    
    def _update_statistics(self) -> Any:
        """Actualiza estadísticas agregadas."""
        total_requests = self.metrics['total_requests']
        if total_requests > 0:
            self.metrics['avg_processing_time'] = self.metrics['total_processing_time'] / total_requests
            self.metrics['avg_memory_usage'] = self.metrics['total_memory_usage'] / total_requests
            self.metrics['success_rate'] = self.metrics['successful_requests'] / total_requests
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene todas las métricas actuales."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        cpu_percent = self.process.cpu_percent()
        
        # Obtener estadísticas de tracemalloc si está habilitado
        tracemalloc_stats = {}
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:5]
            tracemalloc_stats = {
                'top_memory_usage': [
                    {
                        'file': str(stat.traceback.format()[:100]),
                        'size_mb': stat.size / 1024 / 1024,
                        'count': stat.count
                    }
                    for stat in top_stats
                ]
            }
        
        return {
            'current': {
                'memory_usage_mb': current_memory,
                'cpu_percent': cpu_percent,
                'uptime_seconds': time.time() - self.start_time
            },
            'requests': {
                'total': self.metrics['total_requests'],
                'successful': self.metrics['successful_requests'],
                'failed': self.metrics['failed_requests'],
                'success_rate': self.metrics.get('success_rate', 0.0)
            },
            'performance': {
                'avg_processing_time': self.metrics.get('avg_processing_time', 0.0),
                'total_processing_time': self.metrics.get('total_processing_time', 0.0),
                'avg_memory_usage': self.metrics.get('avg_memory_usage', 0.0)
            },
            'timers': dict(self.timers),
            'custom_metrics': {k: v for k, v in self.metrics.items() 
                             if not k.startswith(('total_', 'avg_', 'successful_', 'failed_'))},
            'tracemalloc': tracemalloc_stats
        }
    
    def get_historical_metrics(self, metric_name: str, window: int = 100) -> List[float]:
        """Obtiene métricas históricas para análisis de tendencias."""
        if metric_name in self.historical_data:
            return list(self.historical_data[metric_name])[-window:]
        return []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen de rendimiento."""
        metrics = self.get_metrics()
        
        # Calcular percentiles para tiempos de procesamiento
        processing_times = self.get_historical_metrics('processing_times', 100)
        percentiles = {}
        if processing_times:
            percentiles = {
                'p50': statistics.median(processing_times),
                'p90': statistics.quantiles(processing_times, n=10)[8],
                'p95': statistics.quantiles(processing_times, n=20)[18],
                'p99': statistics.quantiles(processing_times, n=100)[98]
            }
        
        return {
            'summary': {
                'total_requests': metrics['requests']['total'],
                'success_rate': metrics['requests']['success_rate'],
                'avg_response_time': metrics['performance']['avg_processing_time'],
                'current_memory_mb': metrics['current']['memory_usage_mb'],
                'uptime_hours': metrics['current']['uptime_seconds'] / 3600
            },
            'percentiles': percentiles,
            'health': {
                'status': 'healthy' if metrics['requests']['success_rate'] > 0.95 else 'degraded',
                'memory_usage_percent': (metrics['current']['memory_usage_mb'] / 1024) * 100,  # Asumiendo 1GB límite
                'cpu_usage_percent': metrics['current']['cpu_percent']
            }
        }
    
    def reset_metrics(self) -> Any:
        """Resetea todas las métricas."""
        self.metrics.clear()
        self.historical_data.clear()
        self.timers.clear()
        self.start_time = time.time()
    
    def export_metrics(self, format: str = 'json') -> str:
        """Exporta métricas en diferentes formatos."""
        
        metrics = self.get_metrics()
        
        if format == 'json':
            return orjson.dumps(metrics, option=orjson.OPT_INDENT_2).decode()
        elif format == 'csv':
            # Convertir a formato CSV simple
            lines = ['metric,value']
            for category, data in metrics.items():
                if isinstance(data, dict):
                    for key, value in data.items():
                        lines.append(f"{category}.{key},{value}")
                else:
                    lines.append(f"{category},{data}")
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")


class MetricsCollector:
    """Colector de métricas del sistema."""
    
    def __init__(self) -> Any:
        self.process = psutil.Process()
        self.system_metrics = {}
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Recolecta métricas del sistema."""
        try:
            # Métricas del proceso
            process_info = {
                'memory_usage_mb': self.process.memory_info().rss / 1024 / 1024,
                'cpu_percent': self.process.cpu_percent(),
                'num_threads': self.process.num_threads(),
                'open_files': len(self.process.open_files()),
                'connections': len(self.process.connections())
            }
            
            # Métricas del sistema
            system_info = {
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent
            }
            
            return {
                'process': process_info,
                'system': system_info,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def get_memory_profile(self) -> Dict[str, Any]:
        """Obtiene perfil detallado de memoria."""
        try:
            memory_info = self.process.memory_info()
            memory_maps = self.process.memory_maps()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'shared_mb': memory_info.shared / 1024 / 1024,
                'text_mb': memory_info.text / 1024 / 1024,
                'lib_mb': memory_info.lib / 1024 / 1024,
                'data_mb': memory_info.data / 1024 / 1024,
                'dirty_mb': memory_info.dirty / 1024 / 1024,
                'memory_maps_count': len(memory_maps)
            }
        except Exception as e:
            logger.error(f"Error getting memory profile: {e}")
            return {}


class MetricsAggregator:
    """Agregador de métricas para análisis de tendencias."""
    
    def __init__(self, window_size: int = 100):
        
    """__init__ function."""
self.window_size = window_size
        self.metrics_buffer = deque(maxlen=window_size)
        self.aggregation_rules = {
            'avg': statistics.mean,
            'median': statistics.median,
            'min': min,
            'max': max,
            'std': statistics.stdev
        }
    
    def add_metrics(self, metrics: Dict[str, Any]):
        """Agrega métricas al buffer."""
        self.metrics_buffer.append({
            **metrics,
            'timestamp': time.time()
        })
    
    def get_aggregated_metrics(self, metric_path: str, aggregation: str = 'avg') -> Optional[float]:
        """Obtiene métricas agregadas."""
        if aggregation not in self.aggregation_rules:
            return None
        
        values = []
        for metric_set in self.metrics_buffer:
            value = self._get_nested_value(metric_set, metric_path)
            if value is not None and isinstance(value, (int, float)):
                values.append(value)
        
        if not values:
            return None
        
        return self.aggregation_rules[aggregation](values)
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Optional[Dict[str, Any]]:
        """Obtiene valor anidado usando path con puntos."""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def get_trend_analysis(self, metric_path: str, window: int = 10) -> Dict[str, Any]:
        """Analiza tendencias de una métrica."""
        recent_metrics = list(self.metrics_buffer)[-window:]
        if len(recent_metrics) < 2:
            return {'trend': 'insufficient_data'}
        
        values = []
        timestamps = []
        
        for metric_set in recent_metrics:
            value = self._get_nested_value(metric_set, metric_path)
            if value is not None and isinstance(value, (int, float)):
                values.append(value)
                timestamps.append(metric_set['timestamp'])
        
        if len(values) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calcular tendencia
        time_diff = timestamps[-1] - timestamps[0]
        value_diff = values[-1] - values[0]
        
        if time_diff == 0:
            trend = 'stable'
        else:
            rate_of_change = value_diff / time_diff
            if rate_of_change > 0.01:
                trend = 'increasing'
            elif rate_of_change < -0.01:
                trend = 'decreasing'
            else:
                trend = 'stable'
        
        return {
            'trend': trend,
            'rate_of_change': value_diff / time_diff if time_diff > 0 else 0,
            'current_value': values[-1],
            'previous_value': values[0],
            'change_percent': ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
        } 