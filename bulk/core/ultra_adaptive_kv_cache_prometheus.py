"""
Prometheus Metrics Exporter for Ultra-Adaptive K/V Cache Engine
Exports metrics in Prometheus format for monitoring and alerting
"""

import time
import asyncio
from typing import Dict, Any, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available. Install with: pip install prometheus-client")


class PrometheusMetrics:
    """Prometheus metrics for the engine."""
    
    def __init__(self):
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus metrics disabled - prometheus_client not available")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Request metrics
        self.requests_total = Counter(
            'kv_cache_requests_total',
            'Total number of requests',
            ['status', 'session_type']
        )
        
        self.requests_duration = Histogram(
            'kv_cache_request_duration_seconds',
            'Request duration in seconds',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.requests_tokens = Counter(
            'kv_cache_tokens_total',
            'Total tokens processed',
            ['type']  # 'input' or 'output'
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'kv_cache_cache_hits_total',
            'Total cache hits'
        )
        
        self.cache_misses = Counter(
            'kv_cache_cache_misses_total',
            'Total cache misses'
        )
        
        self.cache_size = Gauge(
            'kv_cache_cache_size',
            'Current cache size'
        )
        
        # Performance metrics
        self.response_time_p50 = Gauge(
            'kv_cache_response_time_p50_seconds',
            'P50 response time'
        )
        
        self.response_time_p95 = Gauge(
            'kv_cache_response_time_p95_seconds',
            'P95 response time'
        )
        
        self.response_time_p99 = Gauge(
            'kv_cache_response_time_p99_seconds',
            'P99 response time'
        )
        
        self.throughput = Gauge(
            'kv_cache_throughput_requests_per_second',
            'Current throughput in requests per second'
        )
        
        # Memory metrics
        self.memory_usage = Gauge(
            'kv_cache_memory_usage_ratio',
            'Memory usage ratio (0-1)'
        )
        
        self.memory_bytes = Gauge(
            'kv_cache_memory_bytes',
            'Memory usage in bytes'
        )
        
        # GPU metrics
        self.gpu_utilization = Gauge(
            'kv_cache_gpu_utilization_ratio',
            'GPU utilization ratio',
            ['gpu_id']
        )
        
        self.gpu_memory = Gauge(
            'kv_cache_gpu_memory_bytes',
            'GPU memory usage in bytes',
            ['gpu_id']
        )
        
        # Session metrics
        self.active_sessions = Gauge(
            'kv_cache_active_sessions',
            'Number of active sessions'
        )
        
        # Error metrics
        self.errors_total = Counter(
            'kv_cache_errors_total',
            'Total errors',
            ['error_type']
        )
        
        # Batch metrics
        self.batch_size = Histogram(
            'kv_cache_batch_size',
            'Batch size distribution',
            buckets=[1, 5, 10, 20, 50, 100]
        )
        
        self.batch_duration = Histogram(
            'kv_cache_batch_duration_seconds',
            'Batch processing duration'
        )
    
    def record_request(self, success: bool, duration: float, cached: bool, 
                      tokens: Optional[int] = None):
        """Record a request metric."""
        if not self.enabled:
            return
        
        status = 'success' if success else 'error'
        session_type = 'cached' if cached else 'uncached'
        
        self.requests_total.labels(status=status, session_type=session_type).inc()
        self.requests_duration.observe(duration)
        
        if cached:
            self.cache_hits.inc()
        else:
            self.cache_misses.inc()
        
        if tokens:
            self.requests_tokens.labels(type='output').inc(tokens)
    
    def update_performance_metrics(self, stats: Dict[str, Any]):
        """Update performance metrics from engine stats."""
        if not self.enabled:
            return
        
        engine_stats = stats.get('engine_stats', {})
        
        # Response times
        p50 = engine_stats.get('p50_response_time', 0)
        p95 = engine_stats.get('p95_response_time', 0)
        p99 = engine_stats.get('p99_response_time', 0)
        
        if p50 > 0:
            self.response_time_p50.set(p50)
        if p95 > 0:
            self.response_time_p95.set(p95)
        if p99 > 0:
            self.response_time_p99.set(p99)
        
        # Throughput
        throughput = engine_stats.get('throughput', 0)
        if throughput > 0:
            self.throughput.set(throughput)
        
        # Memory
        memory_usage = stats.get('memory_usage', 0)
        self.memory_usage.set(memory_usage)
        
        # Cache size
        cache_stats = stats.get('cache_stats', {})
        if 'size' in cache_stats:
            self.cache_size.set(cache_stats['size'])
        
        # Active sessions
        active_sessions = stats.get('active_sessions', 0)
        self.active_sessions.set(active_sessions)
        
        # GPU metrics
        gpu_workloads = stats.get('gpu_workloads', {})
        for gpu_id, workload in gpu_workloads.items():
            gpu_label = f'gpu_{gpu_id}'
            
            # Utilization (if available)
            if 'utilization' in workload:
                self.gpu_utilization.labels(gpu_id=gpu_label).set(workload['utilization'])
            
            # Memory
            if 'memory_used' in workload:
                memory_bytes = workload['memory_used'] * (1024 ** 3)  # Convert GB to bytes
                self.gpu_memory.labels(gpu_id=gpu_label).set(memory_bytes)
    
    def record_error(self, error_type: str):
        """Record an error."""
        if not self.enabled:
            return
        
        self.errors_total.labels(error_type=error_type).inc()
    
    def record_batch(self, batch_size: int, duration: float):
        """Record batch processing metrics."""
        if not self.enabled:
            return
        
        self.batch_size.observe(batch_size)
        self.batch_duration.observe(duration)
    
    def start_server(self, port: int = 8000):
        """Start Prometheus metrics HTTP server."""
        if not self.enabled:
            logger.warning("Cannot start Prometheus server - prometheus_client not available")
            return
        
        try:
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")


class MetricsCollector:
    """Collect metrics from engine and export to Prometheus."""
    
    def __init__(self, engine, prometheus_metrics: Optional[PrometheusMetrics] = None):
        self.engine = engine
        self.prometheus = prometheus_metrics or PrometheusMetrics()
        self.last_update = time.time()
        self.update_interval = 5.0  # Update every 5 seconds
    
    async def collect_and_export(self):
        """Collect metrics from engine and export to Prometheus."""
        while True:
            try:
                stats = self.engine.get_performance_stats()
                self.prometheus.update_performance_metrics(stats)
                
                await asyncio.sleep(self.update_interval)
            
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.update_interval)
    
    def start_collection(self):
        """Start background metrics collection."""
        asyncio.create_task(self.collect_and_export())


# Example usage
if __name__ == "__main__":
    if PROMETHEUS_AVAILABLE:
        metrics = PrometheusMetrics()
        metrics.start_server(port=9090)
        
        # Keep server running
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping Prometheus metrics server")
    else:
        print("Install prometheus_client to use Prometheus metrics:")
        print("  pip install prometheus-client")

