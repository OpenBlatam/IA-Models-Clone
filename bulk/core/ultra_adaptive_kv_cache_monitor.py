#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard for Ultra-Adaptive K/V Cache Engine
Provides live monitoring, alerts, and performance analysis
"""

import asyncio
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
from pathlib import Path
import logging

try:
    from ultra_adaptive_kv_cache_engine import UltraAdaptiveKVCacheEngine
except ImportError:
    print("Warning: Ultra-Adaptive K/V Cache Engine not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Alert definition."""
    level: str  # 'critical', 'warning', 'info'
    message: str
    metric: str
    threshold: float
    current_value: float
    timestamp: float


class PerformanceMonitor:
    """Real-time performance monitor for the engine."""
    
    def __init__(self, engine: UltraAdaptiveKVCacheEngine, check_interval: float = 5.0):
        self.engine = engine
        self.check_interval = check_interval
        self.alerts: List[Alert] = []
        self.metrics_history: List[Dict[str, Any]] = []
        self.running = False
        
        # Alert thresholds
        self.thresholds = {
            'error_rate': {'critical': 0.1, 'warning': 0.05},
            'memory_usage': {'critical': 0.95, 'warning': 0.85},
            'avg_response_time': {'critical': 5.0, 'warning': 2.0},  # seconds
            'p95_response_time': {'critical': 10.0, 'warning': 5.0},
            'cache_hit_rate': {'warning': 0.5},  # Lower is worse
            'throughput': {'warning': 1.0}  # Lower is worse (req/s)
        }
    
    async def start_monitoring(self):
        """Start continuous monitoring."""
        self.running = True
        logger.info("Starting performance monitoring...")
        
        while self.running:
            try:
                await self._collect_metrics()
                await self._check_alerts()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        logger.info("Stopped performance monitoring")
    
    async def _collect_metrics(self):
        """Collect current metrics."""
        stats = self.engine.get_performance_stats()
        
        metric_snapshot = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'engine_stats': stats.get('engine_stats', {}),
            'memory_usage': stats.get('memory_usage', 0),
            'active_sessions': stats.get('active_sessions', 0),
            'available_gpus': stats.get('available_gpus', 0),
            'gpu_workloads': stats.get('gpu_workloads', {})
        }
        
        self.metrics_history.append(metric_snapshot)
        
        # Keep only last 1000 snapshots
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    async def _check_alerts(self):
        """Check metrics against thresholds and generate alerts."""
        if not self.metrics_history:
            return
        
        latest = self.metrics_history[-1]
        engine_stats = latest.get('engine_stats', {})
        
        # Check error rate
        error_rate = engine_stats.get('error_rate', 0)
        if error_rate >= self.thresholds['error_rate']['critical']:
            self._add_alert('critical', 'error_rate', error_rate, 
                          self.thresholds['error_rate']['critical'],
                          f"Critical: Error rate is {error_rate*100:.2f}%")
        elif error_rate >= self.thresholds['error_rate']['warning']:
            self._add_alert('warning', 'error_rate', error_rate,
                          self.thresholds['error_rate']['warning'],
                          f"Warning: Error rate is {error_rate*100:.2f}%")
        
        # Check memory usage
        memory_usage = latest.get('memory_usage', 0)
        if memory_usage >= self.thresholds['memory_usage']['critical']:
            self._add_alert('critical', 'memory_usage', memory_usage,
                          self.thresholds['memory_usage']['critical'],
                          f"Critical: Memory usage is {memory_usage*100:.2f}%")
        elif memory_usage >= self.thresholds['memory_usage']['warning']:
            self._add_alert('warning', 'memory_usage', memory_usage,
                          self.thresholds['memory_usage']['warning'],
                          f"Warning: Memory usage is {memory_usage*100:.2f}%")
        
        # Check response times
        avg_response_time = engine_stats.get('avg_response_time', 0)
        if avg_response_time >= self.thresholds['avg_response_time']['critical']:
            self._add_alert('critical', 'avg_response_time', avg_response_time,
                          self.thresholds['avg_response_time']['critical'],
                          f"Critical: Average response time is {avg_response_time:.2f}s")
        
        p95_response_time = engine_stats.get('p95_response_time', 0)
        if p95_response_time >= self.thresholds['p95_response_time']['critical']:
            self._add_alert('critical', 'p95_response_time', p95_response_time,
                          self.thresholds['p95_response_time']['critical'],
                          f"Critical: P95 response time is {p95_response_time:.2f}s")
        
        # Check cache hit rate (lower is worse)
        cache_hit_rate = engine_stats.get('cache_hit_rate', 1.0)
        if cache_hit_rate < self.thresholds['cache_hit_rate'].get('warning', 0.5):
            self._add_alert('warning', 'cache_hit_rate', cache_hit_rate,
                          self.thresholds['cache_hit_rate']['warning'],
                          f"Warning: Cache hit rate is {cache_hit_rate*100:.2f}%")
        
        # Check throughput
        throughput = engine_stats.get('throughput', 0)
        if throughput < self.thresholds['throughput'].get('warning', 1.0):
            self._add_alert('warning', 'throughput', throughput,
                          self.thresholds['throughput']['warning'],
                          f"Warning: Throughput is {throughput:.2f} req/s")
    
    def _add_alert(self, level: str, metric: str, current_value: float, 
                   threshold: float, message: str):
        """Add an alert."""
        alert = Alert(
            level=level,
            message=message,
            metric=metric,
            threshold=threshold,
            current_value=current_value,
            timestamp=time.time()
        )
        
        # Avoid duplicate alerts (same metric, same level within last minute)
        recent_alerts = [a for a in self.alerts 
                        if a.metric == metric and a.level == level 
                        and time.time() - a.timestamp < 60]
        
        if not recent_alerts:
            self.alerts.append(alert)
            logger.warning(f"[{level.upper()}] {message}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        if not self.metrics_history:
            return {'status': 'no_data'}
        
        latest = self.metrics_history[-1]
        engine_stats = latest.get('engine_stats', {})
        
        # Calculate trends
        trends = {}
        if len(self.metrics_history) >= 2:
            prev = self.metrics_history[-2].get('engine_stats', {})
            
            for key in ['avg_response_time', 'throughput', 'error_rate']:
                current = engine_stats.get(key, 0)
                previous = prev.get(key, 0)
                
                if previous > 0:
                    change = ((current - previous) / previous) * 100
                    trends[key] = {
                        'current': current,
                        'previous': previous,
                        'change_percent': change,
                        'direction': 'up' if change > 0 else 'down'
                    }
        
        return {
            'timestamp': latest['timestamp'],
            'datetime': latest['datetime'],
            'metrics': {
                'total_requests': engine_stats.get('total_requests', 0),
                'avg_response_time': engine_stats.get('avg_response_time', 0),
                'p95_response_time': engine_stats.get('p95_response_time', 0),
                'p99_response_time': engine_stats.get('p99_response_time', 0),
                'throughput': engine_stats.get('throughput', 0),
                'error_rate': engine_stats.get('error_rate', 0),
                'cache_hit_rate': engine_stats.get('cache_hit_rate', 0),
                'memory_usage': latest.get('memory_usage', 0),
                'active_sessions': latest.get('active_sessions', 0)
            },
            'trends': trends,
            'alerts': {
                'total': len(self.alerts),
                'critical': len([a for a in self.alerts if a.level == 'critical']),
                'warning': len([a for a in self.alerts if a.level == 'warning']),
                'recent': [asdict(a) for a in self.alerts[-10:]]
            },
            'gpu_info': latest.get('gpu_workloads', {})
        }
    
    def get_metrics_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m['timestamp'] >= cutoff_time]
        
        if not recent_metrics:
            return {'error': 'No data available'}
        
        engine_stats_list = [m['engine_stats'] for m in recent_metrics]
        
        summary = {
            'period_minutes': minutes,
            'samples': len(recent_metrics),
            'metrics': {
                'avg_response_time': {
                    'avg': sum(s.get('avg_response_time', 0) for s in engine_stats_list) / len(engine_stats_list),
                    'min': min((s.get('avg_response_time', 0) for s in engine_stats_list), default=0),
                    'max': max((s.get('avg_response_time', 0) for s in engine_stats_list), default=0)
                },
                'throughput': {
                    'avg': sum(s.get('throughput', 0) for s in engine_stats_list) / len(engine_stats_list),
                    'min': min((s.get('throughput', 0) for s in engine_stats_list), default=0),
                    'max': max((s.get('throughput', 0) for s in engine_stats_list), default=0)
                },
                'error_rate': {
                    'avg': sum(s.get('error_rate', 0) for s in engine_stats_list) / len(engine_stats_list),
                    'max': max((s.get('error_rate', 0) for s in engine_stats_list), default=0)
                },
                'cache_hit_rate': {
                    'avg': sum(s.get('cache_hit_rate', 0) for s in engine_stats_list) / len(engine_stats_list),
                    'min': min((s.get('cache_hit_rate', 0) for s in engine_stats_list), default=0)
                }
            },
            'memory_usage': {
                'avg': sum(m.get('memory_usage', 0) for m in recent_metrics) / len(recent_metrics),
                'max': max((m.get('memory_usage', 0) for m in recent_metrics), default=0)
            },
            'active_sessions': {
                'avg': sum(m.get('active_sessions', 0) for m in recent_metrics) / len(recent_metrics),
                'max': max((m.get('active_sessions', 0) for m in recent_metrics), default=0)
            }
        }
        
        return summary
    
    def export_metrics(self, output_file: str, minutes: Optional[int] = None):
        """Export metrics to JSON file."""
        if minutes:
            cutoff_time = time.time() - (minutes * 60)
            metrics_to_export = [m for m in self.metrics_history if m['timestamp'] >= cutoff_time]
        else:
            metrics_to_export = self.metrics_history
        
        export_data = {
            'export_timestamp': time.time(),
            'export_datetime': datetime.now().isoformat(),
            'total_samples': len(metrics_to_export),
            'metrics': metrics_to_export,
            'alerts': [asdict(a) for a in self.alerts]
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(metrics_to_export)} metric samples to {output_file}")


async def print_dashboard(monitor: PerformanceMonitor, refresh_interval: float = 2.0):
    """Print real-time dashboard."""
    import os
    
    while monitor.running:
        os.system('clear' if os.name != 'nt' else 'cls')  # Clear screen
        
        status = monitor.get_current_status()
        
        print("=" * 80)
        print("ULTRA-ADAPTIVE K/V CACHE ENGINE - REAL-TIME MONITORING")
        print("=" * 80)
        print(f"Time: {status.get('datetime', 'N/A')}")
        print()
        
        # Metrics
        metrics = status.get('metrics', {})
        print("PERFORMANCE METRICS:")
        print(f"  Total Requests: {metrics.get('total_requests', 0):,}")
        print(f"  Avg Response Time: {metrics.get('avg_response_time', 0)*1000:.2f} ms")
        print(f"  P95 Response Time: {metrics.get('p95_response_time', 0)*1000:.2f} ms")
        print(f"  P99 Response Time: {metrics.get('p99_response_time', 0)*1000:.2f} ms")
        print(f"  Throughput: {metrics.get('throughput', 0):.2f} req/s")
        print(f"  Error Rate: {metrics.get('error_rate', 0)*100:.2f}%")
        print(f"  Cache Hit Rate: {metrics.get('cache_hit_rate', 0)*100:.2f}%")
        print(f"  Memory Usage: {metrics.get('memory_usage', 0)*100:.2f}%")
        print(f"  Active Sessions: {metrics.get('active_sessions', 0)}")
        print()
        
        # Alerts
        alerts_info = status.get('alerts', {})
        print("ALERTS:")
        print(f"  Total: {alerts_info.get('total', 0)}")
        print(f"  Critical: {alerts_info.get('critical', 0)}")
        print(f"  Warning: {alerts_info.get('warning', 0)}")
        print()
        
        # Recent alerts
        recent_alerts = alerts_info.get('recent', [])
        if recent_alerts:
            print("RECENT ALERTS:")
            for alert in recent_alerts[-5:]:
                level = alert.get('level', '').upper()
                message = alert.get('message', '')
                print(f"  [{level}] {message}")
            print()
        
        # GPU Info
        gpu_info = status.get('gpu_info', {})
        if gpu_info:
            print("GPU STATUS:")
            for gpu_id, workload in gpu_info.items():
                print(f"  {gpu_id}:")
                print(f"    Active Tasks: {workload.get('active_tasks', 0)}")
                print(f"    Memory Used: {workload.get('memory_used', 0):.2f} GB")
            print()
        
        # Trends
        trends = status.get('trends', {})
        if trends:
            print("TRENDS (Last Check):")
            for metric, trend in trends.items():
                direction = "↑" if trend.get('direction') == 'up' else "↓"
                change = trend.get('change_percent', 0)
                print(f"  {metric}: {change:+.2f}% {direction}")
            print()
        
        print("=" * 80)
        print(f"Refreshing every {refresh_interval}s... Press Ctrl+C to stop")
        
        await asyncio.sleep(refresh_interval)


async def main():
    """Main entry point for monitoring."""
    parser = argparse.ArgumentParser(description="Monitor Ultra-Adaptive K/V Cache Engine")
    parser.add_argument("--interval", type=float, default=5.0, help="Check interval in seconds")
    parser.add_argument("--dashboard", action="store_true", help="Show live dashboard")
    parser.add_argument("--refresh", type=float, default=2.0, help="Dashboard refresh rate")
    parser.add_argument("--summary", type=int, help="Show summary for last N minutes")
    parser.add_argument("--export", type=str, help="Export metrics to JSON file")
    parser.add_argument("--export-minutes", type=int, help="Export only last N minutes")
    
    args = parser.parse_args()
    
    # Note: In real usage, you would pass an actual engine instance
    # For demo purposes, this shows the monitoring structure
    print("Monitoring tool ready. In production, pass engine instance to PerformanceMonitor.")
    
    if args.summary:
        print(f"\nSummary for last {args.summary} minutes:")
        # Would show summary from actual monitor instance
        print("(Pass engine instance to get actual summary)")
    
    if args.export:
        print(f"\nExporting metrics to {args.export}...")
        # Would export from actual monitor instance
        print("(Pass engine instance to get actual export)")


if __name__ == "__main__":
    asyncio.run(main())

