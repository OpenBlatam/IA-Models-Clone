"""
Metrics Collector
=================

Advanced metrics collection system for performance monitoring.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import time
import psutil
from dataclasses import dataclass
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class Metric:
    """Metric data structure."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    metadata: Dict[str, Any]

class MetricsCollector:
    """
    Advanced metrics collector.
    
    Features:
    - System metrics
    - Application metrics
    - Custom metrics
    - Aggregation
    - Alerting
    - Export capabilities
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.aggregated_metrics = defaultdict(list)
        self.alerts = []
        self.collection_interval = 60  # seconds
        self.retention_days = 7
        
    async def initialize(self):
        """Initialize metrics collector."""
        logger.info("Initializing Metrics Collector...")
        
        try:
            # Start background collection
            asyncio.create_task(self._collect_system_metrics())
            asyncio.create_task(self._aggregate_metrics())
            asyncio.create_task(self._cleanup_old_metrics())
            
            logger.info("Metrics Collector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Metrics Collector: {str(e)}")
            raise
    
    async def _collect_system_metrics(self):
        """Collect system metrics in background."""
        while True:
            try:
                await asyncio.sleep(self.collection_interval)
                
                # Collect CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                await self.record_metric('system.cpu.percent', cpu_percent, {'type': 'system'})
                
                # Collect memory metrics
                memory = psutil.virtual_memory()
                await self.record_metric('system.memory.percent', memory.percent, {'type': 'system'})
                await self.record_metric('system.memory.available', memory.available, {'type': 'system'})
                await self.record_metric('system.memory.used', memory.used, {'type': 'system'})
                
                # Collect disk metrics
                disk = psutil.disk_usage('/')
                await self.record_metric('system.disk.percent', disk.percent, {'type': 'system'})
                await self.record_metric('system.disk.free', disk.free, {'type': 'system'})
                await self.record_metric('system.disk.used', disk.used, {'type': 'system'})
                
                # Collect network metrics
                network = psutil.net_io_counters()
                await self.record_metric('system.network.bytes_sent', network.bytes_sent, {'type': 'system'})
                await self.record_metric('system.network.bytes_recv', network.bytes_recv, {'type': 'system'})
                
                # Collect process metrics
                process = psutil.Process()
                await self.record_metric('system.process.cpu_percent', process.cpu_percent(), {'type': 'system'})
                await self.record_metric('system.process.memory_percent', process.memory_percent(), {'type': 'system'})
                await self.record_metric('system.process.memory_info', process.memory_info().rss, {'type': 'system'})
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {str(e)}")
    
    async def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a metric."""
        try:
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                tags=tags or {},
                metadata=metadata or {}
            )
            
            self.metrics[name].append(metric)
            
            # Check for alerts
            await self._check_alerts(name, value, tags)
            
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {str(e)}")
    
    async def _check_alerts(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Check for metric alerts."""
        try:
            # Define alert thresholds
            thresholds = {
                'system.cpu.percent': 80.0,
                'system.memory.percent': 85.0,
                'system.disk.percent': 90.0,
                'system.process.cpu_percent': 90.0,
                'system.process.memory_percent': 80.0
            }
            
            threshold = thresholds.get(name)
            if threshold and value > threshold:
                await self._create_alert(name, value, threshold, tags)
                
        except Exception as e:
            logger.error(f"Failed to check alerts for {name}: {str(e)}")
    
    async def _create_alert(self, name: str, value: float, threshold: float, tags: Optional[Dict[str, str]] = None):
        """Create metric alert."""
        try:
            alert = {
                'id': f"{name}_{int(time.time())}",
                'metric_name': name,
                'value': value,
                'threshold': threshold,
                'tags': tags or {},
                'timestamp': datetime.utcnow(),
                'severity': 'warning' if value < threshold * 1.2 else 'critical'
            }
            
            self.alerts.append(alert)
            
            # Keep only recent alerts
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            self.alerts = [a for a in self.alerts if a['timestamp'] > cutoff_time]
            
            logger.warning(f"Metric alert: {name} = {value} (threshold: {threshold})")
            
        except Exception as e:
            logger.error(f"Failed to create alert: {str(e)}")
    
    async def _aggregate_metrics(self):
        """Aggregate metrics for reporting."""
        while True:
            try:
                await asyncio.sleep(300)  # Aggregate every 5 minutes
                
                current_time = datetime.utcnow()
                aggregation_window = timedelta(minutes=5)
                
                for metric_name, metrics in self.metrics.items():
                    if not metrics:
                        continue
                    
                    # Get metrics from last 5 minutes
                    recent_metrics = [
                        m for m in metrics
                        if m.timestamp > current_time - aggregation_window
                    ]
                    
                    if recent_metrics:
                        values = [m.value for m in recent_metrics]
                        
                        aggregated = {
                            'metric_name': metric_name,
                            'count': len(values),
                            'average': sum(values) / len(values),
                            'min': min(values),
                            'max': max(values),
                            'sum': sum(values),
                            'timestamp': current_time,
                            'tags': recent_metrics[0].tags if recent_metrics else {}
                        }
                        
                        self.aggregated_metrics[metric_name].append(aggregated)
                        
                        # Keep only recent aggregated data
                        cutoff_time = current_time - timedelta(days=7)
                        self.aggregated_metrics[metric_name] = [
                            a for a in self.aggregated_metrics[metric_name]
                            if a['timestamp'] > cutoff_time
                        ]
                
            except Exception as e:
                logger.error(f"Error aggregating metrics: {str(e)}")
    
    async def _cleanup_old_metrics(self):
        """Cleanup old metrics."""
        while True:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                
                cutoff_time = datetime.utcnow() - timedelta(days=self.retention_days)
                
                # Cleanup raw metrics
                for metric_name in self.metrics:
                    self.metrics[metric_name] = [
                        m for m in self.metrics[metric_name]
                        if m.timestamp > cutoff_time
                    ]
                
                # Cleanup aggregated metrics
                for metric_name in self.aggregated_metrics:
                    self.aggregated_metrics[metric_name] = [
                        a for a in self.aggregated_metrics[metric_name]
                        if a['timestamp'] > cutoff_time
                    ]
                
                # Cleanup alerts
                self.alerts = [
                    a for a in self.alerts
                    if a['timestamp'] > cutoff_time
                ]
                
            except Exception as e:
                logger.error(f"Error cleaning up old metrics: {str(e)}")
    
    async def get_metric_history(
        self,
        metric_name: str,
        time_range: str = "24h",
        aggregation: str = "raw"
    ) -> List[Dict[str, Any]]:
        """Get metric history."""
        try:
            # Calculate time range
            if time_range == "1h":
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
            elif time_range == "24h":
                cutoff_time = datetime.utcnow() - timedelta(days=1)
            elif time_range == "7d":
                cutoff_time = datetime.utcnow() - timedelta(days=7)
            elif time_range == "30d":
                cutoff_time = datetime.utcnow() - timedelta(days=30)
            else:
                cutoff_time = datetime.utcnow() - timedelta(days=1)
            
            if aggregation == "raw":
                # Return raw metrics
                metrics = [
                    m for m in self.metrics.get(metric_name, [])
                    if m.timestamp > cutoff_time
                ]
                
                return [
                    {
                        'name': m.name,
                        'value': m.value,
                        'timestamp': m.timestamp.isoformat(),
                        'tags': m.tags,
                        'metadata': m.metadata
                    }
                    for m in metrics
                ]
            else:
                # Return aggregated metrics
                aggregated = [
                    a for a in self.aggregated_metrics.get(metric_name, [])
                    if a['timestamp'] > cutoff_time
                ]
                
                return [
                    {
                        'metric_name': a['metric_name'],
                        'count': a['count'],
                        'average': a['average'],
                        'min': a['min'],
                        'max': a['max'],
                        'sum': a['sum'],
                        'timestamp': a['timestamp'].isoformat(),
                        'tags': a['tags']
                    }
                    for a in aggregated
                ]
                
        except Exception as e:
            logger.error(f"Failed to get metric history: {str(e)}")
            return []
    
    async def get_metric_summary(
        self,
        metric_name: str,
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """Get metric summary."""
        try:
            # Calculate time range
            if time_range == "1h":
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
            elif time_range == "24h":
                cutoff_time = datetime.utcnow() - timedelta(days=1)
            elif time_range == "7d":
                cutoff_time = datetime.utcnow() - timedelta(days=7)
            elif time_range == "30d":
                cutoff_time = datetime.utcnow() - timedelta(days=30)
            else:
                cutoff_time = datetime.utcnow() - timedelta(days=1)
            
            # Get recent metrics
            recent_metrics = [
                m for m in self.metrics.get(metric_name, [])
                if m.timestamp > cutoff_time
            ]
            
            if not recent_metrics:
                return {
                    'metric_name': metric_name,
                    'time_range': time_range,
                    'count': 0,
                    'average': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'sum': 0.0,
                    'latest': 0.0,
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            values = [m.value for m in recent_metrics]
            
            return {
                'metric_name': metric_name,
                'time_range': time_range,
                'count': len(values),
                'average': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'sum': sum(values),
                'latest': values[-1] if values else 0.0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get metric summary: {str(e)}")
            return {}
    
    async def get_all_metrics(self) -> Dict[str, Any]:
        """Get all available metrics."""
        try:
            return {
                'raw_metrics': list(self.metrics.keys()),
                'aggregated_metrics': list(self.aggregated_metrics.keys()),
                'total_metrics': len(self.metrics),
                'total_aggregated': len(self.aggregated_metrics),
                'alerts_count': len(self.alerts),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get all metrics: {str(e)}")
            return {}
    
    async def get_alerts(
        self,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get metric alerts."""
        try:
            alerts = self.alerts
            
            if severity:
                alerts = [a for a in alerts if a['severity'] == severity]
            
            # Sort by timestamp (newest first)
            alerts.sort(key=lambda a: a['timestamp'], reverse=True)
            
            # Limit results
            alerts = alerts[:limit]
            
            return [
                {
                    'id': a['id'],
                    'metric_name': a['metric_name'],
                    'value': a['value'],
                    'threshold': a['threshold'],
                    'severity': a['severity'],
                    'tags': a['tags'],
                    'timestamp': a['timestamp'].isoformat()
                }
                for a in alerts
            ]
            
        except Exception as e:
            logger.error(f"Failed to get alerts: {str(e)}")
            return []
    
    async def export_metrics(
        self,
        format: str = "json",
        time_range: str = "24h"
    ) -> str:
        """Export metrics."""
        try:
            # Calculate time range
            if time_range == "1h":
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
            elif time_range == "24h":
                cutoff_time = datetime.utcnow() - timedelta(days=1)
            elif time_range == "7d":
                cutoff_time = datetime.utcnow() - timedelta(days=7)
            elif time_range == "30d":
                cutoff_time = datetime.utcnow() - timedelta(days=30)
            else:
                cutoff_time = datetime.utcnow() - timedelta(days=1)
            
            # Collect all metrics
            all_metrics = []
            for metric_name, metrics in self.metrics.items():
                recent_metrics = [
                    m for m in metrics
                    if m.timestamp > cutoff_time
                ]
                
                for metric in recent_metrics:
                    all_metrics.append({
                        'name': metric.name,
                        'value': metric.value,
                        'timestamp': metric.timestamp.isoformat(),
                        'tags': metric.tags,
                        'metadata': metric.metadata
                    })
            
            if format == "json":
                return json.dumps(all_metrics, indent=2)
            elif format == "csv":
                import csv
                import io
                
                output = io.StringIO()
                if all_metrics:
                    writer = csv.DictWriter(output, fieldnames=all_metrics[0].keys())
                    writer.writeheader()
                    writer.writerows(all_metrics)
                return output.getvalue()
            else:
                return str(all_metrics)
                
        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")
            return ""
    
    async def cleanup(self):
        """Cleanup metrics collector."""
        try:
            logger.info("Metrics Collector cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Metrics Collector: {str(e)}")











