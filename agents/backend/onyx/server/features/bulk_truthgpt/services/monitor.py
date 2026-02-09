"""
System Monitor
=============

Advanced monitoring system for the Bulk TruthGPT system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import psutil
import json

logger = logging.getLogger(__name__)

class SystemMonitor:
    """
    Advanced system monitoring for Bulk TruthGPT.
    
    Features:
    - System metrics
    - Performance monitoring
    - Health checks
    - Alerting
    - Dashboard data
    """
    
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        self.health_status = {}
        
    async def initialize(self):
        """Initialize system monitor."""
        logger.info("Initializing System Monitor...")
        
        try:
            # Start background monitoring
            asyncio.create_task(self._monitor_system())
            asyncio.create_task(self._cleanup_old_metrics())
            
            logger.info("System Monitor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize System Monitor: {str(e)}")
            raise
    
    async def _monitor_system(self):
        """Background system monitoring."""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Collect system metrics
                metrics = await self._collect_system_metrics()
                
                # Store metrics
                self.metrics_history.append({
                    'timestamp': datetime.utcnow(),
                    'metrics': metrics
                })
                
                # Keep only recent metrics (last 24 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m['timestamp'] > cutoff_time
                ]
                
                # Check for alerts
                await self._check_alerts(metrics)
                
            except Exception as e:
                logger.error(f"System monitoring error: {str(e)}")
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            memory_total = memory.total
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free
            disk_total = disk.total
            
            # Network metrics
            network = psutil.net_io_counters()
            
            return {
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count
                },
                'memory': {
                    'percent': memory_percent,
                    'available_bytes': memory_available,
                    'total_bytes': memory_total
                },
                'disk': {
                    'percent': disk_percent,
                    'free_bytes': disk_free,
                    'total_bytes': disk_total
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}")
            return {}
    
    async def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for alert conditions."""
        try:
            # CPU alert
            if metrics.get('cpu', {}).get('percent', 0) > 80:
                await self._create_alert('high_cpu', f"High CPU usage: {metrics['cpu']['percent']}%")
            
            # Memory alert
            if metrics.get('memory', {}).get('percent', 0) > 85:
                await self._create_alert('high_memory', f"High memory usage: {metrics['memory']['percent']}%")
            
            # Disk alert
            if metrics.get('disk', {}).get('percent', 0) > 90:
                await self._create_alert('high_disk', f"High disk usage: {metrics['disk']['percent']}%")
                
        except Exception as e:
            logger.error(f"Failed to check alerts: {str(e)}")
    
    async def _create_alert(self, alert_type: str, message: str):
        """Create system alert."""
        try:
            alert = {
                'id': f"{alert_type}_{datetime.utcnow().timestamp()}",
                'type': alert_type,
                'message': message,
                'timestamp': datetime.utcnow(),
                'severity': 'warning'
            }
            
            self.alerts.append(alert)
            
            # Keep only recent alerts (last 7 days)
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            self.alerts = [
                a for a in self.alerts 
                if a['timestamp'] > cutoff_time
            ]
            
            logger.warning(f"System alert: {message}")
            
        except Exception as e:
            logger.error(f"Failed to create alert: {str(e)}")
    
    async def _cleanup_old_metrics(self):
        """Cleanup old metrics."""
        while True:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                
                # Remove metrics older than 7 days
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m['timestamp'] > cutoff_time
                ]
                
                # Remove alerts older than 7 days
                self.alerts = [
                    a for a in self.alerts 
                    if a['timestamp'] > cutoff_time
                ]
                
            except Exception as e:
                logger.error(f"Failed to cleanup old metrics: {str(e)}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            metrics = await self._collect_system_metrics()
            
            # Calculate health score
            health_score = self._calculate_health_score(metrics)
            
            # Determine overall status
            if health_score >= 0.8:
                status = "healthy"
            elif health_score >= 0.6:
                status = "warning"
            else:
                status = "critical"
            
            return {
                'status': status,
                'health_score': health_score,
                'metrics': metrics,
                'alerts_count': len(self.alerts),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {str(e)}")
            return {'status': 'unknown', 'error': str(e)}
    
    def _calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate system health score."""
        try:
            cpu_score = 1.0 - (metrics.get('cpu', {}).get('percent', 0) / 100)
            memory_score = 1.0 - (metrics.get('memory', {}).get('percent', 0) / 100)
            disk_score = 1.0 - (metrics.get('disk', {}).get('percent', 0) / 100)
            
            # Weighted average
            health_score = (cpu_score * 0.4 + memory_score * 0.4 + disk_score * 0.2)
            
            return max(0.0, min(1.0, health_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate health score: {str(e)}")
            return 0.0
    
    async def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            return [
                {
                    'timestamp': m['timestamp'].isoformat(),
                    'metrics': m['metrics']
                }
                for m in self.metrics_history
                if m['timestamp'] > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Failed to get metrics history: {str(e)}")
            return []
    
    async def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get system alerts."""
        try:
            return [
                {
                    'id': alert['id'],
                    'type': alert['type'],
                    'message': alert['message'],
                    'timestamp': alert['timestamp'].isoformat(),
                    'severity': alert['severity']
                }
                for alert in self.alerts[-limit:]
            ]
            
        except Exception as e:
            logger.error(f"Failed to get alerts: {str(e)}")
            return []
    
    async def update_generation_metrics(
        self, 
        task_id: str, 
        documents_generated: int, 
        document_id: str
    ):
        """Update generation metrics."""
        try:
            # This would integrate with the generation system
            # to track generation-specific metrics
            
            logger.info(f"Updated generation metrics for task {task_id}: {documents_generated} documents")
            
        except Exception as e:
            logger.error(f"Failed to update generation metrics: {str(e)}")
    
    async def get_generation_metrics(self) -> Dict[str, Any]:
        """Get generation metrics."""
        try:
            # This would return generation-specific metrics
            return {
                'total_tasks': 0,
                'active_tasks': 0,
                'completed_tasks': 0,
                'total_documents': 0,
                'average_generation_time': 0.0,
                'success_rate': 0.0
            }
            
        except Exception as e:
            logger.error(f"Failed to get generation metrics: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Cleanup system monitor."""
        try:
            logger.info("System Monitor cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup System Monitor: {str(e)}")











