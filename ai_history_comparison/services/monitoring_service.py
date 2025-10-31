"""
Monitoring Service

This service orchestrates monitoring functionality including
system monitoring, performance monitoring, and health checks.
"""

import asyncio
import logging
import psutil
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.base import BaseService
from ..core.config import SystemConfig
from ..core.exceptions import MonitoringError

logger = logging.getLogger(__name__)


class MonitoringService(BaseService[Dict[str, Any]]):
    """Service for managing system monitoring and health checks"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        super().__init__(config)
        self._monitoring_tasks = []
        self._metrics_history = []
        self._alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "response_time": 5.0
        }
    
    async def _start(self) -> bool:
        """Start the monitoring service"""
        try:
            # Start monitoring tasks
            await self._start_monitoring_tasks()
            
            logger.info("Monitoring service started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring service: {e}")
            return False
    
    async def _stop(self) -> bool:
        """Stop the monitoring service"""
        try:
            # Stop monitoring tasks
            await self._stop_monitoring_tasks()
            
            logger.info("Monitoring service stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring service: {e}")
            return False
    
    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        try:
            # Start system metrics collection
            self._monitoring_tasks.append(
                asyncio.create_task(self._system_metrics_loop())
            )
            
            # Start health check monitoring
            self._monitoring_tasks.append(
                asyncio.create_task(self._health_check_loop())
            )
            
            # Start alert monitoring
            self._monitoring_tasks.append(
                asyncio.create_task(self._alert_monitoring_loop())
            )
            
            logger.info("Monitoring tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring tasks: {e}")
            raise MonitoringError(f"Failed to start monitoring tasks: {str(e)}")
    
    async def _stop_monitoring_tasks(self):
        """Stop background monitoring tasks"""
        try:
            # Cancel all monitoring tasks
            for task in self._monitoring_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self._monitoring_tasks:
                await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
            
            self._monitoring_tasks.clear()
            logger.info("Monitoring tasks stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring tasks: {e}")
    
    async def _system_metrics_loop(self):
        """Background system metrics collection loop"""
        while self._running:
            try:
                # Collect system metrics
                metrics = await self._collect_system_metrics()
                
                # Store metrics
                self._store_metrics(metrics)
                
                # Update service metrics
                self._update_metrics({
                    "system_metrics_collected": 1,
                    "last_metrics_collection": datetime.utcnow().isoformat()
                })
                
                # Wait before next collection
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(30)  # Wait 30 seconds on error
    
    async def _health_check_loop(self):
        """Background health check monitoring loop"""
        while self._running:
            try:
                # Perform health checks
                health_status = await self._perform_health_checks()
                
                # Update service metrics
                self._update_metrics({
                    "health_checks_performed": 1,
                    "last_health_check": datetime.utcnow().isoformat(),
                    "health_status": health_status
                })
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(15)  # Wait 15 seconds on error
    
    async def _alert_monitoring_loop(self):
        """Background alert monitoring loop"""
        while self._running:
            try:
                # Check for alerts
                alerts = await self._check_alerts()
                
                if alerts:
                    # Process alerts
                    await self._process_alerts(alerts)
                
                # Wait before next check
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Alert monitoring error: {e}")
                await asyncio.sleep(5)  # Wait 5 seconds on error
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_free = disk.free
            
            # Network metrics (if available)
            network_io = psutil.net_io_counters()
            
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": cpu_count
                },
                "memory": {
                    "usage_percent": memory_percent,
                    "available_bytes": memory_available,
                    "total_bytes": memory.total
                },
                "disk": {
                    "usage_percent": disk_percent,
                    "free_bytes": disk_free,
                    "total_bytes": disk.total
                },
                "network": {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    
    def _store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics in history"""
        try:
            # Add to history
            self._metrics_history.append(metrics)
            
            # Keep only last 1000 entries
            if len(self._metrics_history) > 1000:
                self._metrics_history = self._metrics_history[-1000:]
            
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    async def _perform_health_checks(self) -> Dict[str, Any]:
        """Perform comprehensive health checks"""
        try:
            health_status = {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "healthy",
                "checks": {}
            }
            
            # System health checks
            health_status["checks"]["system"] = await self._check_system_health()
            
            # Database health check
            health_status["checks"]["database"] = await self._check_database_health()
            
            # Redis health check
            health_status["checks"]["redis"] = await self._check_redis_health()
            
            # API health check
            health_status["checks"]["api"] = await self._check_api_health()
            
            # Determine overall status
            all_healthy = all(
                check.get("status") == "healthy" 
                for check in health_status["checks"].values()
            )
            health_status["overall_status"] = "healthy" if all_healthy else "unhealthy"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "error",
                "error": str(e)
            }
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check system health"""
        try:
            # Get current metrics
            if self._metrics_history:
                latest_metrics = self._metrics_history[-1]
                
                cpu_usage = latest_metrics.get("cpu", {}).get("usage_percent", 0)
                memory_usage = latest_metrics.get("memory", {}).get("usage_percent", 0)
                disk_usage = latest_metrics.get("disk", {}).get("usage_percent", 0)
                
                # Check thresholds
                cpu_healthy = cpu_usage < self._alert_thresholds["cpu_usage"]
                memory_healthy = memory_usage < self._alert_thresholds["memory_usage"]
                disk_healthy = disk_usage < self._alert_thresholds["disk_usage"]
                
                status = "healthy" if all([cpu_healthy, memory_healthy, disk_healthy]) else "unhealthy"
                
                return {
                    "status": status,
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "disk_usage": disk_usage,
                    "thresholds": self._alert_thresholds
                }
            else:
                return {"status": "unknown", "message": "No metrics available"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            # In a real implementation, you would check database connectivity
            # For now, return a placeholder
            return {
                "status": "healthy",
                "connection": "active",
                "response_time": 0.1
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis health"""
        try:
            # In a real implementation, you would check Redis connectivity
            # For now, return a placeholder
            return {
                "status": "healthy",
                "connection": "active",
                "response_time": 0.05
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _check_api_health(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            # In a real implementation, you would check API endpoints
            # For now, return a placeholder
            return {
                "status": "healthy",
                "endpoints": "responsive",
                "response_time": 0.2
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        try:
            alerts = []
            
            if not self._metrics_history:
                return alerts
            
            latest_metrics = self._metrics_history[-1]
            
            # Check CPU usage
            cpu_usage = latest_metrics.get("cpu", {}).get("usage_percent", 0)
            if cpu_usage > self._alert_thresholds["cpu_usage"]:
                alerts.append({
                    "type": "cpu_usage_high",
                    "severity": "warning",
                    "value": cpu_usage,
                    "threshold": self._alert_thresholds["cpu_usage"],
                    "message": f"CPU usage is {cpu_usage:.1f}%, above threshold of {self._alert_thresholds['cpu_usage']}%"
                })
            
            # Check memory usage
            memory_usage = latest_metrics.get("memory", {}).get("usage_percent", 0)
            if memory_usage > self._alert_thresholds["memory_usage"]:
                alerts.append({
                    "type": "memory_usage_high",
                    "severity": "warning",
                    "value": memory_usage,
                    "threshold": self._alert_thresholds["memory_usage"],
                    "message": f"Memory usage is {memory_usage:.1f}%, above threshold of {self._alert_thresholds['memory_usage']}%"
                })
            
            # Check disk usage
            disk_usage = latest_metrics.get("disk", {}).get("usage_percent", 0)
            if disk_usage > self._alert_thresholds["disk_usage"]:
                alerts.append({
                    "type": "disk_usage_high",
                    "severity": "critical",
                    "value": disk_usage,
                    "threshold": self._alert_thresholds["disk_usage"],
                    "message": f"Disk usage is {disk_usage:.1f}%, above threshold of {self._alert_thresholds['disk_usage']}%"
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to check alerts: {e}")
            return []
    
    async def _process_alerts(self, alerts: List[Dict[str, Any]]):
        """Process and handle alerts"""
        try:
            for alert in alerts:
                # Log alert
                logger.warning(f"ALERT: {alert['message']}")
                
                # In a real implementation, you would:
                # - Send notifications
                # - Update alert database
                # - Trigger automated responses
                
                # Update metrics
                self._update_metrics({
                    f"alerts_{alert['type']}": 1,
                    "last_alert": datetime.utcnow().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Failed to process alerts: {e}")
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            if not self._initialized:
                raise MonitoringError("Monitoring service not initialized")
            
            if self._metrics_history:
                return self._metrics_history[-1]
            else:
                return await self._collect_system_metrics()
                
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            raise MonitoringError(f"Failed to get system metrics: {str(e)}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        try:
            if not self._initialized:
                raise MonitoringError("Monitoring service not initialized")
            
            return await self._perform_health_checks()
            
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            raise MonitoringError(f"Failed to get health status: {str(e)}")
    
    async def get_metrics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get metrics history"""
        try:
            if not self._initialized:
                raise MonitoringError("Monitoring service not initialized")
            
            return self._metrics_history[-limit:] if self._metrics_history else []
            
        except Exception as e:
            logger.error(f"Failed to get metrics history: {e}")
            raise MonitoringError(f"Failed to get metrics history: {str(e)}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring service status"""
        base_status = self.get_health_status()
        base_status.update({
            "monitoring_tasks": len(self._monitoring_tasks),
            "metrics_history_size": len(self._metrics_history),
            "alert_thresholds": self._alert_thresholds,
            "features_enabled": {
                "system_monitoring": True,
                "health_checks": True,
                "alert_monitoring": True
            }
        })
        return base_status





















