"""
Health Checker for Instagram Captions API v10.0

System health monitoring and status checks.
"""

import time
import psutil
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class HealthStatus:
    """Health status information."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: float
    response_time: Optional[float] = None

class HealthChecker:
    """System health monitoring and status checks."""
    
    def __init__(self):
        self.health_history: List[HealthStatus] = []
        self.start_time = time.time()
        self.max_history = 1000
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        health_checks = {
            'system': self._check_system_resources(),
            'memory': self._check_memory_usage(),
            'disk': self._check_disk_usage(),
            'network': self._check_network_status(),
            'process': self._check_process_status()
        }
        
        # Determine overall health
        overall_status = "healthy"
        if any(check['status'] == 'unhealthy' for check in health_checks.values()):
            overall_status = "unhealthy"
        elif any(check['status'] == 'degraded' for check in health_checks.values()):
            overall_status = "degraded"
        
        return {
            'overall_status': overall_status,
            'timestamp': time.time(),
            'uptime_seconds': time.time() - self.start_time,
            'checks': health_checks
        }
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            if cpu_percent > 90:
                status = "unhealthy"
                message = f"CPU usage too high: {cpu_percent}%"
            elif cpu_percent > 80:
                status = "degraded"
                message = f"CPU usage elevated: {cpu_percent}%"
            else:
                status = "healthy"
                message = f"CPU usage normal: {cpu_percent}%"
            
            return {
                'status': status,
                'message': message,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'System check failed: {str(e)}',
                'timestamp': time.time()
            }
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent > 95:
                status = "unhealthy"
                message = f"Memory usage critical: {memory.percent}%"
            elif memory.percent > 85:
                status = "degraded"
                message = f"Memory usage high: {memory.percent}%"
            else:
                status = "healthy"
                message = f"Memory usage normal: {memory.percent}%"
            
            return {
                'status': status,
                'message': message,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Memory check failed: {str(e)}',
                'timestamp': time.time()
            }
    
    def _check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage."""
        try:
            disk = psutil.disk_usage('/')
            
            if disk.percent > 95:
                status = "unhealthy"
                message = f"Disk usage critical: {disk.percent}%"
            elif disk.percent > 85:
                status = "degraded"
                message = f"Disk usage high: {disk.percent}%"
            else:
                status = "healthy"
                message = f"Disk usage normal: {disk.percent}%"
            
            return {
                'status': status,
                'message': message,
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Disk check failed: {str(e)}',
                'timestamp': time.time()
            }
    
    def _check_network_status(self) -> Dict[str, Any]:
        """Check network connectivity."""
        try:
            # Simple network check - could be enhanced with actual endpoint checks
            network_io = psutil.net_io_counters()
            
            if network_io.dropin > 1000 or network_io.dropout > 1000:
                status = "degraded"
                message = "Network packet drops detected"
            else:
                status = "healthy"
                message = "Network status normal"
            
            return {
                'status': status,
                'message': message,
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv,
                'packets_sent': network_io.packets_sent,
                'packets_recv': network_io.packets_recv,
                'dropin': network_io.dropin,
                'dropout': network_io.dropout,
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Network check failed: {str(e)}',
                'timestamp': time.time()
            }
    
    def _check_process_status(self) -> Dict[str, Any]:
        """Check current process status."""
        try:
            process = psutil.Process()
            
            # Check if process is responsive
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            
            if cpu_percent > 100:  # Unusually high for a single process
                status = "degraded"
                message = f"Process CPU usage high: {cpu_percent}%"
            else:
                status = "healthy"
                message = f"Process status normal"
            
            return {
                'status': status,
                'message': message,
                'process_id': process.pid,
                'cpu_percent': cpu_percent,
                'memory_rss_mb': memory_info.rss / (1024**2),
                'memory_vms_mb': memory_info.vms / (1024**2),
                'num_threads': process.num_threads(),
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Process check failed: {str(e)}',
                'timestamp': time.time()
            }
    
    async def check_endpoint_health(self, endpoint: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Check health of a specific endpoint."""
        start_time = time.time()
        
        try:
            # This is a placeholder - in a real implementation, you'd make an HTTP request
            # For now, we'll simulate a health check
            await asyncio.sleep(0.1)  # Simulate network delay
            
            response_time = time.time() - start_time
            
            if response_time > timeout:
                status = "unhealthy"
                message = f"Endpoint response time exceeded timeout: {response_time:.3f}s"
            elif response_time > timeout * 0.8:
                status = "degraded"
                message = f"Endpoint response time elevated: {response_time:.3f}s"
            else:
                status = "healthy"
                message = f"Endpoint response time normal: {response_time:.3f}s"
            
            health_status = HealthStatus(
                component=endpoint,
                status=status,
                message=message,
                timestamp=time.time(),
                response_time=response_time
            )
            
            self._record_health_status(health_status)
            
            return {
                'status': status,
                'message': message,
                'response_time': response_time,
                'timestamp': time.time()
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            health_status = HealthStatus(
                component=endpoint,
                status="unhealthy",
                message=f"Health check failed: {str(e)}",
                timestamp=time.time(),
                response_time=response_time
            )
            
            self._record_health_status(health_status)
            
            return {
                'status': 'unhealthy',
                'message': f'Health check failed: {str(e)}',
                'response_time': response_time,
                'timestamp': time.time()
            }
    
    def _record_health_status(self, health_status: HealthStatus):
        """Record health status for historical analysis."""
        self.health_history.append(health_status)
        
        # Keep only recent history
        if len(self.health_history) > self.max_history:
            self.health_history = self.health_history[-self.max_history:]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of health checks."""
        current_time = time.time()
        one_hour_ago = current_time - 3600
        
        recent_checks = [h for h in self.health_history if h.timestamp > one_hour_ago]
        
        status_counts = {}
        for check in recent_checks:
            status_counts[check.status] = status_counts.get(check.status, 0) + 1
        
        return {
            'total_checks': len(self.health_history),
            'recent_checks_1h': len(recent_checks),
            'status_distribution': status_counts,
            'last_check': self.health_history[-1].timestamp if self.health_history else None,
            'uptime_seconds': current_time - self.start_time
        }
    
    def clear_history(self):
        """Clear health check history."""
        self.health_history.clear()






