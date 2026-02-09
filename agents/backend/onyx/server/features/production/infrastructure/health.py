from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
            import psutil
from typing import Any, List, Dict, Optional
"""
Health Check Infrastructure
===========================

Comprehensive health checker service for all system components.
"""


logger = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    response_time: float
    details: Dict[str, Any]
    timestamp: datetime
    error: Optional[str] = None


class HealthChecker:
    """Comprehensive health checker for all system components."""
    
    def __init__(self, container) -> Any:
        self.container = container
        self.health_checks = {
            "database": self._check_database,
            "cache": self._check_cache,
            "ai_service": self._check_ai_service,
            "event_publisher": self._check_event_publisher,
            "monitoring": self._check_monitoring,
            "system": self._check_system,
            "api": self._check_api
        }
        
        # Health check history
        self.health_history: List[HealthCheck] = []
        self.max_history_size = 100
        
        # Background monitoring
        self._monitoring_task = None
        self._running = False
        
        logger.info("HealthChecker initialized")
    
    async def initialize(self) -> Any:
        """Initialize health checker."""
        try:
            # Start background monitoring
            self._monitoring_task = asyncio.create_task(self._monitor_health())
            self._running = True
            
            logger.info("HealthChecker initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize HealthChecker: {e}")
            raise
    
    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            start_time = time.time()
            health_results = {}
            overall_status = "healthy"
            
            # Run all health checks concurrently
            tasks = []
            for check_name, check_func in self.health_checks.items():
                task = asyncio.create_task(self._run_health_check(check_name, check_func))
                tasks.append(task)
            
            # Wait for all checks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, (check_name, check_func) in enumerate(self.health_checks.items()):
                result = results[i]
                
                if isinstance(result, Exception):
                    health_results[check_name] = {
                        "status": "unhealthy",
                        "error": str(result),
                        "response_time": 0.0,
                        "timestamp": datetime.now().isoformat()
                    }
                    overall_status = "unhealthy"
                else:
                    health_results[check_name] = result
                    
                    if result["status"] == "unhealthy":
                        overall_status = "unhealthy"
                    elif result["status"] == "degraded" and overall_status == "healthy":
                        overall_status = "degraded"
            
            # Calculate overall response time
            total_time = time.time() - start_time
            
            # Create health check result
            health_check = HealthCheck(
                name="overall",
                status=overall_status,
                response_time=total_time,
                details=health_results,
                timestamp=datetime.now()
            )
            
            # Store in history
            self._store_health_check(health_check)
            
            return {
                "status": overall_status,
                "checks": health_results,
                "response_time": total_time,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _run_health_check(self, name: str, check_func) -> Dict[str, Any]:
        """Run individual health check with timeout."""
        try:
            start_time = time.time()
            
            # Run check with timeout
            result = await asyncio.wait_for(check_func(), timeout=10.0)
            
            response_time = time.time() - start_time
            
            # Add response time to result
            result["response_time"] = response_time
            result["timestamp"] = datetime.now().isoformat()
            
            return result
            
        except asyncio.TimeoutError:
            return {
                "status": "unhealthy",
                "error": "Health check timeout",
                "response_time": 10.0,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time": 0.0,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            repository = self.container.get_service("repository")
            
            # Test database connection
            start_time = time.time()
            stats = await repository.get_statistics()
            response_time = time.time() - start_time
            
            # Check if stats are reasonable
            if not stats or "total_requests" not in stats:
                return {
                    "status": "unhealthy",
                    "error": "Invalid database response",
                    "details": {"stats": stats}
                }
            
            status = "healthy"
            if response_time > 5.0:
                status = "degraded"
            if response_time > 10.0:
                status = "unhealthy"
            
            return {
                "status": status,
                "details": {
                    "total_requests": stats.get("total_requests", 0),
                    "total_responses": stats.get("total_responses", 0),
                    "average_processing_time": stats.get("average_processing_time", 0.0),
                    "recent_requests_24h": stats.get("recent_requests_24h", 0)
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_cache(self) -> Dict[str, Any]:
        """Check cache health."""
        try:
            cache_service = self.container.get_service("cache_service")
            
            # Test cache operations
            test_key = "health_check_test"
            test_value = {"test": "data", "timestamp": datetime.now().isoformat()}
            
            # Set test value
            await cache_service.set(test_key, test_value, ttl=60)
            
            # Get test value
            retrieved_value = await cache_service.get(test_key)
            
            # Delete test value
            await cache_service.delete(test_key)
            
            if retrieved_value != test_value:
                return {
                    "status": "unhealthy",
                    "error": "Cache data integrity check failed"
                }
            
            # Get cache stats
            cache_stats = await cache_service.get_cache_stats()
            
            status = "healthy"
            if cache_stats.get("hit_ratio", 1.0) < 0.5:
                status = "degraded"
            
            return {
                "status": status,
                "details": {
                    "hit_ratio": cache_stats.get("hit_ratio", 0.0),
                    "total_requests": cache_stats.get("total_requests", 0),
                    "memory_usage": cache_stats.get("memory_usage", 0)
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_ai_service(self) -> Dict[str, Any]:
        """Check AI service health."""
        try:
            ai_service = self.container.get_service("ai_service")
            
            # Check if AI service is available
            is_available = await ai_service.is_available()
            
            if not is_available:
                return {
                    "status": "unhealthy",
                    "error": "AI service not available"
                }
            
            # Get model info
            model_info = await ai_service.get_model_info()
            
            # Check GPU usage if available
            gpu_usage = 0.0
            if model_info.get("use_gpu"):
                gpu_info = model_info.get("gpu_info", {})
                gpu_usage = gpu_info.get("memory_allocated", 0) / gpu_info.get("memory_reserved", 1) * 100
            
            status = "healthy"
            if gpu_usage > 90:
                status = "degraded"
            if gpu_usage > 95:
                status = "unhealthy"
            
            return {
                "status": status,
                "details": {
                    "model_name": model_info.get("model_name", "unknown"),
                    "use_gpu": model_info.get("use_gpu", False),
                    "gpu_usage": gpu_usage,
                    "cache_size": model_info.get("cache_size", 0),
                    "performance_metrics": model_info.get("performance_metrics", {})
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_event_publisher(self) -> Dict[str, Any]:
        """Check event publisher health."""
        try:
            event_publisher = self.container.get_service("event_publisher")
            
            # Get event stats
            event_stats = await event_publisher.get_event_stats()
            
            # Check for errors
            error_count = event_stats.get("errors", 0)
            
            status = "healthy"
            if error_count > 10:
                status = "degraded"
            if error_count > 50:
                status = "unhealthy"
            
            return {
                "status": status,
                "details": {
                    "events_published": event_stats.get("events_published", 0),
                    "events_handled": event_stats.get("events_handled", 0),
                    "errors": error_count,
                    "active_handlers": event_stats.get("active_handlers", 0)
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_monitoring(self) -> Dict[str, Any]:
        """Check monitoring service health."""
        try:
            monitoring_service = self.container.get_service("monitoring_service")
            
            if not monitoring_service:
                return {
                    "status": "healthy",
                    "details": {"message": "Monitoring service not configured"}
                }
            
            # Get monitoring metrics
            metrics = await monitoring_service.get_metrics()
            
            # Check if metrics are being collected
            if "error" in metrics:
                return {
                    "status": "unhealthy",
                    "error": metrics["error"]
                }
            
            return {
                "status": "healthy",
                "details": {
                    "metrics_available": "prometheus" in metrics,
                    "custom_metrics": "custom" in metrics
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_system(self) -> Dict[str, Any]:
        """Check system health."""
        try:
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Network connectivity
            network_io = psutil.net_io_counters()
            
            # Determine overall status
            status = "healthy"
            if cpu_percent > 90 or memory.percent > 90 or disk_percent > 90:
                status = "degraded"
            if cpu_percent > 95 or memory.percent > 95 or disk_percent > 95:
                status = "unhealthy"
            
            return {
                "status": status,
                "details": {
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory.percent,
                    "memory_available": memory.available,
                    "disk_usage": disk_percent,
                    "disk_free": disk_usage.free,
                    "network_bytes_sent": network_io.bytes_sent,
                    "network_bytes_recv": network_io.bytes_recv
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async async def _check_api(self) -> Dict[str, Any]:
        """Check API health."""
        try:
            # This would typically check if the API endpoints are responding
            # For now, we'll just return a basic health status
            
            return {
                "status": "healthy",
                "details": {
                    "endpoints_available": True,
                    "api_version": "1.0.0"
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def _store_health_check(self, health_check: HealthCheck):
        """Store health check result in history."""
        self.health_history.append(health_check)
        
        # Keep only recent history
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size:]
    
    async def get_health_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get health check history."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            history = []
            for check in self.health_history:
                if check.timestamp > cutoff_time:
                    history.append({
                        "name": check.name,
                        "status": check.status,
                        "response_time": check.response_time,
                        "timestamp": check.timestamp.isoformat(),
                        "error": check.error
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting health history: {e}")
            return []
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get health check summary."""
        try:
            if not self.health_history:
                return {"message": "No health check history available"}
            
            # Get recent checks (last hour)
            recent_checks = [check for check in self.health_history 
                           if check.timestamp > datetime.now() - timedelta(hours=1)]
            
            if not recent_checks:
                return {"message": "No recent health checks"}
            
            # Calculate statistics
            total_checks = len(recent_checks)
            healthy_checks = sum(1 for check in recent_checks if check.status == "healthy")
            degraded_checks = sum(1 for check in recent_checks if check.status == "degraded")
            unhealthy_checks = sum(1 for check in recent_checks if check.status == "unhealthy")
            
            avg_response_time = sum(check.response_time for check in recent_checks) / total_checks
            
            return {
                "total_checks": total_checks,
                "healthy_percentage": (healthy_checks / total_checks) * 100,
                "degraded_percentage": (degraded_checks / total_checks) * 100,
                "unhealthy_percentage": (unhealthy_checks / total_checks) * 100,
                "average_response_time": avg_response_time,
                "last_check": self.health_history[-1].timestamp.isoformat() if self.health_history else None
            }
            
        except Exception as e:
            logger.error(f"Error getting health summary: {e}")
            return {"error": str(e)}
    
    async def _monitor_health(self) -> Any:
        """Background health monitoring."""
        while self._running:
            try:
                # Perform health check every 5 minutes
                await self.check_health()
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def cleanup(self) -> Any:
        """Cleanup health checker."""
        try:
            self._running = False
            
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("HealthChecker cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up HealthChecker: {e}") 