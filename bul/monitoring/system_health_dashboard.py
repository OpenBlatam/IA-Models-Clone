"""
Ultimate BUL System - System Health Dashboard
Comprehensive system health monitoring with real-time status and alerts
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import psutil
import aiohttp
import redis
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"

class ComponentType(str, Enum):
    """Component types"""
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    AI_MODEL = "ai_model"
    WORKFLOW = "workflow"
    ANALYTICS = "analytics"
    INTEGRATION = "integration"
    MONITORING = "monitoring"
    STORAGE = "storage"
    NETWORK = "network"

@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    component_type: ComponentType
    endpoint: str
    timeout: int = 30
    interval: int = 60
    retries: int = 3
    critical: bool = True
    enabled: bool = True
    expected_status: int = 200
    expected_response: Optional[Dict[str, Any]] = None

@dataclass
class HealthResult:
    """Health check result"""
    name: str
    component_type: ComponentType
    status: HealthStatus
    message: str
    response_time: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class SystemHealth:
    """System health summary"""
    overall_status: HealthStatus
    timestamp: datetime
    components: Dict[str, HealthResult]
    metrics: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    uptime: float
    version: str

class SystemHealthDashboard:
    """Comprehensive system health monitoring dashboard"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.health_checks = self._initialize_health_checks()
        self.health_results = {}
        self.monitoring_active = False
        self.start_time = datetime.utcnow()
        
        # Prometheus metrics
        self.prometheus_metrics = self._initialize_prometheus_metrics()
        
        # Health check history
        self.health_history = []
        self.alert_history = []
        
        # Component status tracking
        self.component_status = {}
        self.last_check_times = {}
        
        # Performance tracking
        self.performance_metrics = {
            "response_times": [],
            "error_rates": [],
            "availability": []
        }
        
        # Start monitoring
        self.start_monitoring()
    
    def _initialize_health_checks(self) -> List[HealthCheck]:
        """Initialize health checks for all components"""
        return [
            # API Health Checks
            HealthCheck(
                name="bul_api",
                component_type=ComponentType.API,
                endpoint="/health",
                timeout=10,
                interval=30,
                critical=True
            ),
            HealthCheck(
                name="bul_api_docs",
                component_type=ComponentType.API,
                endpoint="/docs",
                timeout=10,
                interval=60,
                critical=False
            ),
            
            # Database Health Checks
            HealthCheck(
                name="postgresql",
                component_type=ComponentType.DATABASE,
                endpoint="/health/database",
                timeout=15,
                interval=60,
                critical=True
            ),
            
            # Cache Health Checks
            HealthCheck(
                name="redis",
                component_type=ComponentType.CACHE,
                endpoint="/health/cache",
                timeout=10,
                interval=30,
                critical=True
            ),
            
            # AI Model Health Checks
            HealthCheck(
                name="openai",
                component_type=ComponentType.AI_MODEL,
                endpoint="/health/ai/openai",
                timeout=30,
                interval=120,
                critical=True
            ),
            HealthCheck(
                name="anthropic",
                component_type=ComponentType.AI_MODEL,
                endpoint="/health/ai/anthropic",
                timeout=30,
                interval=120,
                critical=True
            ),
            HealthCheck(
                name="openrouter",
                component_type=ComponentType.AI_MODEL,
                endpoint="/health/ai/openrouter",
                timeout=30,
                interval=120,
                critical=False
            ),
            
            # Workflow Health Checks
            HealthCheck(
                name="workflow_engine",
                component_type=ComponentType.WORKFLOW,
                endpoint="/health/workflows",
                timeout=15,
                interval=60,
                critical=True
            ),
            HealthCheck(
                name="celery_worker",
                component_type=ComponentType.WORKFLOW,
                endpoint="/health/celery",
                timeout=15,
                interval=60,
                critical=True
            ),
            
            # Analytics Health Checks
            HealthCheck(
                name="analytics_dashboard",
                component_type=ComponentType.ANALYTICS,
                endpoint="/health/analytics",
                timeout=15,
                interval=120,
                critical=False
            ),
            
            # Integration Health Checks
            HealthCheck(
                name="google_integration",
                component_type=ComponentType.INTEGRATION,
                endpoint="/health/integrations/google",
                timeout=20,
                interval=300,
                critical=False
            ),
            HealthCheck(
                name="microsoft_integration",
                component_type=ComponentType.INTEGRATION,
                endpoint="/health/integrations/microsoft",
                timeout=20,
                interval=300,
                critical=False
            ),
            
            # Monitoring Health Checks
            HealthCheck(
                name="prometheus",
                component_type=ComponentType.MONITORING,
                endpoint="/metrics",
                timeout=10,
                interval=120,
                critical=False
            ),
            HealthCheck(
                name="grafana",
                component_type=ComponentType.MONITORING,
                endpoint="/api/health",
                timeout=10,
                interval=120,
                critical=False
            ),
            
            # Storage Health Checks
            HealthCheck(
                name="minio",
                component_type=ComponentType.STORAGE,
                endpoint="/minio/health/live",
                timeout=10,
                interval=120,
                critical=False
            ),
            
            # Network Health Checks
            HealthCheck(
                name="external_connectivity",
                component_type=ComponentType.NETWORK,
                endpoint="https://httpbin.org/status/200",
                timeout=10,
                interval=300,
                critical=False
            )
        ]
    
    def _initialize_prometheus_metrics(self) -> Dict[str, Any]:
        """Initialize Prometheus metrics for health monitoring"""
        return {
            "health_check_duration": Histogram(
                "bul_health_check_duration_seconds",
                "Health check duration in seconds",
                ["component", "name"]
            ),
            "health_check_status": Gauge(
                "bul_health_check_status",
                "Health check status (1=healthy, 0.5=warning, 0=critical)",
                ["component", "name"]
            ),
            "system_uptime": Gauge(
                "bul_system_uptime_seconds",
                "System uptime in seconds"
            ),
            "component_availability": Gauge(
                "bul_component_availability",
                "Component availability percentage",
                ["component"]
            ),
            "alert_count": Gauge(
                "bul_alert_count",
                "Number of active alerts",
                ["severity"]
            )
        }
    
    async def start_monitoring(self):
        """Start system health monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("Starting system health monitoring")
        
        # Start Prometheus metrics server
        start_http_server(9091)
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_health_checks())
        asyncio.create_task(self._update_system_metrics())
        asyncio.create_task(self._cleanup_old_data())
    
    async def stop_monitoring(self):
        """Stop system health monitoring"""
        self.monitoring_active = False
        logger.info("Stopping system health monitoring")
    
    async def _monitor_health_checks(self):
        """Monitor all health checks"""
        while self.monitoring_active:
            try:
                # Run health checks in parallel
                tasks = []
                for health_check in self.health_checks:
                    if health_check.enabled:
                        # Check if it's time to run this health check
                        last_check = self.last_check_times.get(health_check.name, datetime.min)
                        if (datetime.utcnow() - last_check).seconds >= health_check.interval:
                            tasks.append(self._run_health_check(health_check))
                
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, Exception):
                            logger.error(f"Health check failed with exception: {result}")
                        else:
                            self._process_health_result(result)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in health check monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _run_health_check(self, health_check: HealthCheck) -> HealthResult:
        """Run a single health check"""
        start_time = time.time()
        
        try:
            # Update last check time
            self.last_check_times[health_check.name] = datetime.utcnow()
            
            # Make HTTP request
            timeout = aiohttp.ClientTimeout(total=health_check.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{self.config.get('base_url', 'http://localhost:8000')}{health_check.endpoint}"
                
                async with session.get(url) as response:
                    response_time = time.time() - start_time
                    
                    # Determine status based on response
                    if response.status == health_check.expected_status:
                        status = HealthStatus.HEALTHY
                        message = f"Component is healthy"
                    elif response.status < 500:
                        status = HealthStatus.WARNING
                        message = f"Component returned status {response.status}"
                    else:
                        status = HealthStatus.CRITICAL
                        message = f"Component returned error status {response.status}"
                    
                    # Check response content if expected
                    details = {}
                    if health_check.expected_response:
                        try:
                            response_data = await response.json()
                            details["response_data"] = response_data
                        except:
                            pass
                    
                    result = HealthResult(
                        name=health_check.name,
                        component_type=health_check.component_type,
                        status=status,
                        message=message,
                        response_time=response_time,
                        timestamp=datetime.utcnow(),
                        details=details
                    )
                    
                    # Update Prometheus metrics
                    self.prometheus_metrics["health_check_duration"].labels(
                        component=health_check.component_type.value,
                        name=health_check.name
                    ).observe(response_time)
                    
                    status_value = 1.0 if status == HealthStatus.HEALTHY else (0.5 if status == HealthStatus.WARNING else 0.0)
                    self.prometheus_metrics["health_check_status"].labels(
                        component=health_check.component_type.value,
                        name=health_check.name
                    ).set(status_value)
                    
                    return result
                    
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            return HealthResult(
                name=health_check.name,
                component_type=health_check.component_type,
                status=HealthStatus.CRITICAL,
                message="Health check timed out",
                response_time=response_time,
                timestamp=datetime.utcnow(),
                error="timeout"
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthResult(
                name=health_check.name,
                component_type=health_check.component_type,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                response_time=response_time,
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    def _process_health_result(self, result: HealthResult):
        """Process health check result"""
        # Store result
        self.health_results[result.name] = result
        
        # Add to history
        self.health_history.append(result)
        
        # Update component status
        self.component_status[result.component_type.value] = result.status
        
        # Check for alerts
        if result.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
            self._create_alert(result)
        
        # Update performance metrics
        self.performance_metrics["response_times"].append(result.response_time)
        if len(self.performance_metrics["response_times"]) > 1000:
            self.performance_metrics["response_times"] = self.performance_metrics["response_times"][-1000:]
        
        logger.info(f"Health check {result.name}: {result.status.value} - {result.message}")
    
    def _create_alert(self, result: HealthResult):
        """Create alert for health check result"""
        alert = {
            "id": f"{result.name}_{int(time.time())}",
            "component": result.name,
            "component_type": result.component_type.value,
            "severity": result.status.value,
            "message": result.message,
            "timestamp": result.timestamp.isoformat(),
            "response_time": result.response_time,
            "details": result.details
        }
        
        self.alert_history.append(alert)
        
        # Update Prometheus metrics
        self.prometheus_metrics["alert_count"].labels(severity=result.status.value).inc()
        
        logger.warning(f"Health alert: {result.name} - {result.message}")
    
    async def _update_system_metrics(self):
        """Update system-level metrics"""
        while self.monitoring_active:
            try:
                # Update uptime
                uptime = (datetime.utcnow() - self.start_time).total_seconds()
                self.prometheus_metrics["system_uptime"].set(uptime)
                
                # Update component availability
                for component_type in ComponentType:
                    component_results = [
                        r for r in self.health_results.values()
                        if r.component_type == component_type
                    ]
                    
                    if component_results:
                        healthy_count = sum(1 for r in component_results if r.status == HealthStatus.HEALTHY)
                        availability = (healthy_count / len(component_results)) * 100
                        self.prometheus_metrics["component_availability"].labels(
                            component=component_type.value
                        ).set(availability)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error updating system metrics: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_data(self):
        """Cleanup old health check data"""
        while self.monitoring_active:
            try:
                # Keep last 24 hours of health history
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.health_history = [
                    h for h in self.health_history
                    if h.timestamp > cutoff_time
                ]
                
                # Keep last 7 days of alerts
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                self.alert_history = [
                    a for a in self.alert_history
                    if datetime.fromisoformat(a["timestamp"]) > cutoff_time
                ]
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Error cleaning up old data: {e}")
                await asyncio.sleep(3600)
    
    def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health status"""
        # Determine overall status
        critical_components = [
            r for r in self.health_results.values()
            if r.status == HealthStatus.CRITICAL
        ]
        
        warning_components = [
            r for r in self.health_results.values()
            if r.status == HealthStatus.WARNING
        ]
        
        if critical_components:
            overall_status = HealthStatus.CRITICAL
        elif warning_components:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Calculate uptime
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Get active alerts
        active_alerts = [
            a for a in self.alert_history
            if (datetime.utcnow() - datetime.fromisoformat(a["timestamp"])).total_seconds() < 3600
        ]
        
        # Calculate metrics
        metrics = {
            "total_components": len(self.health_results),
            "healthy_components": len([r for r in self.health_results.values() if r.status == HealthStatus.HEALTHY]),
            "warning_components": len(warning_components),
            "critical_components": len(critical_components),
            "average_response_time": sum(r.response_time for r in self.health_results.values()) / len(self.health_results) if self.health_results else 0,
            "active_alerts": len(active_alerts),
            "system_load": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }
        
        return SystemHealth(
            overall_status=overall_status,
            timestamp=datetime.utcnow(),
            components=self.health_results,
            metrics=metrics,
            alerts=active_alerts,
            uptime=uptime,
            version=self.config.get("version", "3.0.0")
        )
    
    def get_component_health(self, component_name: str) -> Optional[HealthResult]:
        """Get health status for a specific component"""
        return self.health_results.get(component_name)
    
    def get_health_history(self, component_name: Optional[str] = None, limit: int = 100) -> List[HealthResult]:
        """Get health check history"""
        history = self.health_history
        
        if component_name:
            history = [h for h in history if h.name == component_name]
        
        return history[-limit:]
    
    def get_alerts(self, severity: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get alerts"""
        alerts = self.alert_history
        
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        
        return alerts[-limit:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        response_times = self.performance_metrics["response_times"]
        
        if not response_times:
            return {"status": "no_data"}
        
        return {
            "average_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)],
            "p99_response_time": sorted(response_times)[int(len(response_times) * 0.99)],
            "total_checks": len(response_times)
        }
    
    def get_availability_report(self) -> Dict[str, Any]:
        """Get availability report"""
        if not self.health_history:
            return {"status": "no_data"}
        
        # Group by component
        component_history = {}
        for result in self.health_history:
            if result.name not in component_history:
                component_history[result.name] = []
            component_history[result.name].append(result)
        
        # Calculate availability for each component
        availability_report = {}
        for component, results in component_history.items():
            if not results:
                continue
            
            # Calculate availability over last 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            recent_results = [r for r in results if r.timestamp > cutoff_time]
            
            if recent_results:
                healthy_count = sum(1 for r in recent_results if r.status == HealthStatus.HEALTHY)
                availability = (healthy_count / len(recent_results)) * 100
                
                availability_report[component] = {
                    "availability_percent": availability,
                    "total_checks": len(recent_results),
                    "healthy_checks": healthy_count,
                    "warning_checks": sum(1 for r in recent_results if r.status == HealthStatus.WARNING),
                    "critical_checks": sum(1 for r in recent_results if r.status == HealthStatus.CRITICAL),
                    "average_response_time": sum(r.response_time for r in recent_results) / len(recent_results)
                }
        
        return availability_report
    
    def enable_health_check(self, component_name: str):
        """Enable a health check"""
        for health_check in self.health_checks:
            if health_check.name == component_name:
                health_check.enabled = True
                logger.info(f"Enabled health check for {component_name}")
                break
    
    def disable_health_check(self, component_name: str):
        """Disable a health check"""
        for health_check in self.health_checks:
            if health_check.name == component_name:
                health_check.enabled = False
                logger.info(f"Disabled health check for {component_name}")
                break
    
    def update_health_check_interval(self, component_name: str, interval: int):
        """Update health check interval"""
        for health_check in self.health_checks:
            if health_check.name == component_name:
                health_check.interval = interval
                logger.info(f"Updated health check interval for {component_name} to {interval} seconds")
                break
    
    def add_custom_health_check(self, health_check: HealthCheck):
        """Add a custom health check"""
        self.health_checks.append(health_check)
        logger.info(f"Added custom health check: {health_check.name}")
    
    def remove_health_check(self, component_name: str):
        """Remove a health check"""
        self.health_checks = [h for h in self.health_checks if h.name != component_name]
        logger.info(f"Removed health check: {component_name}")
    
    def export_health_data(self) -> Dict[str, Any]:
        """Export health data for analysis"""
        return {
            "system_health": self.get_system_health().__dict__,
            "health_history": [h.__dict__ for h in self.health_history[-1000:]],
            "alerts": self.alert_history[-1000:],
            "performance_metrics": self.get_performance_metrics(),
            "availability_report": self.get_availability_report(),
            "export_timestamp": datetime.utcnow().isoformat()
        }

# Global health dashboard instance
health_dashboard = None

def get_health_dashboard() -> SystemHealthDashboard:
    """Get the global health dashboard instance"""
    global health_dashboard
    if health_dashboard is None:
        config = {
            "base_url": "http://localhost:8000",
            "version": "3.0.0"
        }
        health_dashboard = SystemHealthDashboard(config)
    return health_dashboard

if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "base_url": "http://localhost:8000",
            "version": "3.0.0"
        }
        
        dashboard = SystemHealthDashboard(config)
        
        # Run for 5 minutes
        await asyncio.sleep(300)
        
        # Get system health
        health = dashboard.get_system_health()
        print("System Health:")
        print(json.dumps(health.__dict__, indent=2, default=str))
        
        # Get performance metrics
        metrics = dashboard.get_performance_metrics()
        print("\nPerformance Metrics:")
        print(json.dumps(metrics, indent=2))
        
        # Get availability report
        availability = dashboard.get_availability_report()
        print("\nAvailability Report:")
        print(json.dumps(availability, indent=2))
        
        await dashboard.stop_monitoring()
    
    asyncio.run(main())













