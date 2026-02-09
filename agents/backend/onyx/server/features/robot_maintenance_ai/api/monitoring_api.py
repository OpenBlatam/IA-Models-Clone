"""
Advanced Monitoring API for system health and performance tracking.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends, Query
from typing import Dict, Any
from datetime import datetime, timedelta
import psutil
import os

from .base_router import BaseRouter
from ..utils.metrics import metrics_collector
from ...utils.file_helpers import get_iso_timestamp
from ..utils.data_helpers import round_decimal, get_nested_value, ensure_minimum, count_matching, count_matching

# Create base router instance
base = BaseRouter(
    prefix="/api/monitoring",
    tags=["Monitoring"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router


@router.get("/health/detailed")
@base.timed_endpoint("get_detailed_health")
async def get_detailed_health(
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get detailed system health information.
    """
    base.log_request("get_detailed_health")
    
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Process metrics
    process = psutil.Process(os.getpid())
    process_memory = process.memory_info()
    
    # API metrics
    api_metrics = metrics_collector.get_stats()
    
    health = {
        "timestamp": get_iso_timestamp(),
        "status": "healthy",
        "system": {
            "cpu": {
                "percent": cpu_percent,
                "cores": psutil.cpu_count(),
                "status": "ok" if cpu_percent < 80 else "warning"
            },
            "memory": {
                "total_gb": round_decimal(memory.total / (1024**3)),
                "available_gb": round_decimal(memory.available / (1024**3)),
                "used_percent": memory.percent,
                "status": "ok" if memory.percent < 80 else "warning"
            },
            "disk": {
                "total_gb": round_decimal(disk.total / (1024**3)),
                "free_gb": round_decimal(disk.free / (1024**3)),
                "used_percent": round_decimal(disk.used / disk.total * 100),
                "status": "ok" if disk.used / disk.total < 80 else "warning"
            }
        },
        "application": {
            "memory_mb": round_decimal(process_memory.rss / (1024**2)),
            "cpu_percent": process.cpu_percent(),
            "threads": process.num_threads(),
            "uptime_seconds": (datetime.now() - datetime.fromtimestamp(process.create_time())).total_seconds()
        },
        "api": {
            "total_requests": api_metrics.get("total_requests", 0),
            "uptime_seconds": api_metrics.get("uptime_seconds", "0"),
            "cache_hit_rate": get_nested_value(api_metrics, "cache_stats", "hit_rate", default="0%")
        }
    }
    
    # Determine overall status
    if cpu_percent > 90 or memory.percent > 90 or disk.used / disk.total > 90:
        health["status"] = "critical"
    elif cpu_percent > 80 or memory.percent > 80 or disk.used / disk.total > 80:
        health["status"] = "warning"
    
    return base.success(health)


@router.get("/performance/history")
@base.timed_endpoint("get_performance_history")
async def get_performance_history(
    hours: int = Query(24, ge=1, le=168, description="Number of hours"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get performance history over time.
    """
    base.log_request("get_performance_history", hours=hours)
    
    # In a real implementation, this would query historical metrics
    # For now, return current metrics with timestamp
    
    api_metrics = metrics_collector.get_stats()
    
    # Simulate history (in production, this would be from time-series DB)
    history = []
    for i in range(hours):
        timestamp = (datetime.now() - timedelta(hours=i)).isoformat()
        history.append({
            "timestamp": timestamp,
            "requests": api_metrics.get("total_requests", 0) // int(ensure_minimum(hours)),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        })
    
    return base.success({
        "period_hours": hours,
        "history": list(reversed(history)),
        "summary": {
            "avg_cpu": round_decimal(sum(h["cpu_percent"] for h in history) / len(history)) if history else 0,
            "avg_memory": round_decimal(sum(h["memory_percent"] for h in history) / len(history)) if history else 0,
            "total_requests": sum(h["requests"] for h in history)
        }
    })


@router.get("/alerts/system")
@base.timed_endpoint("get_system_alerts")
async def get_system_alerts(
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get system-level alerts based on monitoring.
    """
    base.log_request("get_system_alerts")
    
    alerts = []
    
    # Check CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    if cpu_percent > 90:
        alerts.append({
            "level": "critical",
            "component": "cpu",
            "message": f"CPU usage is critically high: {cpu_percent}%",
            "recommendation": "Consider scaling or optimizing workload"
        })
    elif cpu_percent > 80:
        alerts.append({
            "level": "warning",
            "component": "cpu",
            "message": f"CPU usage is high: {cpu_percent}%",
            "recommendation": "Monitor CPU usage closely"
        })
    
    # Check Memory
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        alerts.append({
            "level": "critical",
            "component": "memory",
            "message": f"Memory usage is critically high: {memory.percent}%",
            "recommendation": "Consider increasing memory or optimizing application"
        })
    elif memory.percent > 80:
        alerts.append({
            "level": "warning",
            "component": "memory",
            "message": f"Memory usage is high: {memory.percent}%",
            "recommendation": "Monitor memory usage"
        })
    
    # Check Disk
    disk = psutil.disk_usage('/')
    disk_percent = disk.used / disk.total * 100
    if disk_percent > 90:
        alerts.append({
            "level": "critical",
            "component": "disk",
            "message": f"Disk usage is critically high: {disk_percent:.1f}%",
            "recommendation": "Free up disk space immediately"
        })
    elif disk_percent > 80:
        alerts.append({
            "level": "warning",
            "component": "disk",
            "message": f"Disk usage is high: {disk_percent:.1f}%",
            "recommendation": "Consider cleaning up old files"
        })
    
    return base.success({
        "alerts": alerts,
        "total": len(alerts),
        "critical": count_matching(alerts, lambda a: a["level"] == "critical"),
        "warnings": count_matching(alerts, lambda a: a["level"] == "warning")
    })


@router.get("/resources/usage")
@base.timed_endpoint("get_resource_usage")
async def get_resource_usage(
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get current resource usage statistics.
    """
    base.log_request("get_resource_usage")
    
    process = psutil.Process(os.getpid())
    
    usage = {
        "timestamp": get_iso_timestamp(),
        "process": {
            "pid": process.pid,
            "memory_mb": round_decimal(process.memory_info().rss / (1024**2)),
            "cpu_percent": process.cpu_percent(interval=0.1),
            "threads": process.num_threads(),
            "open_files": len(process.open_files()),
            "connections": len(process.connections())
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": round_decimal(psutil.disk_usage('/').used / psutil.disk_usage('/').total * 100)
        }
    }
    
    return base.success(usage)




