"""
BUL - Business Universal Language (Performance Optimizer)
========================================================

Advanced performance optimization and monitoring system for BUL.
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int
    response_time_avg: float
    requests_per_second: float
    error_rate: float
    cache_hit_rate: float
    timestamp: datetime

class PerformanceOptimizer:
    """Advanced performance optimization system."""
    
    def __init__(self):
        self.metrics_history: List[SystemMetrics] = []
        self.optimization_rules = {
            "cpu_threshold": 80.0,
            "memory_threshold": 85.0,
            "response_time_threshold": 2.0,
            "error_rate_threshold": 5.0,
            "cache_hit_threshold": 70.0
        }
        self.auto_optimizations = {
            "cache_cleanup": True,
            "connection_pooling": True,
            "query_optimization": True,
            "resource_scaling": True
        }
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start continuous system monitoring."""
        def monitor_loop():
            while True:
                try:
                    metrics = self.collect_metrics()
                    self.metrics_history.append(metrics)
                    
                    # Keep only last 1000 metrics
                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-1000:]
                    
                    # Check for optimization triggers
                    self.check_optimization_triggers(metrics)
                    
                    time.sleep(5)  # Monitor every 5 seconds
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(10)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # Application metrics (simulated)
            active_connections = len(psutil.net_connections())
            response_time_avg = self.calculate_avg_response_time()
            requests_per_second = self.calculate_requests_per_second()
            error_rate = self.calculate_error_rate()
            cache_hit_rate = self.calculate_cache_hit_rate()
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage=disk_usage,
                network_io=network_io,
                active_connections=active_connections,
                response_time_avg=response_time_avg,
                requests_per_second=requests_per_second,
                error_rate=error_rate,
                cache_hit_rate=cache_hit_rate,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return SystemMetrics(0, 0, 0, {}, 0, 0, 0, 0, 0, datetime.now())
    
    def calculate_avg_response_time(self) -> float:
        """Calculate average response time."""
        # Simulate response time calculation
        return 0.5 + (psutil.cpu_percent() / 100) * 2.0
    
    def calculate_requests_per_second(self) -> float:
        """Calculate requests per second."""
        # Simulate RPS calculation
        return 100 + (psutil.cpu_percent() / 10)
    
    def calculate_error_rate(self) -> float:
        """Calculate error rate percentage."""
        # Simulate error rate calculation
        return max(0, (psutil.cpu_percent() - 50) / 10)
    
    def calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        # Simulate cache hit rate calculation
        return max(50, 95 - (psutil.cpu_percent() / 5))
    
    def check_optimization_triggers(self, metrics: SystemMetrics):
        """Check if optimization triggers are met."""
        optimizations_applied = []
        
        # CPU optimization
        if metrics.cpu_percent > self.optimization_rules["cpu_threshold"]:
            optimizations_applied.append(self.optimize_cpu_usage())
        
        # Memory optimization
        if metrics.memory_percent > self.optimization_rules["memory_threshold"]:
            optimizations_applied.append(self.optimize_memory_usage())
        
        # Response time optimization
        if metrics.response_time_avg > self.optimization_rules["response_time_threshold"]:
            optimizations_applied.append(self.optimize_response_time())
        
        # Error rate optimization
        if metrics.error_rate > self.optimization_rules["error_rate_threshold"]:
            optimizations_applied.append(self.optimize_error_handling())
        
        # Cache optimization
        if metrics.cache_hit_rate < self.optimization_rules["cache_hit_threshold"]:
            optimizations_applied.append(self.optimize_cache())
        
        if optimizations_applied:
            logger.info(f"Applied optimizations: {optimizations_applied}")
    
    def optimize_cpu_usage(self) -> str:
        """Optimize CPU usage."""
        if self.auto_optimizations["query_optimization"]:
            # Simulate query optimization
            logger.info("Optimizing database queries")
            return "query_optimization"
        return "cpu_optimization"
    
    def optimize_memory_usage(self) -> str:
        """Optimize memory usage."""
        if self.auto_optimizations["cache_cleanup"]:
            # Simulate cache cleanup
            logger.info("Cleaning up cache")
            return "cache_cleanup"
        return "memory_optimization"
    
    def optimize_response_time(self) -> str:
        """Optimize response time."""
        if self.auto_optimizations["connection_pooling"]:
            # Simulate connection pooling optimization
            logger.info("Optimizing connection pooling")
            return "connection_pooling"
        return "response_time_optimization"
    
    def optimize_error_handling(self) -> str:
        """Optimize error handling."""
        logger.info("Improving error handling")
        return "error_handling_optimization"
    
    def optimize_cache(self) -> str:
        """Optimize cache performance."""
        logger.info("Optimizing cache strategy")
        return "cache_optimization"
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        latest = self.metrics_history[-1]
        avg_metrics = self.calculate_average_metrics()
        
        return {
            "current_metrics": {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "disk_usage": latest.disk_usage,
                "response_time_avg": latest.response_time_avg,
                "requests_per_second": latest.requests_per_second,
                "error_rate": latest.error_rate,
                "cache_hit_rate": latest.cache_hit_rate
            },
            "average_metrics": avg_metrics,
            "optimization_status": {
                "rules": self.optimization_rules,
                "auto_optimizations": self.auto_optimizations
            },
            "recommendations": self.generate_recommendations(latest),
            "timestamp": latest.timestamp.isoformat()
        }
    
    def calculate_average_metrics(self) -> Dict[str, float]:
        """Calculate average metrics over time."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        return {
            "avg_cpu_percent": sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            "avg_memory_percent": sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            "avg_response_time": sum(m.response_time_avg for m in recent_metrics) / len(recent_metrics),
            "avg_requests_per_second": sum(m.requests_per_second for m in recent_metrics) / len(recent_metrics),
            "avg_error_rate": sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
            "avg_cache_hit_rate": sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        }
    
    def generate_recommendations(self, metrics: SystemMetrics) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if metrics.cpu_percent > 70:
            recommendations.append("Consider scaling horizontally or optimizing CPU-intensive operations")
        
        if metrics.memory_percent > 80:
            recommendations.append("Consider increasing memory or optimizing memory usage")
        
        if metrics.response_time_avg > 1.5:
            recommendations.append("Consider implementing caching or optimizing database queries")
        
        if metrics.error_rate > 3:
            recommendations.append("Review error handling and implement better error recovery")
        
        if metrics.cache_hit_rate < 80:
            recommendations.append("Consider increasing cache size or improving cache strategy")
        
        if not recommendations:
            recommendations.append("System performance is optimal")
        
        return recommendations

# Initialize performance optimizer
performance_optimizer = PerformanceOptimizer()

# FastAPI app for performance monitoring
app = FastAPI(
    title="BUL Performance Optimizer",
    description="Advanced performance monitoring and optimization for BUL system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "BUL Performance Optimizer",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "Real-time Performance Monitoring",
            "Automatic Optimization",
            "Performance Analytics",
            "Resource Management",
            "Cache Optimization"
        ]
    }

@app.get("/metrics/performance")
async def get_performance_metrics():
    """Get current performance metrics."""
    return performance_optimizer.get_performance_report()

@app.get("/metrics/history")
async def get_metrics_history():
    """Get metrics history."""
    return {
        "metrics_count": len(performance_optimizer.metrics_history),
        "latest_metrics": performance_optimizer.metrics_history[-1].__dict__ if performance_optimizer.metrics_history else None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/optimize/manual")
async def trigger_manual_optimization():
    """Trigger manual optimization."""
    try:
        current_metrics = performance_optimizer.collect_metrics()
        performance_optimizer.check_optimization_triggers(current_metrics)
        return {
            "message": "Manual optimization triggered",
            "current_metrics": current_metrics.__dict__,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/optimize/rules")
async def get_optimization_rules():
    """Get current optimization rules."""
    return {
        "rules": performance_optimizer.optimization_rules,
        "auto_optimizations": performance_optimizer.auto_optimizations
    }

@app.post("/optimize/rules")
async def update_optimization_rules(rules: Dict[str, Any]):
    """Update optimization rules."""
    try:
        performance_optimizer.optimization_rules.update(rules)
        return {
            "message": "Optimization rules updated",
            "new_rules": performance_optimizer.optimization_rules
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
