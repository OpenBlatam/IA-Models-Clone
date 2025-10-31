"""
ML NLP Benchmark Monitor System
Real, working monitoring and metrics for ML NLP Benchmark system
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class MLNLPBenchmarkMonitor:
    """Advanced monitoring system for ML NLP Benchmark"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.start_time = time.time()
        
        # Metrics storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=history_size))
        self.current_metrics = {}
        self.alerts = []
        
        # Performance tracking
        self.request_times = deque(maxlen=history_size)
        self.error_counts = defaultdict(int)
        self.success_counts = defaultdict(int)
        
        # System monitoring
        self.system_metrics = {}
        self.last_system_check = 0
        
        # Threading
        self.lock = threading.Lock()
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Alert thresholds
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "response_time": 5.0,
            "error_rate": 5.0
        }
    
    def start_monitoring(self, interval: int = 30):
        """Start background monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Monitoring stopped")
    
    def _monitoring_loop(self, interval: int):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                self.collect_system_metrics()
                self.check_alerts()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def collect_system_metrics(self):
        """Collect system metrics"""
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
            disk_percent = (disk.used / disk.total) * 100
            disk_free = disk.free
            disk_total = disk.total
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            current_time = time.time()
            
            system_metrics = {
                "timestamp": current_time,
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count
                },
                "memory": {
                    "percent": memory_percent,
                    "available": memory_available,
                    "total": memory_total,
                    "used": memory_total - memory_available
                },
                "disk": {
                    "percent": disk_percent,
                    "free": disk_free,
                    "total": disk_total,
                    "used": disk_total - disk_free
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "process": {
                    "memory_rss": process_memory.rss,
                    "memory_vms": process_memory.vms,
                    "cpu_percent": process_cpu
                }
            }
            
            with self.lock:
                self.system_metrics = system_metrics
                self.last_system_check = current_time
                
                # Store in history
                self.metrics_history["cpu_percent"].append(cpu_percent)
                self.metrics_history["memory_percent"].append(memory_percent)
                self.metrics_history["disk_percent"].append(disk_percent)
                self.metrics_history["process_memory"].append(process_memory.rss)
                self.metrics_history["process_cpu"].append(process_cpu)
        
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def record_request(self, endpoint: str, method: str, processing_time: float, 
                      status_code: int, user_id: Optional[str] = None):
        """Record API request"""
        with self.lock:
            self.request_times.append(processing_time)
            
            if 200 <= status_code < 400:
                self.success_counts[endpoint] += 1
            else:
                self.error_counts[endpoint] += 1
            
            # Store in history
            self.metrics_history["request_time"].append(processing_time)
            self.metrics_history["request_count"].append(1)
    
    def record_error(self, error_type: str, endpoint: Optional[str] = None):
        """Record error"""
        with self.lock:
            self.error_counts[error_type] += 1
            self.metrics_history["error_count"].append(1)
    
    def record_analysis(self, analysis_type: str, processing_time: float, 
                       text_length: int, result_count: int):
        """Record analysis operation"""
        with self.lock:
            self.metrics_history["analysis_time"].append(processing_time)
            self.metrics_history["text_length"].append(text_length)
            self.metrics_history["result_count"].append(result_count)
    
    def check_alerts(self):
        """Check for alert conditions"""
        if not self.system_metrics:
            return
        
        current_time = time.time()
        alerts = []
        
        # CPU alert
        if self.system_metrics["cpu"]["percent"] > self.alert_thresholds["cpu_usage"]:
            alerts.append({
                "type": "cpu_usage",
                "level": "warning",
                "message": f"High CPU usage: {self.system_metrics['cpu']['percent']:.1f}%",
                "value": self.system_metrics["cpu"]["percent"],
                "threshold": self.alert_thresholds["cpu_usage"],
                "timestamp": current_time
            })
        
        # Memory alert
        if self.system_metrics["memory"]["percent"] > self.alert_thresholds["memory_usage"]:
            alerts.append({
                "type": "memory_usage",
                "level": "warning",
                "message": f"High memory usage: {self.system_metrics['memory']['percent']:.1f}%",
                "value": self.system_metrics["memory"]["percent"],
                "threshold": self.alert_thresholds["memory_usage"],
                "timestamp": current_time
            })
        
        # Disk alert
        if self.system_metrics["disk"]["percent"] > self.alert_thresholds["disk_usage"]:
            alerts.append({
                "type": "disk_usage",
                "level": "critical",
                "message": f"High disk usage: {self.system_metrics['disk']['percent']:.1f}%",
                "value": self.system_metrics["disk"]["percent"],
                "threshold": self.alert_thresholds["disk_usage"],
                "timestamp": current_time
            })
        
        # Response time alert
        if self.request_times:
            avg_response_time = sum(self.request_times) / len(self.request_times)
            if avg_response_time > self.alert_thresholds["response_time"]:
                alerts.append({
                    "type": "response_time",
                    "level": "warning",
                    "message": f"High response time: {avg_response_time:.2f}s",
                    "value": avg_response_time,
                    "threshold": self.alert_thresholds["response_time"],
                    "timestamp": current_time
                })
        
        # Error rate alert
        total_requests = sum(self.success_counts.values()) + sum(self.error_counts.values())
        if total_requests > 0:
            error_rate = (sum(self.error_counts.values()) / total_requests) * 100
            if error_rate > self.alert_thresholds["error_rate"]:
                alerts.append({
                    "type": "error_rate",
                    "level": "critical",
                    "message": f"High error rate: {error_rate:.1f}%",
                    "value": error_rate,
                    "threshold": self.alert_thresholds["error_rate"],
                    "timestamp": current_time
                })
        
        # Add new alerts
        with self.lock:
            self.alerts.extend(alerts)
            # Keep only last 100 alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        with self.lock:
            uptime = time.time() - self.start_time
            
            return {
                "uptime_seconds": uptime,
                "uptime_hours": uptime / 3600,
                "system_metrics": self.system_metrics,
                "last_system_check": self.last_system_check,
                "request_stats": {
                    "total_requests": len(self.request_times),
                    "success_count": sum(self.success_counts.values()),
                    "error_count": sum(self.error_counts.values()),
                    "average_response_time": sum(self.request_times) / len(self.request_times) if self.request_times else 0,
                    "min_response_time": min(self.request_times) if self.request_times else 0,
                    "max_response_time": max(self.request_times) if self.request_times else 0
                },
                "endpoint_stats": {
                    "success_by_endpoint": dict(self.success_counts),
                    "errors_by_endpoint": dict(self.error_counts)
                },
                "monitoring_active": self.monitoring_active
            }
    
    def get_metrics_history(self, metric_name: str, hours: int = 24) -> List[float]:
        """Get metrics history for a specific metric"""
        with self.lock:
            if metric_name not in self.metrics_history:
                return []
            
            # Filter by time range
            cutoff_time = time.time() - (hours * 3600)
            # For simplicity, return all data (in real implementation, filter by timestamp)
            return list(self.metrics_history[metric_name])
    
    def get_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alerts from the last N hours"""
        with self.lock:
            cutoff_time = time.time() - (hours * 3600)
            return [alert for alert in self.alerts if alert["timestamp"] > cutoff_time]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        with self.lock:
            if not self.request_times:
                return {"error": "No request data available"}
            
            response_times = list(self.request_times)
            
            return {
                "total_requests": len(response_times),
                "average_response_time": sum(response_times) / len(response_times),
                "median_response_time": sorted(response_times)[len(response_times) // 2],
                "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)],
                "p99_response_time": sorted(response_times)[int(len(response_times) * 0.99)],
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "throughput_per_second": len(response_times) / (time.time() - self.start_time),
                "success_rate": sum(self.success_counts.values()) / (sum(self.success_counts.values()) + sum(self.error_counts.values())) * 100 if (sum(self.success_counts.values()) + sum(self.error_counts.values())) > 0 else 0,
                "error_rate": sum(self.error_counts.values()) / (sum(self.success_counts.values()) + sum(self.error_counts.values())) * 100 if (sum(self.success_counts.values()) + sum(self.error_counts.values())) > 0 else 0
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        current_metrics = self.get_current_metrics()
        alerts = self.get_alerts(hours=1)  # Last hour
        
        # Determine health status
        critical_alerts = [alert for alert in alerts if alert["level"] == "critical"]
        warning_alerts = [alert for alert in alerts if alert["level"] == "warning"]
        
        if critical_alerts:
            health_status = "critical"
        elif warning_alerts:
            health_status = "warning"
        else:
            health_status = "healthy"
        
        return {
            "status": health_status,
            "timestamp": time.time(),
            "uptime_seconds": current_metrics["uptime_seconds"],
            "system_metrics": current_metrics["system_metrics"],
            "request_stats": current_metrics["request_stats"],
            "alerts": {
                "total": len(alerts),
                "critical": len(critical_alerts),
                "warning": len(warning_alerts)
            },
            "recent_alerts": alerts[-5:] if alerts else []
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        with self.lock:
            self.metrics_history.clear()
            self.request_times.clear()
            self.error_counts.clear()
            self.success_counts.clear()
            self.alerts.clear()
            self.start_time = time.time()
        
        logger.info("Metrics reset")
    
    def set_alert_threshold(self, metric: str, threshold: float):
        """Set alert threshold for a metric"""
        if metric in self.alert_thresholds:
            self.alert_thresholds[metric] = threshold
            logger.info(f"Alert threshold for {metric} set to {threshold}")
        else:
            logger.warning(f"Unknown metric: {metric}")

# Global monitor instance
ml_nlp_benchmark_monitor = MLNLPBenchmarkMonitor()

def get_monitor() -> MLNLPBenchmarkMonitor:
    """Get the global monitor instance"""
    return ml_nlp_benchmark_monitor

def start_monitoring(interval: int = 30):
    """Start monitoring"""
    ml_nlp_benchmark_monitor.start_monitoring(interval)

def stop_monitoring():
    """Stop monitoring"""
    ml_nlp_benchmark_monitor.stop_monitoring()

def record_request(endpoint: str, method: str, processing_time: float, 
                  status_code: int, user_id: Optional[str] = None):
    """Record API request"""
    ml_nlp_benchmark_monitor.record_request(endpoint, method, processing_time, status_code, user_id)

def record_error(error_type: str, endpoint: Optional[str] = None):
    """Record error"""
    ml_nlp_benchmark_monitor.record_error(error_type, endpoint)

def record_analysis(analysis_type: str, processing_time: float, 
                   text_length: int, result_count: int):
    """Record analysis operation"""
    ml_nlp_benchmark_monitor.record_analysis(analysis_type, processing_time, text_length, result_count)











