"""
Performance Monitoring and Analytics System
==========================================

Advanced performance monitoring, metrics collection, and system analytics
for the AI Document Classifier.
"""

import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import threading
from collections import deque, defaultdict
import asyncio
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    active_connections: int
    timestamp: datetime

@dataclass
class ClassificationMetrics:
    """Classification performance metrics"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_processing_time: float
    avg_confidence: float
    document_type_distribution: Dict[str, int]
    method_performance: Dict[str, Dict[str, float]]
    error_distribution: Dict[str, int]
    timestamp: datetime

class PerformanceMonitor:
    """
    Advanced performance monitoring system
    """
    
    def __init__(self, db_path: Optional[str] = None, retention_days: int = 30):
        """
        Initialize performance monitor
        
        Args:
            db_path: Path to metrics database
            retention_days: Number of days to retain metrics
        """
        self.db_path = Path(db_path) if db_path else Path(__file__).parent.parent / "data" / "metrics.db"
        self.db_path.parent.mkdir(exist_ok=True)
        self.retention_days = retention_days
        
        # Metrics storage
        self.metrics_buffer = deque(maxlen=10000)
        self.system_metrics = deque(maxlen=1000)
        self.classification_metrics = deque(maxlen=1000)
        
        # Performance tracking
        self.request_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.method_performance = defaultdict(lambda: {"total": 0, "success": 0, "errors": 0, "total_time": 0.0})
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread = None
        self.collection_interval = 30  # seconds
        
        # Initialize database
        self._init_database()
        
        # Start background monitoring
        self.start_monitoring()
    
    def _init_database(self):
        """Initialize metrics database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    tags TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cpu_percent REAL NOT NULL,
                    memory_percent REAL NOT NULL,
                    memory_used_mb REAL NOT NULL,
                    disk_usage_percent REAL NOT NULL,
                    network_io TEXT NOT NULL,
                    active_connections INTEGER NOT NULL,
                    timestamp TIMESTAMP NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS classification_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_requests INTEGER NOT NULL,
                    successful_requests INTEGER NOT NULL,
                    failed_requests INTEGER NOT NULL,
                    avg_processing_time REAL NOT NULL,
                    avg_confidence REAL NOT NULL,
                    document_type_distribution TEXT NOT NULL,
                    method_performance TEXT NOT NULL,
                    error_distribution TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_name ON performance_metrics(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_classification_timestamp ON classification_metrics(timestamp)")
    
    def start_monitoring(self):
        """Start background monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics.append(system_metrics)
                
                # Collect classification metrics
                classification_metrics = self._collect_classification_metrics()
                self.classification_metrics.append(classification_metrics)
                
                # Store metrics in database
                self._store_metrics_batch()
                
                # Clean old metrics
                self._cleanup_old_metrics()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # Active connections (approximate)
            try:
                connections = len(psutil.net_connections())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                connections = 0
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                disk_usage_percent=disk_usage_percent,
                network_io=network_io,
                active_connections=connections,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                disk_usage_percent=0.0,
                network_io={},
                active_connections=0,
                timestamp=datetime.now()
            )
    
    def _collect_classification_metrics(self) -> ClassificationMetrics:
        """Collect classification performance metrics"""
        try:
            # Calculate totals
            total_requests = sum(method["total"] for method in self.method_performance.values())
            successful_requests = sum(method["success"] for method in self.method_performance.values())
            failed_requests = sum(method["errors"] for method in self.method_performance.values())
            
            # Calculate averages
            avg_processing_time = 0.0
            if total_requests > 0:
                total_time = sum(method["total_time"] for method in self.method_performance.values())
                avg_processing_time = total_time / total_requests
            
            # Document type distribution (would need to be tracked separately)
            document_type_distribution = {}
            
            # Method performance
            method_performance = {}
            for method, stats in self.method_performance.items():
                if stats["total"] > 0:
                    method_performance[method] = {
                        "total_requests": stats["total"],
                        "success_rate": stats["success"] / stats["total"],
                        "error_rate": stats["errors"] / stats["total"],
                        "avg_processing_time": stats["total_time"] / stats["total"]
                    }
            
            # Error distribution
            error_distribution = dict(self.error_counts)
            
            return ClassificationMetrics(
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                avg_processing_time=avg_processing_time,
                avg_confidence=0.0,  # Would need to be tracked separately
                document_type_distribution=document_type_distribution,
                method_performance=method_performance,
                error_distribution=error_distribution,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error collecting classification metrics: {e}")
            return ClassificationMetrics(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_processing_time=0.0,
                avg_confidence=0.0,
                document_type_distribution={},
                method_performance={},
                error_distribution={},
                timestamp=datetime.now()
            )
    
    def record_classification_request(
        self, 
        method: str, 
        processing_time: float, 
        success: bool, 
        error: Optional[str] = None,
        document_type: Optional[str] = None,
        confidence: Optional[float] = None
    ):
        """
        Record a classification request
        
        Args:
            method: Classification method used
            processing_time: Time taken to process
            success: Whether the request was successful
            error: Error message if failed
            document_type: Detected document type
            confidence: Classification confidence
        """
        # Update method performance
        self.method_performance[method]["total"] += 1
        self.method_performance[method]["total_time"] += processing_time
        
        if success:
            self.method_performance[method]["success"] += 1
        else:
            self.method_performance[method]["errors"] += 1
            if error:
                self.error_counts[error] += 1
        
        # Record request time
        self.request_times[method].append(processing_time)
        
        # Keep only recent request times
        if len(self.request_times[method]) > 1000:
            self.request_times[method] = self.request_times[method][-500:]
        
        # Record metrics
        self.record_metric("classification_processing_time", processing_time, {
            "method": method,
            "success": str(success)
        })
        
        if confidence is not None:
            self.record_metric("classification_confidence", confidence, {
                "method": method,
                "document_type": document_type or "unknown"
            })
    
    def record_metric(
        self, 
        name: str, 
        value: float, 
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a custom metric
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
            metadata: Optional metadata
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self.metrics_buffer.append(metric)
    
    def _store_metrics_batch(self):
        """Store metrics batch to database"""
        if not self.metrics_buffer:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Store performance metrics
                metrics_to_store = list(self.metrics_buffer)
                self.metrics_buffer.clear()
                
                for metric in metrics_to_store:
                    conn.execute("""
                        INSERT INTO performance_metrics (name, value, timestamp, tags, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        metric.name,
                        metric.value,
                        metric.timestamp.isoformat(),
                        json.dumps(metric.tags),
                        json.dumps(metric.metadata)
                    ))
                
                # Store system metrics
                if self.system_metrics:
                    system_metric = self.system_metrics[-1]
                    conn.execute("""
                        INSERT INTO system_metrics 
                        (cpu_percent, memory_percent, memory_used_mb, disk_usage_percent, 
                         network_io, active_connections, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        system_metric.cpu_percent,
                        system_metric.memory_percent,
                        system_metric.memory_used_mb,
                        system_metric.disk_usage_percent,
                        json.dumps(system_metric.network_io),
                        system_metric.active_connections,
                        system_metric.timestamp.isoformat()
                    ))
                
                # Store classification metrics
                if self.classification_metrics:
                    classification_metric = self.classification_metrics[-1]
                    conn.execute("""
                        INSERT INTO classification_metrics 
                        (total_requests, successful_requests, failed_requests, avg_processing_time,
                         avg_confidence, document_type_distribution, method_performance, 
                         error_distribution, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        classification_metric.total_requests,
                        classification_metric.successful_requests,
                        classification_metric.failed_requests,
                        classification_metric.avg_processing_time,
                        classification_metric.avg_confidence,
                        json.dumps(classification_metric.document_type_distribution),
                        json.dumps(classification_metric.method_performance),
                        json.dumps(classification_metric.error_distribution),
                        classification_metric.timestamp.isoformat()
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics based on retention policy"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            with sqlite3.connect(self.db_path) as conn:
                # Clean performance metrics
                conn.execute("""
                    DELETE FROM performance_metrics 
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                
                # Clean system metrics
                conn.execute("""
                    DELETE FROM system_metrics 
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                
                # Clean classification metrics
                conn.execute("""
                    DELETE FROM classification_metrics 
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}")
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance summary for the last N hours
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Performance summary
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get recent metrics
                cursor = conn.execute("""
                    SELECT name, AVG(value) as avg_value, MIN(value) as min_value, 
                           MAX(value) as max_value, COUNT(*) as count
                    FROM performance_metrics 
                    WHERE timestamp >= ?
                    GROUP BY name
                """, (cutoff_time.isoformat(),))
                
                metrics_summary = {}
                for row in cursor.fetchall():
                    metrics_summary[row[0]] = {
                        "average": row[1],
                        "minimum": row[2],
                        "maximum": row[3],
                        "count": row[4]
                    }
                
                # Get system metrics
                cursor = conn.execute("""
                    SELECT AVG(cpu_percent), AVG(memory_percent), AVG(memory_used_mb),
                           AVG(disk_usage_percent), AVG(active_connections)
                    FROM system_metrics 
                    WHERE timestamp >= ?
                """, (cutoff_time.isoformat(),))
                
                system_row = cursor.fetchone()
                system_summary = {
                    "avg_cpu_percent": system_row[0] if system_row[0] else 0,
                    "avg_memory_percent": system_row[1] if system_row[1] else 0,
                    "avg_memory_used_mb": system_row[2] if system_row[2] else 0,
                    "avg_disk_usage_percent": system_row[3] if system_row[3] else 0,
                    "avg_active_connections": system_row[4] if system_row[4] else 0
                }
                
                # Get classification metrics
                cursor = conn.execute("""
                    SELECT total_requests, successful_requests, failed_requests,
                           avg_processing_time, avg_confidence
                    FROM classification_metrics 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (cutoff_time.isoformat(),))
                
                classification_row = cursor.fetchone()
                classification_summary = {
                    "total_requests": classification_row[0] if classification_row[0] else 0,
                    "successful_requests": classification_row[1] if classification_row[1] else 0,
                    "failed_requests": classification_row[2] if classification_row[2] else 0,
                    "avg_processing_time": classification_row[3] if classification_row[3] else 0,
                    "avg_confidence": classification_row[4] if classification_row[4] else 0
                }
                
                return {
                    "time_range_hours": hours,
                    "metrics": metrics_summary,
                    "system": system_summary,
                    "classification": classification_summary,
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status"""
        try:
            # Get current system metrics
            current_system = self._collect_system_metrics()
            
            # Get recent performance
            recent_metrics = self.get_performance_summary(hours=1)
            
            # Determine health status
            health_status = "healthy"
            issues = []
            
            # Check CPU usage
            if current_system.cpu_percent > 80:
                health_status = "degraded"
                issues.append(f"High CPU usage: {current_system.cpu_percent:.1f}%")
            
            # Check memory usage
            if current_system.memory_percent > 85:
                health_status = "degraded"
                issues.append(f"High memory usage: {current_system.memory_percent:.1f}%")
            
            # Check disk usage
            if current_system.disk_usage_percent > 90:
                health_status = "critical"
                issues.append(f"High disk usage: {current_system.disk_usage_percent:.1f}%")
            
            # Check error rate
            if recent_metrics.get("classification", {}).get("failed_requests", 0) > 0:
                total_requests = recent_metrics.get("classification", {}).get("total_requests", 1)
                error_rate = recent_metrics["classification"]["failed_requests"] / total_requests
                if error_rate > 0.1:  # 10% error rate
                    health_status = "degraded"
                    issues.append(f"High error rate: {error_rate:.1%}")
            
            return {
                "status": health_status,
                "issues": issues,
                "system_metrics": {
                    "cpu_percent": current_system.cpu_percent,
                    "memory_percent": current_system.memory_percent,
                    "memory_used_mb": current_system.memory_used_mb,
                    "disk_usage_percent": current_system.disk_usage_percent,
                    "active_connections": current_system.active_connections
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                "status": "unknown",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def export_metrics(self, format: str = "json", hours: int = 24) -> str:
        """
        Export metrics in specified format
        
        Args:
            format: Export format (json, csv)
            hours: Number of hours to export
            
        Returns:
            Exported metrics
        """
        try:
            summary = self.get_performance_summary(hours)
            
            if format == "json":
                return json.dumps(summary, indent=2)
            elif format == "csv":
                # Convert to CSV format
                csv_lines = ["metric,value"]
                for category, data in summary.items():
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, (int, float, str)):
                                csv_lines.append(f"{category}.{key},{value}")
                return "\n".join(csv_lines)
            else:
                return str(summary)
                
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return f"Error: {e}"

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# Decorator for automatic performance tracking
def track_performance(method_name: str = None):
    """Decorator to automatically track method performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                processing_time = time.time() - start_time
                method = method_name or func.__name__
                performance_monitor.record_classification_request(
                    method=method,
                    processing_time=processing_time,
                    success=success,
                    error=error
                )
        
        return wrapper
    return decorator

# Example usage
if __name__ == "__main__":
    # Initialize monitor
    monitor = PerformanceMonitor()
    
    # Record some test metrics
    monitor.record_metric("test_metric", 42.0, {"test": "true"})
    monitor.record_classification_request("test_method", 0.5, True)
    
    # Get performance summary
    summary = monitor.get_performance_summary(hours=1)
    print("Performance Summary:")
    print(json.dumps(summary, indent=2))
    
    # Get health status
    health = monitor.get_health_status()
    print("\nHealth Status:")
    print(json.dumps(health, indent=2))
    
    # Stop monitoring
    monitor.stop_monitoring()



























