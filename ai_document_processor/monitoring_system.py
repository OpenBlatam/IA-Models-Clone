"""
Real-time Monitoring System
Real, working monitoring and metrics collection for AI document processing
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import psutil
import os

logger = logging.getLogger(__name__)

class RealTimeMonitoring:
    """Real-time monitoring system for AI document processing"""
    
    def __init__(self):
        self.metrics = {
            "system_metrics": {},
            "ai_metrics": {},
            "processing_metrics": {},
            "upload_metrics": {},
            "performance_metrics": {}
        }
        
        self.alerts = []
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "processing_time": 10.0,
            "error_rate": 10.0
        }
        
        self.start_time = time.time()
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect real-time system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            system_metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": cpu_count,
                    "frequency_mhz": cpu_freq.current if cpu_freq else None,
                    "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
                },
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "usage_percent": memory.percent,
                    "swap_total_gb": round(swap.total / (1024**3), 2),
                    "swap_used_gb": round(swap.used / (1024**3), 2)
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "usage_percent": round((disk.used / disk.total) * 100, 2)
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "process": {
                    "memory_mb": round(process_memory.rss / (1024**2), 2),
                    "cpu_percent": process_cpu,
                    "pid": process.pid,
                    "create_time": process.create_time()
                }
            }
            
            self.metrics["system_metrics"] = system_metrics
            return system_metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {"error": str(e)}
    
    async def collect_ai_metrics(self, basic_processor, advanced_processor) -> Dict[str, Any]:
        """Collect AI processing metrics"""
        try:
            basic_stats = basic_processor.get_stats()
            advanced_stats = advanced_processor.get_stats()
            
            ai_metrics = {
                "timestamp": datetime.now().isoformat(),
                "basic_processor": {
                    "total_requests": basic_stats["stats"]["total_requests"],
                    "successful_requests": basic_stats["stats"]["successful_requests"],
                    "failed_requests": basic_stats["stats"]["failed_requests"],
                    "success_rate": basic_stats["success_rate"],
                    "average_processing_time": basic_stats["stats"]["average_processing_time"],
                    "uptime_hours": basic_stats["uptime_hours"]
                },
                "advanced_processor": {
                    "total_requests": advanced_stats["stats"]["total_requests"],
                    "successful_requests": advanced_stats["stats"]["successful_requests"],
                    "failed_requests": advanced_stats["stats"]["failed_requests"],
                    "success_rate": advanced_stats["success_rate"],
                    "average_processing_time": advanced_stats["stats"]["average_processing_time"],
                    "cache_hit_rate": advanced_stats["cache_hit_rate"],
                    "uptime_hours": advanced_stats["uptime_hours"]
                },
                "combined": {
                    "total_requests": basic_stats["stats"]["total_requests"] + advanced_stats["stats"]["total_requests"],
                    "total_successful": basic_stats["stats"]["successful_requests"] + advanced_stats["stats"]["successful_requests"],
                    "total_failed": basic_stats["stats"]["failed_requests"] + advanced_stats["stats"]["failed_requests"],
                    "overall_success_rate": round(
                        (basic_stats["stats"]["successful_requests"] + advanced_stats["stats"]["successful_requests"]) / 
                        max(1, basic_stats["stats"]["total_requests"] + advanced_stats["stats"]["total_requests"]) * 100, 2
                    )
                }
            }
            
            self.metrics["ai_metrics"] = ai_metrics
            return ai_metrics
            
        except Exception as e:
            logger.error(f"Error collecting AI metrics: {e}")
            return {"error": str(e)}
    
    async def collect_upload_metrics(self, upload_processor) -> Dict[str, Any]:
        """Collect document upload metrics"""
        try:
            upload_stats = upload_processor.get_stats()
            
            upload_metrics = {
                "timestamp": datetime.now().isoformat(),
                "total_uploads": upload_stats["stats"]["total_uploads"],
                "successful_uploads": upload_stats["stats"]["successful_uploads"],
                "failed_uploads": upload_stats["stats"]["failed_uploads"],
                "success_rate": upload_stats["success_rate"],
                "average_processing_time": upload_stats["stats"]["average_processing_time"],
                "uptime_hours": upload_stats["uptime_hours"],
                "supported_formats": upload_stats["supported_formats"]
            }
            
            self.metrics["upload_metrics"] = upload_metrics
            return upload_metrics
            
        except Exception as e:
            logger.error(f"Error collecting upload metrics: {e}")
            return {"error": str(e)}
    
    async def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics"""
        try:
            uptime = time.time() - self.start_time
            
            performance_metrics = {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": round(uptime, 2),
                "uptime_hours": round(uptime / 3600, 2),
                "uptime_days": round(uptime / 86400, 2),
                "system_load": {
                    "cpu_usage": psutil.cpu_percent(),
                    "memory_usage": psutil.virtual_memory().percent,
                    "disk_usage": psutil.disk_usage('/').percent
                },
                "performance_score": self._calculate_performance_score()
            }
            
            self.metrics["performance_metrics"] = performance_metrics
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)"""
        try:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
            
            # Performance score based on resource usage
            # Lower usage = higher score
            cpu_score = max(0, 100 - cpu_usage)
            memory_score = max(0, 100 - memory_usage)
            disk_score = max(0, 100 - disk_usage)
            
            # Weighted average
            performance_score = (cpu_score * 0.4 + memory_score * 0.4 + disk_score * 0.2)
            return round(performance_score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.0
    
    async def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        current_time = datetime.now().isoformat()
        
        try:
            # System alerts
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
            
            if cpu_usage > self.alert_thresholds["cpu_usage"]:
                alerts.append({
                    "type": "warning",
                    "message": f"High CPU usage: {cpu_usage}%",
                    "timestamp": current_time,
                    "threshold": self.alert_thresholds["cpu_usage"]
                })
            
            if memory_usage > self.alert_thresholds["memory_usage"]:
                alerts.append({
                    "type": "warning",
                    "message": f"High memory usage: {memory_usage}%",
                    "timestamp": current_time,
                    "threshold": self.alert_thresholds["memory_usage"]
                })
            
            if disk_usage > self.alert_thresholds["disk_usage"]:
                alerts.append({
                    "type": "critical",
                    "message": f"High disk usage: {disk_usage}%",
                    "timestamp": current_time,
                    "threshold": self.alert_thresholds["disk_usage"]
                })
            
            # Store alerts
            self.alerts.extend(alerts)
            
            # Keep only last 100 alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
            return []
    
    async def get_comprehensive_metrics(self, basic_processor, advanced_processor, upload_processor) -> Dict[str, Any]:
        """Get comprehensive monitoring metrics"""
        try:
            # Collect all metrics
            system_metrics = await self.collect_system_metrics()
            ai_metrics = await self.collect_ai_metrics(basic_processor, advanced_processor)
            upload_metrics = await self.collect_upload_metrics(upload_processor)
            performance_metrics = await self.collect_performance_metrics()
            alerts = await self.check_alerts()
            
            comprehensive_metrics = {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": system_metrics,
                "ai_metrics": ai_metrics,
                "upload_metrics": upload_metrics,
                "performance_metrics": performance_metrics,
                "alerts": alerts,
                "alert_count": len(alerts),
                "overall_health": self._calculate_overall_health()
            }
            
            return comprehensive_metrics
            
        except Exception as e:
            logger.error(f"Error getting comprehensive metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall system health"""
        try:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
            
            # Determine health based on resource usage
            if cpu_usage > 90 or memory_usage > 95 or disk_usage > 95:
                return "critical"
            elif cpu_usage > 80 or memory_usage > 85 or disk_usage > 90:
                return "warning"
            elif cpu_usage > 70 or memory_usage > 75 or disk_usage > 80:
                return "degraded"
            else:
                return "healthy"
                
        except Exception as e:
            logger.error(f"Error calculating overall health: {e}")
            return "unknown"
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            "metrics": self.metrics,
            "alerts": self.alerts[-10:],  # Last 10 alerts
            "alert_thresholds": self.alert_thresholds,
            "uptime": time.time() - self.start_time
        }

# Global instance
monitoring_system = RealTimeMonitoring()













