"""
Monitoring Service
Advanced health checks and system monitoring
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import psutil
import os

logger = logging.getLogger(__name__)


class MonitoringService:
    """Advanced system monitoring"""
    
    def __init__(self):
        self.health_checks: Dict[str, callable] = {}
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks"""
        self.health_checks = {
            "disk_space": self._check_disk_space,
            "memory": self._check_memory,
            "cpu": self._check_cpu,
            "ffmpeg": self._check_ffmpeg,
            "api_keys": self._check_api_keys,
        }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space"""
        try:
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024 ** 3)
            total_gb = disk.total / (1024 ** 3)
            percent_free = (disk.free / disk.total) * 100
            
            status = "healthy" if percent_free > 20 else "warning" if percent_free > 10 else "critical"
            
            return {
                "status": status,
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "percent_free": round(percent_free, 2),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            used_gb = memory.used / (1024 ** 3)
            total_gb = memory.total / (1024 ** 3)
            percent_used = memory.percent
            
            status = "healthy" if percent_used < 80 else "warning" if percent_used < 90 else "critical"
            
            return {
                "status": status,
                "used_gb": round(used_gb, 2),
                "total_gb": round(total_gb, 2),
                "percent_used": round(percent_used, 2),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
    
    def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            status = "healthy" if cpu_percent < 80 else "warning" if cpu_percent < 90 else "critical"
            
            return {
                "status": status,
                "cpu_percent": round(cpu_percent, 2),
                "cpu_count": cpu_count,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
    
    def _check_ffmpeg(self) -> Dict[str, Any]:
        """Check FFmpeg availability"""
        import shutil
        
        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")
        
        if ffmpeg_path and ffprobe_path:
            return {
                "status": "healthy",
                "ffmpeg": ffmpeg_path,
                "ffprobe": ffprobe_path,
            }
        else:
            return {
                "status": "error",
                "error": "FFmpeg not found",
            }
    
    def _check_api_keys(self) -> Dict[str, Any]:
        """Check API keys configuration"""
        keys_status = {}
        
        openai_key = os.getenv("OPENAI_API_KEY")
        stability_key = os.getenv("STABILITY_AI_API_KEY")
        elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        
        keys_status["openai"] = "configured" if openai_key else "not_configured"
        keys_status["stability_ai"] = "configured" if stability_key else "not_configured"
        keys_status["elevenlabs"] = "configured" if elevenlabs_key else "not_configured"
        
        configured_count = sum(1 for v in keys_status.values() if v == "configured")
        total_count = len(keys_status)
        
        status = "healthy" if configured_count > 0 else "warning"
        
        return {
            "status": status,
            "configured": configured_count,
            "total": total_count,
            "keys": keys_status,
        }
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        overall_status = "healthy"
        
        for check_name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[check_name] = result
                
                if result.get("status") == "critical":
                    overall_status = "critical"
                elif result.get("status") == "warning" and overall_status == "healthy":
                    overall_status = "warning"
                elif result.get("status") == "error" and overall_status == "healthy":
                    overall_status = "error"
                    
            except Exception as e:
                results[check_name] = {
                    "status": "error",
                    "error": str(e),
                }
                if overall_status == "healthy":
                    overall_status = "error"
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": results,
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                },
                "memory": {
                    "total_gb": round(memory.total / (1024 ** 3), 2),
                    "used_gb": round(memory.used / (1024 ** 3), 2),
                    "percent": memory.percent,
                },
                "disk": {
                    "total_gb": round(disk.total / (1024 ** 3), 2),
                    "free_gb": round(disk.free / (1024 ** 3), 2),
                    "percent": round((disk.used / disk.total) * 100, 2),
                },
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }


_monitoring_service: Optional[MonitoringService] = None


def get_monitoring_service() -> MonitoringService:
    """Get monitoring service instance (singleton)"""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service

