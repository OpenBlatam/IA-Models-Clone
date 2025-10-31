"""
Health Checker for BUL System
=============================

Comprehensive health monitoring and system status checks.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import httpx
import aiohttp

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    """Individual health check result"""
    name: str
    status: HealthStatus
    message: str
    response_time: float
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None

class HealthChecker:
    """
    Comprehensive health checker for BUL system components
    """
    
    def __init__(self):
        self.checks: List[HealthCheck] = []
        self.last_check_time: Optional[datetime] = None
        self.check_interval = 60  # seconds
        
    async def check_openrouter_api(self, api_key: str) -> HealthCheck:
        """Check OpenRouter API connectivity"""
        start_time = time.time()
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "openai/gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    return HealthCheck(
                        name="openrouter_api",
                        status=HealthStatus.HEALTHY,
                        message="OpenRouter API is responding",
                        response_time=response_time,
                        timestamp=datetime.now(),
                        details={"status_code": response.status_code}
                    )
                elif response.status_code == 401:
                    return HealthCheck(
                        name="openrouter_api",
                        status=HealthStatus.CRITICAL,
                        message="OpenRouter API authentication failed",
                        response_time=response_time,
                        timestamp=datetime.now(),
                        details={"status_code": response.status_code}
                    )
                else:
                    return HealthCheck(
                        name="openrouter_api",
                        status=HealthStatus.WARNING,
                        message=f"OpenRouter API returned status {response.status_code}",
                        response_time=response_time,
                        timestamp=datetime.now(),
                        details={"status_code": response.status_code}
                    )
                    
        except httpx.TimeoutException:
            return HealthCheck(
                name="openrouter_api",
                status=HealthStatus.CRITICAL,
                message="OpenRouter API timeout",
                response_time=time.time() - start_time,
                timestamp=datetime.now()
            )
        except Exception as e:
            return HealthCheck(
                name="openrouter_api",
                status=HealthStatus.CRITICAL,
                message=f"OpenRouter API error: {str(e)}",
                response_time=time.time() - start_time,
                timestamp=datetime.now()
            )
    
    async def check_openai_api(self, api_key: str) -> HealthCheck:
        """Check OpenAI API connectivity"""
        start_time = time.time()
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    return HealthCheck(
                        name="openai_api",
                        status=HealthStatus.HEALTHY,
                        message="OpenAI API is responding",
                        response_time=response_time,
                        timestamp=datetime.now(),
                        details={"status_code": response.status_code}
                    )
                elif response.status_code == 401:
                    return HealthCheck(
                        name="openai_api",
                        status=HealthStatus.WARNING,
                        message="OpenAI API authentication failed",
                        response_time=response_time,
                        timestamp=datetime.now(),
                        details={"status_code": response.status_code}
                    )
                else:
                    return HealthCheck(
                        name="openai_api",
                        status=HealthStatus.WARNING,
                        message=f"OpenAI API returned status {response.status_code}",
                        response_time=response_time,
                        timestamp=datetime.now(),
                        details={"status_code": response.status_code}
                    )
                    
        except Exception as e:
            return HealthCheck(
                name="openai_api",
                status=HealthStatus.WARNING,
                message=f"OpenAI API error: {str(e)}",
                response_time=time.time() - start_time,
                timestamp=datetime.now()
            )
    
    async def check_system_resources(self) -> HealthCheck:
        """Check system resource usage"""
        start_time = time.time()
        
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            response_time = time.time() - start_time
            
            # Determine status based on resource usage
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                status = HealthStatus.CRITICAL
                message = "High resource usage detected"
            elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 70:
                status = HealthStatus.WARNING
                message = "Moderate resource usage"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources are normal"
            
            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_free_gb": disk.free / (1024**3)
                }
            )
            
        except ImportError:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message="psutil not available for system monitoring",
                response_time=time.time() - start_time,
                timestamp=datetime.now()
            )
        except Exception as e:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.WARNING,
                message=f"System resource check failed: {str(e)}",
                response_time=time.time() - start_time,
                timestamp=datetime.now()
            )
    
    async def check_cache_health(self, cache_manager) -> HealthCheck:
        """Check cache system health"""
        start_time = time.time()
        
        try:
            stats = await cache_manager.get_stats()
            response_time = time.time() - start_time
            
            # Simple cache health check
            if stats['total_entries'] > stats['max_size'] * 0.9:
                status = HealthStatus.WARNING
                message = "Cache is nearly full"
            else:
                status = HealthStatus.HEALTHY
                message = "Cache is healthy"
            
            return HealthCheck(
                name="cache_system",
                status=status,
                message=message,
                response_time=response_time,
                timestamp=datetime.now(),
                details=stats
            )
            
        except Exception as e:
            return HealthCheck(
                name="cache_system",
                status=HealthStatus.WARNING,
                message=f"Cache health check failed: {str(e)}",
                response_time=time.time() - start_time,
                timestamp=datetime.now()
            )
    
    async def run_all_checks(self, openrouter_key: str, openai_key: Optional[str] = None, cache_manager=None) -> List[HealthCheck]:
        """Run all health checks"""
        checks = []
        
        # Run checks in parallel
        tasks = [
            self.check_openrouter_api(openrouter_key),
            self.check_system_resources()
        ]
        
        if openai_key:
            tasks.append(self.check_openai_api(openai_key))
        
        if cache_manager:
            tasks.append(self.check_cache_health(cache_manager))
        
        # Execute all checks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, HealthCheck):
                checks.append(result)
            else:
                # Handle exceptions
                checks.append(HealthCheck(
                    name="unknown",
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(result)}",
                    response_time=0.0,
                    timestamp=datetime.now()
                ))
        
        self.checks = checks
        self.last_check_time = datetime.now()
        
        return checks
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status"""
        if not self.checks:
            return HealthStatus.UNKNOWN
        
        # Determine overall status based on individual checks
        critical_count = sum(1 for check in self.checks if check.status == HealthStatus.CRITICAL)
        warning_count = sum(1 for check in self.checks if check.status == HealthStatus.WARNING)
        
        if critical_count > 0:
            return HealthStatus.CRITICAL
        elif warning_count > 0:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        overall_status = self.get_overall_status()
        
        return {
            "overall_status": overall_status.value,
            "last_check": self.last_check_time.isoformat() if self.last_check_time else None,
            "total_checks": len(self.checks),
            "healthy_checks": sum(1 for check in self.checks if check.status == HealthStatus.HEALTHY),
            "warning_checks": sum(1 for check in self.checks if check.status == HealthStatus.WARNING),
            "critical_checks": sum(1 for check in self.checks if check.status == HealthStatus.CRITICAL),
            "average_response_time": sum(check.response_time for check in self.checks) / len(self.checks) if self.checks else 0,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "response_time": check.response_time,
                    "timestamp": check.timestamp.isoformat(),
                    "details": check.details
                }
                for check in self.checks
            ]
        }

# Global health checker instance
_health_checker: Optional[HealthChecker] = None

def get_health_checker() -> HealthChecker:
    """Get the global health checker instance"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker




