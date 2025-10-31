"""
Health Check Utilities
Comprehensive health checks for services and dependencies
"""

import time
import asyncio
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime


class HealthCheck:
    """Comprehensive health check manager"""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.status_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 5  # Cache health checks for 5 seconds
    
    def register_check(self, name: str, check_func: Callable, async_check: bool = False):
        """Register a health check"""
        self.checks[name] = {
            "func": check_func,
            "async": async_check
        }
    
    async def check_service(self, name: str) -> Dict[str, Any]:
        """Check health of a specific service"""
        if name not in self.checks:
            return {
                "status": "unknown",
                "message": f"Health check '{name}' not registered",
                "timestamp": time.time()
            }
        
        # Check cache first
        cached = self.status_cache.get(name)
        if cached and (time.time() - cached.get("checked_at", 0)) < self.cache_ttl:
            return cached["result"]
        
        check_info = self.checks[name]
        check_func = check_info["func"]
        is_async = check_info["async"]
        
        start_time = time.time()
        
        try:
            if is_async:
                result = await check_func()
            else:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
            
            duration = time.time() - start_time
            
            health_result = {
                "status": "healthy" if result else "unhealthy",
                "duration_seconds": round(duration, 3),
                "timestamp": time.time()
            }
            
        except Exception as e:
            duration = time.time() - start_time
            health_result = {
                "status": "unhealthy",
                "error": str(e),
                "duration_seconds": round(duration, 3),
                "timestamp": time.time()
            }
        
        # Cache result
        self.status_cache[name] = {
            "result": health_result,
            "checked_at": time.time()
        }
        
        return health_result
    
    async def check_all(self) -> Dict[str, Any]:
        """Check health of all registered services"""
        results = {}
        
        for name in self.checks.keys():
            results[name] = await self.check_service(name)
        
        # Overall status
        all_healthy = all(
            result.get("status") == "healthy" 
            for result in results.values()
        )
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "services": results,
            "timestamp": time.time(),
            "uptime_seconds": time.time() - (self.start_time if hasattr(self, 'start_time') else time.time())
        }
    
    def start(self):
        """Mark health check system as started"""
        self.start_time = time.time()


# Global health check instance
health_check = HealthCheck()


async def check_webhook_health() -> bool:
    """Check webhook system health"""
    try:
        from ..webhooks import webhook_manager
        # Simple check - verify manager exists
        return webhook_manager is not None
    except Exception:
        return False


async def check_cache_health() -> bool:
    """Check cache system health"""
    try:
        from ..cache import get_cache_stats
        stats = get_cache_stats()
        return stats is not None
    except Exception:
        return False


async def check_database_health() -> bool:
    """Check database health (if available)"""
    try:
        # This would check actual database connection
        # For now, just return True if no database errors
        return True
    except Exception:
        return False


async def check_ai_ml_health() -> bool:
    """Check AI/ML service health"""
    try:
        from ..ai_ml_enhanced import ai_ml_engine
        return hasattr(ai_ml_engine, 'initialized') and ai_ml_engine.initialized
    except Exception:
        return False


def setup_default_health_checks():
    """Setup default health checks"""
    health_check.register_check("webhook", check_webhook_health, async_check=True)
    health_check.register_check("cache", check_cache_health, async_check=True)
    health_check.register_check("database", check_database_health, async_check=True)
    health_check.register_check("ai_ml", check_ai_ml_health, async_check=True)
    health_check.start()






