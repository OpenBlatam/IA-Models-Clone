"""
Health check utilities for API endpoints
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class HealthChecker:
    """Health check manager"""
    
    def __init__(self):
        self.checks: Dict[str, callable] = {}
    
    def register_check(self, name: str, check_func: callable):
        """Register a health check"""
        self.checks[name] = check_func
    
    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            "status": "healthy",
            "checks": {}
        }
        
        for name, check_func in self.checks.items():
            try:
                if hasattr(check_func, '__call__'):
                    if hasattr(check_func, '__code__') and 'async' in str(check_func):
                        check_result = await check_func()
                    else:
                        check_result = check_func()
                else:
                    check_result = {"status": "unknown"}
                
                results["checks"][name] = {
                    "status": "healthy" if check_result.get("status") != "unhealthy" else "unhealthy",
                    "details": check_result
                }
                
                if check_result.get("status") == "unhealthy":
                    results["status"] = "unhealthy"
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results["checks"][name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                results["status"] = "unhealthy"
        
        return results
    
    async def check_spotify(self) -> Dict[str, Any]:
        """Check Spotify service health"""
        try:
            from ...config.service_registry import get_service
            spotify_service = get_service("spotify_service")
            spotify_service._get_access_token()
            return {"status": "healthy", "service": "spotify"}
        except Exception as e:
            return {"status": "unhealthy", "service": "spotify", "error": str(e)}
    
    async def check_services(self) -> Dict[str, Any]:
        """Check all registered services"""
        try:
            from ...config.service_registry import get_service
            services = [
                "spotify_service",
                "music_analyzer",
                "music_coach"
            ]
            
            results = {}
            for service_name in services:
                try:
                    service = get_service(service_name)
                    results[service_name] = "available"
                except Exception as e:
                    results[service_name] = f"unavailable: {str(e)}"
            
            return {"status": "healthy", "services": results}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


health_checker = HealthChecker()
health_checker.register_check("spotify", health_checker.check_spotify)
health_checker.register_check("services", health_checker.check_services)

