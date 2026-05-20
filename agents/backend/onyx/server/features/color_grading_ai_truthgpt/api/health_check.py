"""
Health Check for Color Grading API
===================================

Comprehensive health check endpoints.
"""

import logging
from typing import Dict, Any
from pathlib import Path
import psutil

logger = logging.getLogger(__name__)


class HealthChecker:
    """Health check manager."""
    
    def __init__(self, agent=None):
        """
        Initialize health checker.
        
        Args:
            agent: Optional agent instance
        """
        self.agent = agent
    
    def check_system(self) -> Dict[str, Any]:
        """Check system health."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            
            return {
                "status": "healthy",
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
            }
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def check_agent(self) -> Dict[str, Any]:
        """Check agent health."""
        if not self.agent:
            return {
                "status": "unhealthy",
                "error": "Agent not initialized"
            }
        
        try:
            # Check services
            services_status = {}
            for name, service in self.agent.services.items():
                services_status[name] = "available" if service else "unavailable"
            
            return {
                "status": "healthy",
                "agent_initialized": True,
                "services": services_status,
                "output_dirs": {
                    name: str(path) for name, path in self.agent.output_dirs.items()
                }
            }
        except Exception as e:
            logger.error(f"Error checking agent health: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check external dependencies."""
        dependencies = {}
        
        # Check FFmpeg
        try:
            import subprocess
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                timeout=5
            )
            dependencies["ffmpeg"] = {
                "status": "available" if result.returncode == 0 else "unavailable",
                "version": result.stdout.decode().split("\n")[0] if result.returncode == 0 else None
            }
        except Exception as e:
            dependencies["ffmpeg"] = {
                "status": "unavailable",
                "error": str(e)
            }
        
        # Check OpenRouter (basic check)
        dependencies["openrouter"] = {
            "status": "configured" if self.agent and self.agent.config.openrouter.api_key else "not_configured"
        }
        
        return {
            "status": "healthy" if all(dep.get("status") in ["available", "configured"] for dep in dependencies.values()) else "degraded",
            "dependencies": dependencies
        }
    
    def get_full_health(self) -> Dict[str, Any]:
        """Get full health status."""
        system = self.check_system()
        agent = self.check_agent()
        dependencies = self.check_dependencies()
        
        overall_status = "healthy"
        if system["status"] != "healthy" or agent["status"] != "healthy":
            overall_status = "unhealthy"
        elif dependencies["status"] != "healthy":
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": None,  # Will be set by endpoint
            "system": system,
            "agent": agent,
            "dependencies": dependencies
        }




