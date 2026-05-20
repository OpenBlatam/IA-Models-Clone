"""
Health Check Endpoint
=====================
"""

from fastapi import APIRouter
from datetime import datetime
from ...config.app_config import get_config
from ...utils.health_checks import get_health_checker
from ...infrastructure.openrouter.openrouter_client import OpenRouterClient

router = APIRouter(prefix="/api/v1", tags=["health"])

config = get_config()


@router.get("/health")
async def health_check():
    """Health check endpoint básico"""
    return {
        "status": "healthy",
        "service": config.app_name,
        "version": config.app_version,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/health/detailed")
async def detailed_health_check():
    """Health check detallado con múltiples checks"""
    checker = get_health_checker()
    
    # Registrar checks si no están registrados
    if not checker.checks:
        # Check de OpenRouter
        async def check_openrouter():
            client = OpenRouterClient()
            health = await client.health_check()
            return health
        
        checker.register_simple("openrouter", check_openrouter, critical=True)
        
        # Check de configuración
        async def check_config():
            from ...utils.config_validator import validate_config
            try:
                validate_config()
                return {"valid": True}
            except Exception as e:
                return {"valid": False, "error": str(e)}
        
        checker.register_simple("config", check_config, critical=True)
    
    result = await checker.run_all()
    return result

