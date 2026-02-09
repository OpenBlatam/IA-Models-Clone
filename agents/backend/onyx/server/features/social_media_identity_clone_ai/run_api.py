"""
Script para ejecutar la API
"""

import uvicorn
from .config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "social_media_identity_clone_ai.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level=settings.log_level.lower()
    )




