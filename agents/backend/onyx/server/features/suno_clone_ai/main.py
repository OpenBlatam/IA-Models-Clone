"""
Aplicación principal FastAPI optimizada con arquitectura modular

Configuraciones de rendimiento:
- Serialización JSON rápida con orjson
- Connection pooling
- Caching agresivo
- Async I/O para todas las operaciones
- Arquitectura modular con microservicios
"""

import logging

from config.settings import settings
from core.app_factory import create_application, register_routes, register_endpoints
from core.logging_config import configure_logging

configure_logging()

logger = logging.getLogger(__name__)

app = create_application()
register_routes(app)
register_endpoints(app)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=1 if settings.debug else 4,
        loop="uvloop",
        log_level="info"
    )
