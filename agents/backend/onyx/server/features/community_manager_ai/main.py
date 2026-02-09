"""
Main Entry Point - Punto de Entrada Principal
===============================================

Script principal para ejecutar la aplicación.
"""

import uvicorn
import logging
from community_manager_ai.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Función principal"""
    settings = get_settings()
    
    logger.info(f"Iniciando {settings.app_name} v{settings.app_version}")
    logger.info(f"Servidor en http://{settings.api_host}:{settings.api_port}")
    
    uvicorn.run(
        "api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug"
    )


if __name__ == "__main__":
    main()

