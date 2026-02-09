"""
Physical Store Designer AI - Main Entry Point
==============================================

Punto de entrada principal para el diseñador de locales físicos con IA.

Uso:
    python main.py
"""

"""
Physical Store Designer AI - Main Entry Point

Punto de entrada principal para el diseñador de locales físicos con IA.
"""

import uvicorn
from .core.logging_config import setup_logging, get_logger
from .config.settings import settings

# Setup logging
setup_logging()
logger = get_logger(__name__)


def main() -> None:
    """
    Función principal - Inicia el servidor FastAPI
    
    Configura logging, carga la aplicación y ejecuta el servidor uvicorn.
    """
    logger.info("🏪 Iniciando Physical Store Designer AI...")
    
    from .api.main import app
    
    logger.info("✅ Aplicación creada exitosamente")
    logger.info(f"📡 Servidor disponible en http://{settings.host}:{settings.port}")
    logger.info(f"🏥 Health check: http://{settings.host}:{settings.port}/health")
    logger.info(f"📊 Documentación: http://{settings.host}:{settings.port}/docs")
    logger.info(f"🔧 Configuración: log_level={settings.log_level}, log_format={settings.log_format}")
    
    # Ejecutar servidor
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()




