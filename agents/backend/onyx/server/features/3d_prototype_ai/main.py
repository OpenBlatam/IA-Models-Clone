"""
3D Prototype AI - Main Entry Point
===================================

Punto de entrada principal para el generador de prototipos 3D.

Uso:
    python main.py
"""

import logging
import sys
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Función principal"""
    logger.info("🚀 Iniciando 3D Prototype AI...")
    logger.info("📡 Servidor disponible en http://0.0.0.0:8030")
    logger.info("🏥 Health check: http://0.0.0.0:8030/health")
    logger.info("📊 API Docs: http://0.0.0.0:8030/docs")
    
    # Importar la app desde el módulo API
    try:
        from .api.prototype_api import app
    except ImportError:
        from api.prototype_api import app
    
    # Ejecutar servidor
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8030,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()

