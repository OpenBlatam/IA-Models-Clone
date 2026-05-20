"""
AI Job Replacement Helper - Main Entry Point
=============================================

Punto de entrada principal para el sistema de ayuda cuando una IA te quita tu trabajo.
"""

import logging
import sys
import uvicorn
from pathlib import Path

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
    """Función principal - Inicio rápido y fácil"""
    logger.info("🎯 Iniciando AI Job Replacement Helper...")

    from .api.app_factory import create_app
    
    app = create_app(
        title="AI Job Replacement Helper API",
        version="1.0.0"
    )

    logger.info("✅ Aplicación creada exitosamente")
    logger.info("📡 Servidor disponible en http://0.0.0.0:8030")
    logger.info("🏥 Health check: http://0.0.0.0:8030/health")
    logger.info("📊 Dashboard: http://0.0.0.0:8030/dashboard")

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




