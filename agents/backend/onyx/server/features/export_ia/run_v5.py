"""
Export IA V5.0 - ULTIMATE ENTERPRISE SYSTEM
Script principal para ejecutar el sistema completo
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Agregar el directorio del proyecto al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('export_ia_v5.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """Función principal."""
    try:
        logger.info("🚀 Iniciando Export IA V5.0 - ULTIMATE ENTERPRISE SYSTEM")
        
        # Importar y ejecutar la aplicación
        from app.api.enhanced_app_v5 import app
        import uvicorn
        
        # Configuración del servidor
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True,
            use_colors=True
        )
        
        server = uvicorn.Server(config)
        
        logger.info("🌟 Sistema Export IA V5.0 iniciado exitosamente")
        logger.info("📡 Servidor disponible en: http://localhost:8000")
        logger.info("📚 Documentación disponible en: http://localhost:8000/docs")
        logger.info("🔧 ReDoc disponible en: http://localhost:8000/redoc")
        
        await server.serve()
        
    except KeyboardInterrupt:
        logger.info("🛑 Sistema detenido por el usuario")
    except Exception as e:
        logger.error(f"❌ Error al iniciar el sistema: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())




