"""
Script de inicio alternativo para el servidor
=============================================

Ejecuta el servidor sin problemas de imports relativos.
"""

import sys
from pathlib import Path

# Agregar el directorio actual al path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Ahora podemos importar normalmente
from config.character_consistency_config import CharacterConsistencyConfig
from core.character_consistency_service import CharacterConsistencyService
from api.character_consistency_api import app
import uvicorn
import logging
import signal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    logger.info("Starting Character Consistency AI Server")
    
    # Create configuration
    config = CharacterConsistencyConfig.from_env()
    config.validate()
    
    # Create service
    service = CharacterConsistencyService(config=config)
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, shutting down...")
        service.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start API server
        logger.info(f"Starting API server on {config.api_host}:{config.api_port}")
        logger.info(f"API Documentation: http://{config.api_host}:{config.api_port}/docs")
        logger.info(f"Health Check: http://{config.api_host}:{config.api_port}/api/v1/health")
        
        uvicorn.run(
            app,
            host=config.api_host,
            port=config.api_port,
            log_level="info",
        )
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Error running service: {e}", exc_info=True)
    finally:
        service.close()
        logger.info("Service stopped")


if __name__ == "__main__":
    main()


