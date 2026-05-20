"""
Addition Removal AI - Main Entry Point
=======================================

Punto de entrada principal para el sistema de IA de adiciones y eliminaciones.
"""

import asyncio
import logging
import sys
import argparse
from pathlib import Path
import os

from .config.config_manager import ConfigManager
from .api.server import create_app

# Intentar cargar .env automáticamente
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        parent_env = Path(__file__).parent.parent / '.env'
        if parent_env.exists():
            load_dotenv(parent_env)
        else:
            load_dotenv()
except ImportError:
    pass

# Setup logging
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'addition_removal_ai.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Addition Removal AI - Sistema IA de Adiciones y Eliminaciones"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host para el servidor API"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8010,
        help="Puerto para el servidor API"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Activar recarga automática (desarrollo)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Ruta al archivo de configuración"
    )

    args = parser.parse_args()

    # Cargar configuración
    config_manager = ConfigManager(config_path=args.config)
    config = config_manager.get_config()

    # Crear aplicación FastAPI
    app = create_app(config)

    # Iniciar servidor
    import uvicorn
    logger.info(f"Iniciando Addition Removal AI en {args.host}:{args.port}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()






