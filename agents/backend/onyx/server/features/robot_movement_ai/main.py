"""
Robot Movement AI - Main Entry Point
=====================================

Punto de entrada principal para la plataforma IA de movimiento robótico.
Sistema tipo Tesla Prime para control de robots mediante chat.
"""

import asyncio
import logging
import sys
import argparse
from pathlib import Path
from typing import Optional
import os

from .config.robot_config import RobotConfig
from .api.robot_api import RobotAPI

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

# Setup logging (import after path setup)
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)


def main() -> None:
    """
    Main entry point.
    
    Inicializa y ejecuta el servidor API de Robot Movement AI.
    """
    parser = argparse.ArgumentParser(
        description="Robot Movement AI - Plataforma IA de Movimiento Robótico"
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
        "--config",
        help="Ruta al archivo de configuración (.env)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Modo debug"
    )
    parser.add_argument(
        "--robot-brand",
        choices=["kuka", "abb", "fanuc", "universal_robots", "generic"],
        default="generic",
        help="Marca del robot"
    )
    parser.add_argument(
        "--ros-enabled",
        action="store_true",
        default=True,
        help="Habilitar integración ROS"
    )
    parser.add_argument(
        "--feedback-frequency",
        type=int,
        default=1000,
        help="Frecuencia de feedback en Hz (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # Configurar logging básico primero
    import logging
    logging.basicConfig(level=logging.INFO)
    temp_logger = logging.getLogger(__name__)
    
    # Setup configuration
    config: RobotConfig = RobotConfig()
    
    # Cargar configuración adicional desde archivo si se especifica
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            temp_logger.warning(f"Config file not found: {args.config}")
        else:
            try:
                from dotenv import load_dotenv
                load_dotenv(args.config)
                config = RobotConfig()
                temp_logger.info(f"Loaded configuration from: {args.config}")
            except Exception as e:
                temp_logger.warning(f"Could not load config file {args.config}: {e}")
    
    # Configurar logging estructurado
    try:
        from .core.config.logging_config import setup_logging, get_logger
    except ImportError:
        from .core.logging_config import setup_logging, get_logger
    
    log_file = str(log_dir / 'robot_movement_ai.log')
    structured = os.getenv("STRUCTURED_LOGGING", "false").lower() == "true"
    colored = not structured and sys.stdout.isatty()
    
    try:
        setup_logging(
            level=config.log_level,
            structured=structured,
            colored=colored,
            log_file=log_file
        )
    except Exception as e:
        temp_logger.warning(f"Could not setup structured logging: {e}")
        logging.basicConfig(level=logging.INFO)
    
    logger = get_logger(__name__)
    
    # Aplicar argumentos de línea de comandos
    if args.debug:
        config.log_level = "DEBUG"
        setup_logging(
            level="DEBUG",
            structured=structured,
            colored=colored,
            log_file=log_file
        )
        logger.debug("Debug mode enabled")
    
    config.robot_brand = args.robot_brand
    config.ros_enabled = args.ros_enabled
    config.feedback_frequency = args.feedback_frequency
    
    # Crear directorios necesarios
    Path(config.storage_path).mkdir(exist_ok=True)
    Path(config.logs_directory).mkdir(exist_ok=True)
    
    # Inicializar sistema completo
    from .core.initialization import initialize_system
    
    logger.info("=" * 60)
    logger.info("Initializing Robot Movement AI System...")
    logger.info("=" * 60)
    
    init_result = asyncio.run(initialize_system())
    
    if not init_result.get("success", False):
        logger.error("System initialization failed!")
        logger.error("Please check the errors above")
        sys.exit(1)
    
    # Create and run API
    logger.info("=" * 60)
    logger.info("Starting Robot Movement AI API...")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Robot Brand: {config.robot_brand}")
    logger.info(f"ROS Enabled: {config.ros_enabled}")
    logger.info(f"Feedback Frequency: {config.feedback_frequency} Hz")
    logger.info("=" * 60)
    
    api = RobotAPI(config)
    
    try:
        logger.info(f"🚀 Server starting at http://{args.host}:{args.port}")
        logger.info(f"📚 API Documentation: http://{args.host}:{args.port}/docs")
        logger.info(f"🤖 Robot Control: http://{args.host}:{args.port}/chat")
        logger.info("Press Ctrl+C to stop the server")
        
        api.run(
            host=args.host,
            port=args.port,
            reload=args.debug
        )
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutdown requested by user")
        logger.info("Gracefully shutting down robots...")
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"❌ Port {args.port} is already in use. Try a different port with --port")
        else:
            logger.error(f"❌ OS Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error running API: {e}", exc_info=args.debug)
        sys.exit(1)


if __name__ == "__main__":
    main()

