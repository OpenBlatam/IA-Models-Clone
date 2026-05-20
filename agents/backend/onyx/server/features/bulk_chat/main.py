"""
Bulk Chat - Main Entry Point
=============================

Punto de entrada principal para el sistema de chat continuo proactivo.
"""

import asyncio
import logging
import sys
import argparse
from pathlib import Path
import os

from .config.chat_config import ChatConfig
from .api.chat_api import ChatAPI

# Intentar cargar .env automáticamente
try:
    from dotenv import load_dotenv
    # Cargar .env desde el directorio actual y el directorio padre
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Intentar cargar desde el directorio padre
        parent_env = Path(__file__).parent.parent / '.env'
        if parent_env.exists():
            load_dotenv(parent_env)
        else:
            # Intentar cargar .env si existe en el directorio actual
            load_dotenv()
except ImportError:
    # dotenv no está instalado, continuar sin él
    pass

# Setup logging
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / 'bulk_chat.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bulk Chat - Sistema de Chat Continuo Proactivo"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host para el servidor API"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8006,
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
        "--llm-provider",
        choices=["openai", "anthropic", "mock"],
        default="openai",
        help="Proveedor de LLM"
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4",
        help="Modelo de LLM a usar"
    )
    parser.add_argument(
        "--auto-continue",
        action="store_true",
        default=True,
        help="Continuar automáticamente después de responder"
    )
    parser.add_argument(
        "--response-interval",
        type=float,
        default=2.0,
        help="Intervalo entre respuestas automáticas (segundos)"
    )
    
    args = parser.parse_args()
    
    # Setup configuration
    config = ChatConfig()
    
    # Cargar configuración adicional desde archivo si se especifica
    if args.config:
        try:
            from dotenv import load_dotenv
            load_dotenv(args.config)
            config = ChatConfig()  # Recargar con nueva configuración
            logger.info(f"Loaded configuration from: {args.config}")
        except Exception as e:
            logger.warning(f"Could not load config file {args.config}: {e}")
    
    # Aplicar argumentos de línea de comandos
    if args.debug:
        config.log_level = "DEBUG"
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    if args.llm_provider:
        config.llm_provider = args.llm_provider
    
    if args.llm_model:
        config.llm_model = args.llm_model
    
    config.auto_continue = args.auto_continue
    config.response_interval = args.response_interval
    
    # Validar configuración crítica
    if config.llm_provider != "mock" and not config.llm_api_key:
        logger.warning(
            f"No API key found for provider '{config.llm_provider}'. "
            "Switching to mock provider. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
        )
        config.llm_provider = "mock"
    
    # Crear directorios necesarios
    Path(config.storage_path).mkdir(exist_ok=True)
    Path(config.backup_directory).mkdir(exist_ok=True)
    
    # Create and run API
    logger.info("=" * 60)
    logger.info("Starting Bulk Chat API...")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"LLM Provider: {config.llm_provider}")
    logger.info(f"LLM Model: {config.llm_model}")
    logger.info(f"Auto Continue: {config.auto_continue}")
    logger.info(f"Response Interval: {config.response_interval}s")
    logger.info("=" * 60)
    
    api = ChatAPI(config)
    
    try:
        logger.info(f"🚀 Server starting at http://{args.host}:{args.port}")
        logger.info(f"📚 API Documentation: http://{args.host}:{args.port}/docs")
        logger.info(f"📊 Dashboard: http://{args.host}:{args.port}/dashboard")
        logger.info("Press Ctrl+C to stop the server")
        
        api.run(
            host=args.host,
            port=args.port,
            reload=args.debug
        )
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutdown requested by user")
        logger.info("Gracefully shutting down...")
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

