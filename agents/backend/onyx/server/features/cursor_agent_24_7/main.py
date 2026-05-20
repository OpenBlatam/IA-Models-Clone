"""
Cursor Agent 24/7 - Main Entry Point
=====================================

Punto de entrada principal para el agente persistente.
"""

import asyncio
import logging
import sys
import argparse
from pathlib import Path
from typing import Optional

from .core.agent import CursorAgent, AgentConfig
from .core.persistent_service import PersistentService
from .core.error_handling import safe_async_call, error_context
from .core.validation_utils import validate_port
from .api.agent_api import AgentAPI

# Setup logging avanzado
try:
    from .core.logger_config import setup_logging
    log_file = Path(__file__).parent / "logs" / "agent.log"
    setup_logging(
        log_level="INFO",
        log_file=str(log_file),
        use_json=False,
        use_colors=True
    )
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

logger = logging.getLogger(__name__)


async def run_service() -> None:
    """
    Ejecutar como servicio persistente.
    
    Raises:
        RuntimeError: Si hay error al iniciar el servicio.
    """
    config = AgentConfig(
        persistent_storage=True,
        auto_restart=True
    )
    
    agent = CursorAgent(config)
    service = PersistentService(agent)
    
    logger.info("🚀 Starting Cursor Agent 24/7 as persistent service...")
    
    async def run_service_instance():
        await service.run()
    
    await safe_async_call(
        run_service_instance,
        operation="running persistent service",
        logger_instance=logger,
        reraise=True
    )


async def run_api(host: str = "0.0.0.0", port: int = 8024) -> None:
    """
    Ejecutar API REST.
    
    Args:
        host: Host para el servidor (default: "0.0.0.0").
        port: Puerto para el servidor (default: 8024).
    
    Raises:
        RuntimeError: Si hay error al iniciar la API.
        ValueError: Si el puerto es inválido.
    """
    validate_port(port, "port")
    
    config = AgentConfig(
        persistent_storage=True,
        auto_restart=True
    )
    
    agent = CursorAgent(config)
    api = AgentAPI(agent, host=host, port=port)
    
    logger.info(f"🌐 Starting API server on http://{host}:{port}")
    
    async def run_api_server():
        await api.run()
    
    await safe_async_call(
        run_api_server,
        operation="running API server",
        logger_instance=logger,
        reraise=True
    )


def main() -> None:
    """
    Función principal del agente.
    
    Parsea argumentos de línea de comandos e inicia el agente en el modo
    especificado (service o api).
    
    Uso recomendado: Usar 'python run.py' o 'python cli.py' para mejor experiencia.
    """
    parser = argparse.ArgumentParser(
        description="Cursor Agent 24/7 - Agente persistente para ejecutar comandos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py                    # Iniciar API en puerto 8024
  python main.py --port 8080        # Puerto personalizado
  python main.py --mode service     # Modo servicio persistente
  python main.py --aws              # Habilitar servicios AWS
  
Recomendado: Usar 'python run.py' para mejor experiencia CLI.
        """
    )
    parser.add_argument(
        "--mode",
        choices=["service", "api"],
        default="api",
        help="Modo de ejecución: service (servicio persistente) o api (API REST)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host para API (solo modo api, default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8024,
        help="Puerto para API (solo modo api, default: 8024)"
    )
    parser.add_argument(
        "--aws",
        action="store_true",
        help="Habilitar modo AWS (DynamoDB, Redis, CloudWatch)"
    )
    parser.add_argument(
        "--aws-region",
        default=None,
        help="Región AWS (default: us-east-1 o de variable de entorno)"
    )
    
    args = parser.parse_args()
    
    # Configurar AWS si se solicita
    if args.aws or args.aws_region:
        import os
        if args.aws_region:
            os.environ["AWS_REGION"] = args.aws_region
        os.environ["AWS_REGION"] = os.getenv("AWS_REGION", "us-east-1")
        logger.info(f"🌩️  AWS mode enabled (region: {os.getenv('AWS_REGION')})")
    
    with error_context("main execution"):
        try:
            if args.mode == "service":
                asyncio.run(run_service())
            else:
                asyncio.run(run_api(host=args.host, port=args.port))
                    
        except KeyboardInterrupt:
            logger.info("⛔ Interrupted by user")
            sys.exit(0)
        except ValueError as e:
            logger.error(f"❌ Invalid argument: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()

