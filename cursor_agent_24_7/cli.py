"""
CLI - Interfaz de línea de comandos para Cursor Agent 24/7
===========================================================

Comando simple para iniciar el agente:
    cursor-agent start
    cursor-agent start --port 8024
    cursor-agent start --aws
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

# Intentar importar typer, fallback a argparse
try:
    import typer
    from typer import Option
    HAS_TYPER = True
except ImportError:
    HAS_TYPER = False
    import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Crear app typer si está disponible
if HAS_TYPER:
    app = typer.Typer(
        name="cursor-agent",
        help="🚀 Cursor Agent 24/7 - Agente persistente para ejecutar comandos",
        add_completion=False
    )


async def _run_api(
    host: str = "0.0.0.0",
    port: int = 8024,
    use_aws: bool = False,
    aws_region: Optional[str] = None
) -> None:
    """Ejecutar API REST."""
    from .api.agent_api import AgentAPI
    from .core.agent import CursorAgent, AgentConfig
    
    # Configurar AWS si se solicita
    if use_aws or aws_region:
        if aws_region:
            os.environ["AWS_REGION"] = aws_region
        os.environ["AWS_REGION"] = os.getenv("AWS_REGION", "us-east-1")
        logger.info(f"🌩️  AWS mode enabled (region: {os.getenv('AWS_REGION')})")
    
    config = AgentConfig(
        persistent_storage=True,
        auto_restart=True
    )
    
    agent = CursorAgent(config)
    api = AgentAPI(agent, host=host, port=port)
    
    logger.info(f"🌐 Starting API server on http://{host}:{port}")
    logger.info(f"📊 Health check: http://{host}:{port}/api/health")
    logger.info(f"📖 API docs: http://{host}:{port}/docs")
    
    try:
        await api.run()
    except Exception as e:
        logger.error(f"API error: {e}", exc_info=True)
        raise RuntimeError(f"Failed to run API: {e}") from e


async def _run_service(use_aws: bool = False) -> None:
    """Ejecutar como servicio persistente."""
    from .core.agent import CursorAgent, AgentConfig
    from .core.persistent_service import PersistentService
    
    # Configurar AWS si se solicita
    if use_aws:
        os.environ["AWS_REGION"] = os.getenv("AWS_REGION", "us-east-1")
        logger.info(f"🌩️  AWS mode enabled (region: {os.getenv('AWS_REGION')})")
    
    config = AgentConfig(
        persistent_storage=True,
        auto_restart=True
    )
    
    agent = CursorAgent(config)
    service = PersistentService(agent)
    
    logger.info("🚀 Starting Cursor Agent 24/7 as persistent service...")
    try:
        await service.run()
    except Exception as e:
        logger.error(f"Service error: {e}", exc_info=True)
        raise RuntimeError(f"Failed to run service: {e}") from e


if HAS_TYPER:
    @app.command()
    def start(
        host: str = Option("0.0.0.0", "--host", "-h", help="Host para el servidor"),
        port: int = Option(8024, "--port", "-p", help="Puerto para el servidor"),
        mode: str = Option("api", "--mode", "-m", help="Modo: 'api' o 'service'"),
        aws: bool = Option(False, "--aws", help="Habilitar modo AWS (DynamoDB, Redis, CloudWatch)"),
        aws_region: Optional[str] = Option(None, "--aws-region", help="Región AWS (default: us-east-1)")
    ):
        """
        🚀 Iniciar Cursor Agent 24/7
        
        Ejemplos:
            cursor-agent start                    # Iniciar API en puerto 8024
            cursor-agent start --port 8080        # Iniciar en puerto personalizado
            cursor-agent start --mode service     # Modo servicio persistente
            cursor-agent start --aws              # Habilitar servicios AWS
            cursor-agent start --aws --aws-region eu-west-1  # AWS en región específica
        """
        try:
            if mode == "service":
                asyncio.run(_run_service(use_aws=aws))
            else:
                asyncio.run(_run_api(host=host, port=port, use_aws=aws, aws_region=aws_region))
        except KeyboardInterrupt:
            logger.info("⛔ Interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            sys.exit(1)
    
    @app.command()
    def version():
        """Mostrar versión."""
        typer.echo("Cursor Agent 24/7 v1.0.0")
    
    @app.command()
    def health(
        url: str = Option("http://localhost:8024", "--url", "-u", help="URL del servidor")
    ):
        """Verificar salud del agente."""
        import httpx
        
        try:
            response = httpx.get(f"{url}/api/health", timeout=5.0)
            if response.status_code == 200:
                typer.echo(f"✅ Agent is healthy: {response.json()}")
            else:
                typer.echo(f"❌ Agent returned status {response.status_code}")
                sys.exit(1)
        except Exception as e:
            typer.echo(f"❌ Error checking health: {e}")
            sys.exit(1)


def main():
    """Punto de entrada principal."""
    if HAS_TYPER:
        app()
    else:
        # Fallback a argparse
        parser = argparse.ArgumentParser(
            description="🚀 Cursor Agent 24/7 - Agente persistente para ejecutar comandos"
        )
        parser.add_argument(
            "command",
            choices=["start", "version", "health"],
            help="Comando a ejecutar"
        )
        parser.add_argument("--host", default="0.0.0.0", help="Host para el servidor")
        parser.add_argument("--port", type=int, default=8024, help="Puerto para el servidor")
        parser.add_argument("--mode", choices=["api", "service"], default="api", help="Modo de ejecución")
        parser.add_argument("--aws", action="store_true", help="Habilitar modo AWS")
        parser.add_argument("--aws-region", help="Región AWS")
        parser.add_argument("--url", default="http://localhost:8024", help="URL para health check")
        
        args = parser.parse_args()
        
        try:
            if args.command == "start":
                if args.mode == "service":
                    asyncio.run(_run_service(use_aws=args.aws))
                else:
                    asyncio.run(_run_api(host=args.host, port=args.port, use_aws=args.aws, aws_region=args.aws_region))
            elif args.command == "version":
                print("Cursor Agent 24/7 v1.0.0")
            elif args.command == "health":
                import httpx
                response = httpx.get(f"{args.url}/api/health", timeout=5.0)
                if response.status_code == 200:
                    print(f"✅ Agent is healthy: {response.json()}")
                else:
                    print(f"❌ Agent returned status {response.status_code}")
                    sys.exit(1)
        except KeyboardInterrupt:
            logger.info("⛔ Interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()




