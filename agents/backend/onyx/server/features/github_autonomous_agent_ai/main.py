"""
GitHub Autonomous Agent AI - Main Entry Point
==============================================

Punto de entrada principal para el agente autónomo de GitHub.
"""

import asyncio
import logging
import sys
import argparse
from pathlib import Path

from .core.agent import GitHubAutonomousAgent
from .core.service import PersistentService
from .api.app import create_app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)


async def run_service() -> None:
    """Ejecutar como servicio persistente. Solo se detiene cuando el usuario presiona el botón de parar."""
    logger.info("🚀 Iniciando GitHub Autonomous Agent AI como servicio...")
    logger.info("⚠️  IMPORTANTE: El servicio NO se detendrá automáticamente.")
    logger.info("⚠️  Solo se detendrá cuando presiones el botón de parar en la interfaz web.")
    
    service = None
    try:
        service = PersistentService()
        await service.start()
        
        logger.info("✅ Servicio iniciado correctamente")
        logger.info("📡 El agente está escuchando instrucciones...")
        logger.info("🔄 El servicio continuará ejecutándose indefinidamente hasta que lo detengas manualmente.")
        
        await service.run_forever()
    except KeyboardInterrupt:
        # KeyboardInterrupt no debería detener el servicio automáticamente
        logger.warning("⚠️  Interrupción de teclado recibida, pero el servicio continuará.")
        logger.warning("⚠️  Para detener el servicio, usa el botón de parar en la interfaz web.")
        if service and service.is_running():
            logger.info("El servicio sigue ejecutándose. Usa la interfaz web para detenerlo.")
    except Exception as e:
        logger.error(f"❌ Error en el servicio: {e}", exc_info=True)
        # En lugar de hacer raise, intentar reiniciar el servicio
        logger.warning("⚠️  Intentando continuar a pesar del error...")
        if service:
            try:
                await asyncio.sleep(5)
                await service.start()
                await service.run_forever()
            except Exception as retry_error:
                logger.error(f"❌ Error al reintentar: {retry_error}", exc_info=True)
                # Solo en caso de error crítico, hacer raise
                raise


def run_api_server(host: str = "0.0.0.0", port: int = 8025) -> None:
    """Ejecutar servidor API FastAPI."""
    import uvicorn
    
    logger.info(f"🚀 Iniciando servidor API en {host}:{port}...")
    
    app = create_app()
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


def main() -> None:
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="GitHub Autonomous Agent AI"
    )
    parser.add_argument(
        "--mode",
        choices=["service", "api"],
        default="service",
        help="Modo de ejecución: service (daemon) o api (servidor HTTP)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host para el servidor API (solo en modo api)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8025,
        help="Puerto para el servidor API (solo en modo api)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "service":
        asyncio.run(run_service())
    elif args.mode == "api":
        run_api_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
