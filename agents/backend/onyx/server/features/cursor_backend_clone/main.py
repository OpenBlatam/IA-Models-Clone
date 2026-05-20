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

from .core.agent import CursorAgent, AgentConfig
from .core.services.persistent_service import PersistentService
from .api.agent_api import AgentAPI
from .core.mcp.server import MCPServer
from .core.signal_handler import SignalHandler
from .core.infrastructure.monitoring.health import HealthChecker

# Setup logging avanzado
try:
    from .core.utils.logging.logging_config import setup_logging
    log_file = Path(__file__).parent / "logs" / "agent.log"
    log_file.parent.mkdir(exist_ok=True)
    setup_logging(
        level="INFO",
        format_type="colored",
        log_file=str(log_file),
        enable_structured=True
    )
except ImportError:
    # Fallback a logging básico
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

logger = logging.getLogger(__name__)


async def run_service():
    """Ejecutar como servicio persistente"""
    signal_handler = SignalHandler()
    signal_handler.setup()
    
    config = AgentConfig.from_env_or_constants()
    
    agent = CursorAgent(config)
    service = PersistentService(agent)
    
    signal_handler.register_shutdown_callback(agent.stop)
    signal_handler.register_shutdown_callback(service.shutdown)
    
    try:
        logger.info("🚀 Starting Cursor Agent 24/7 as persistent service...")
        
        service_task = asyncio.create_task(service.run())
        shutdown_task = asyncio.create_task(signal_handler.wait_for_shutdown())
        
        done, pending = await asyncio.wait(
            [service_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        if signal_handler.is_shutdown_requested():
            logger.info("🛑 Shutdown signal received, initiating graceful shutdown...")
            await signal_handler.shutdown()
        
    finally:
        signal_handler.restore()


async def run_api():
    """Ejecutar API REST"""
    parser = argparse.ArgumentParser(description="Cursor Agent 24/7 API")
    parser.add_argument("--host", default="0.0.0.0", help="Host para el servidor")
    parser.add_argument("--port", type=int, default=8024, help="Puerto para el servidor")
    parser.add_argument("--mcp-port", type=int, default=8025, help="Puerto para servidor MCP")
    parser.add_argument("--enable-mcp", action="store_true", help="Habilitar servidor MCP para Cursor IDE")
    args = parser.parse_args()
    
    signal_handler = SignalHandler()
    signal_handler.setup()
    
    config = AgentConfig.from_env_or_constants()
    
    agent = CursorAgent(config)
    api = AgentAPI(agent, host=args.host, port=args.port)
    
    mcp_server = None
    if args.enable_mcp:
        mcp_server = MCPServer(agent, host=args.host, port=args.mcp_port)
        signal_handler.register_shutdown_callback(mcp_server.shutdown)
        logger.info(f"🔌 MCP server enabled on http://{args.host}:{args.mcp_port}")
    
    signal_handler.register_shutdown_callback(agent.stop)
    signal_handler.register_shutdown_callback(api.shutdown)
    
    try:
        logger.info(f"🌐 Starting API server on http://{args.host}:{args.port}")
        
        if mcp_server:
            asyncio.create_task(mcp_server.run())
        
        api_task = asyncio.create_task(api.run())
        shutdown_task = asyncio.create_task(signal_handler.wait_for_shutdown())
        
        done, pending = await asyncio.wait(
            [api_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        if signal_handler.is_shutdown_requested():
            logger.info("🛑 Shutdown signal received, initiating graceful shutdown...")
            await signal_handler.shutdown()
        
    finally:
        signal_handler.restore()


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Cursor Agent 24/7")
    parser.add_argument(
        "--mode",
        choices=["service", "api"],
        default="api",
        help="Modo de ejecución: service (servicio persistente) o api (API REST)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host para API (solo modo api)")
    parser.add_argument("--port", type=int, default=8024, help="Puerto para API (solo modo api)")
    parser.add_argument("--mcp-port", type=int, default=8025, help="Puerto para servidor MCP (solo modo api)")
    parser.add_argument("--enable-mcp", action="store_true", help="Habilitar servidor MCP para Cursor IDE (solo modo api)")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "service":
            asyncio.run(run_service())
        else:
            # Modificar sys.argv para pasar host, port y opciones MCP a run_api
            import sys
            original_argv = sys.argv
            sys_argv = ["main.py", "--host", args.host, "--port", str(args.port)]
            if args.enable_mcp:
                sys_argv.extend(["--enable-mcp", "--mcp-port", str(args.mcp_port)])
            sys.argv = sys_argv
            try:
                asyncio.run(run_api())
            finally:
                sys.argv = original_argv
                
    except KeyboardInterrupt:
        logger.info("⛔ Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

