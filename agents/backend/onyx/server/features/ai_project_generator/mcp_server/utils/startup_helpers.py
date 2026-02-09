"""
Startup Helpers - Utilidades para inicialización y startup del servidor
========================================================================

Funciones helper para facilitar el startup y shutdown del servidor MCP.
"""

import logging
import signal
import sys
from typing import Optional, Callable, List
from contextlib import asynccontextmanager
from fastapi import FastAPI

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager para FastAPI.
    
    Maneja startup y shutdown del servidor.
    
    Args:
        app: Aplicación FastAPI
    
    Yields:
        None
    """
    # Startup
    logger.info("Starting MCP Server...")
    
    # Ejecutar callbacks de startup
    startup_callbacks = getattr(app.state, "startup_callbacks", [])
    for callback in startup_callbacks:
        try:
            if callable(callback):
                if hasattr(callback, "__call__"):
                    if hasattr(callback, "__await__"):
                        await callback()
                    else:
                        callback()
        except Exception as e:
            logger.error(f"Error in startup callback: {e}", exc_info=True)
    
    logger.info("MCP Server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MCP Server...")
    
    # Ejecutar callbacks de shutdown
    shutdown_callbacks = getattr(app.state, "shutdown_callbacks", [])
    for callback in shutdown_callbacks:
        try:
            if callable(callback):
                if hasattr(callback, "__await__"):
                    await callback()
                else:
                    callback()
        except Exception as e:
            logger.error(f"Error in shutdown callback: {e}", exc_info=True)
    
    logger.info("MCP Server shut down")


def register_startup_callback(app: FastAPI, callback: Callable) -> None:
    """
    Registrar callback de startup.
    
    Args:
        app: Aplicación FastAPI
        callback: Función a ejecutar en startup
    """
    if not hasattr(app.state, "startup_callbacks"):
        app.state.startup_callbacks = []
    app.state.startup_callbacks.append(callback)


def register_shutdown_callback(app: FastAPI, callback: Callable) -> None:
    """
    Registrar callback de shutdown.
    
    Args:
        app: Aplicación FastAPI
        callback: Función a ejecutar en shutdown
    """
    if not hasattr(app.state, "shutdown_callbacks"):
        app.state.shutdown_callbacks = []
    app.state.shutdown_callbacks.append(callback)


def setup_signal_handlers(
    app: FastAPI,
    shutdown_callback: Optional[Callable] = None
) -> None:
    """
    Configurar handlers de señales para shutdown graceful.
    
    Args:
        app: Aplicación FastAPI
        shutdown_callback: Callback de shutdown (opcional)
    """
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        if shutdown_callback:
            try:
                shutdown_callback()
            except Exception as e:
                logger.error(f"Error in shutdown callback: {e}", exc_info=True)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def check_dependencies() -> List[str]:
    """
    Verificar dependencias requeridas.
    
    Returns:
        Lista de dependencias faltantes (vacía si todo está bien)
    """
    missing = []
    
    # Verificar dependencias opcionales
    try:
        import yaml
    except ImportError:
        missing.append("pyyaml (opcional, para soporte YAML)")
    
    try:
        import redis
    except ImportError:
        missing.append("redis (opcional, para cache Redis)")
    
    try:
        import opentelemetry
    except ImportError:
        missing.append("opentelemetry (opcional, para observabilidad)")
    
    return missing


def print_startup_banner(version: str = "1.0.0") -> None:
    """
    Imprimir banner de startup.
    
    Args:
        version: Versión del servidor
    """
    banner = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║          MCP Server - Model Context Protocol            ║
    ║                                                          ║
    ║                    Version {version:<20}                    ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def validate_environment() -> bool:
    """
    Validar entorno de ejecución.
    
    Returns:
        True si el entorno es válido
    """
    import sys
    
    # Verificar versión de Python
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ is required")
        return False
    
    return True

