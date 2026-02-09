"""
Graceful Shutdown - Cierre ordenado
===================================

Sistema para cierre ordenado de la aplicación.
"""

import asyncio
import logging
import signal
from typing import List, Callable, Awaitable

logger = logging.getLogger(__name__)


class GracefulShutdown:
    """Gestor de cierre ordenado."""
    
    def __init__(self):
        self.shutdown_handlers: List[Callable[[], Awaitable[None]]] = []
        self.is_shutting_down = False
    
    def register(self, handler: Callable[[], Awaitable[None]]) -> None:
        """
        Registrar handler de shutdown.
        
        Args:
            handler: Función async para ejecutar en shutdown.
        """
        self.shutdown_handlers.append(handler)
        logger.debug(f"Shutdown handler registered: {handler.__name__}")
    
    async def shutdown(self) -> None:
        """Ejecutar todos los handlers de shutdown."""
        if self.is_shutting_down:
            return
        
        self.is_shutting_down = True
        logger.info("Starting graceful shutdown...")
        
        # Ejecutar handlers en paralelo
        tasks = [handler() for handler in self.shutdown_handlers]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Graceful shutdown completed")
    
    def setup_signal_handlers(self) -> None:
        """Configurar handlers de señales."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info("Signal handlers configured")


# Shutdown manager global
_shutdown_manager: 'GracefulShutdown' = None


def get_shutdown_manager() -> 'GracefulShutdown':
    """Obtener gestor de shutdown global."""
    global _shutdown_manager
    
    if _shutdown_manager is None:
        _shutdown_manager = GracefulShutdown()
    
    return _shutdown_manager




