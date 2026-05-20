"""
Graceful Shutdown
=================
Utilidades para shutdown graceful de la aplicación.
"""

import asyncio
import signal
from typing import List, Callable, Optional
from datetime import datetime

from .logger import get_logger

logger = get_logger(__name__)


class GracefulShutdown:
    """Manager para shutdown graceful."""
    
    def __init__(self, shutdown_timeout: float = 30.0):
        """
        Inicializar graceful shutdown.
        
        Args:
            shutdown_timeout: Timeout para shutdown en segundos
        """
        self.shutdown_timeout = shutdown_timeout
        self.shutdown_handlers: List[Callable] = []
        self.is_shutting_down = False
        self.shutdown_started_at: Optional[datetime] = None
    
    def register_handler(self, handler: Callable):
        """
        Registrar handler de shutdown.
        
        Args:
            handler: Función async a ejecutar en shutdown
        """
        self.shutdown_handlers.append(handler)
        logger.debug(f"Registered shutdown handler: {handler.__name__}")
    
    async def shutdown(self):
        """Ejecutar shutdown graceful."""
        if self.is_shutting_down:
            logger.warning("Shutdown already in progress")
            return
        
        self.is_shutting_down = True
        self.shutdown_started_at = datetime.now()
        
        logger.info(f"Starting graceful shutdown (timeout: {self.shutdown_timeout}s)")
        
        try:
            # Ejecutar handlers con timeout
            tasks = [asyncio.create_task(handler()) for handler in self.shutdown_handlers]
            
            done, pending = await asyncio.wait(
                tasks,
                timeout=self.shutdown_timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancelar tareas pendientes
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            logger.info(f"Graceful shutdown completed. Handlers executed: {len(done)}")
            
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
        finally:
            duration = (datetime.now() - self.shutdown_started_at).total_seconds()
            logger.info(f"Shutdown completed in {duration:.2f}s")
    
    def setup_signal_handlers(self):
        """Configurar handlers de señales del sistema."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)


# Instancia global
_graceful_shutdown: Optional[GracefulShutdown] = None


def get_graceful_shutdown() -> GracefulShutdown:
    """Obtener instancia global."""
    global _graceful_shutdown
    if _graceful_shutdown is None:
        _graceful_shutdown = GracefulShutdown()
    return _graceful_shutdown

