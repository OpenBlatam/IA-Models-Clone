"""
Signal Handler - Manejo de Señales del Sistema
===============================================

Manejo robusto de señales del sistema para shutdown graceful.
"""

import asyncio
import logging
import signal
import sys
from typing import Optional, Callable, List
from functools import partial

logger = logging.getLogger(__name__)


class SignalHandler:
    """
    Manejador de señales del sistema para shutdown graceful.
    """
    
    def __init__(self):
        self._shutdown_event = asyncio.Event()
        self._shutdown_callbacks: List[Callable] = []
        self._signal_received = False
        self._original_handlers = {}
        
    def register_shutdown_callback(self, callback: Callable) -> None:
        """
        Registrar callback para ejecutar durante shutdown.
        
        Args:
            callback: Función async o sync a ejecutar
        """
        self._shutdown_callbacks.append(callback)
    
    def setup(self) -> None:
        """Configurar handlers de señales"""
        if sys.platform == "win32":
            signals = [signal.SIGINT, signal.SIGBREAK]
        else:
            signals = [signal.SIGINT, signal.SIGTERM]
        
        for sig in signals:
            try:
                self._original_handlers[sig] = signal.signal(sig, self._signal_handler)
            except (ValueError, OSError) as e:
                logger.debug(f"Could not set signal handler for {sig}: {e}")
    
    def restore(self) -> None:
        """Restaurar handlers originales"""
        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
    
    def _signal_handler(self, signum, frame) -> None:
        """Manejador de señales"""
        signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        logger.info(f"📶 Received signal {signal_name} ({signum}), initiating graceful shutdown...")
        
        if not self._signal_received:
            self._signal_received = True
            self._shutdown_event.set()
        else:
            logger.warning("⚠️ Second signal received, forcing immediate shutdown")
            sys.exit(1)
    
    async def wait_for_shutdown(self, timeout: float = 30.0) -> bool:
        """
        Esperar por señal de shutdown.
        
        Args:
            timeout: Tiempo máximo de espera en segundos
            
        Returns:
            True si se recibió señal, False si fue timeout
        """
        try:
            await asyncio.wait_for(self._shutdown_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
    
    async def shutdown(self, timeout: float = 30.0) -> None:
        """
        Ejecutar shutdown graceful con todos los callbacks.
        
        Args:
            timeout: Tiempo máximo para completar shutdown
        """
        if not self._signal_received:
            logger.info("🛑 Initiating manual shutdown...")
            self._shutdown_event.set()
        
        logger.info(f"🔄 Executing {len(self._shutdown_callbacks)} shutdown callbacks...")
        
        shutdown_tasks = []
        for callback in self._shutdown_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    task = asyncio.create_task(callback())
                else:
                    task = asyncio.create_task(self._run_sync_callback(callback))
                shutdown_tasks.append(task)
            except Exception as e:
                logger.error(f"Error creating shutdown task for {callback}: {e}")
        
        if shutdown_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*shutdown_tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ Shutdown timeout after {timeout}s, some callbacks may not have completed")
        
        logger.info("✅ Graceful shutdown complete")
    
    async def _run_sync_callback(self, callback: Callable) -> None:
        """Ejecutar callback síncrono en thread pool"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, callback)
    
    def is_shutdown_requested(self) -> bool:
        """Verificar si se ha solicitado shutdown"""
        return self._signal_received or self._shutdown_event.is_set()

