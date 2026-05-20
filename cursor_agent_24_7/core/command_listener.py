"""
Command Listener - Escucha comandos desde Cursor
=================================================

Escucha comandos desde la ventana de Cursor y los envía al agente.
Preparado para integración futura con Cursor API.
"""

import asyncio
import logging
from typing import Optional, Callable, Union, Awaitable
from datetime import datetime

logger = logging.getLogger(__name__)


class CommandListener:
    """
    Escucha comandos desde Cursor.
    
    Gestiona la recepción de comandos desde la API de Cursor
    y los envía al agente para su ejecución.
    """
    
    def __init__(
        self,
        callback: Optional[Callable[[str], Union[None, Awaitable[None]]]] = None
    ) -> None:
        """
        Inicializar listener de comandos.
        
        Args:
            callback: Función o coroutine a llamar cuando se recibe un comando.
        """
        self.callback: Optional[Callable[[str], Union[None, Awaitable[None]]]] = (
            callback
        )
        self.running: bool = False
        self._listen_task: Optional[asyncio.Task[None]] = None
        self._cursor_api_client: Optional[Any] = None
        
    async def start(self) -> None:
        """
        Iniciar listener.
        
        Raises:
            RuntimeError: Si hay error al iniciar.
        """
        if self.running:
            logger.warning("Listener is already running")
            return
        
        logger.info("👂 Starting command listener...")
        self.running = True
        
        try:
            # Inicializar cliente de Cursor API
            # TODO: Inicializar cliente real de Cursor API cuando esté disponible
            # self._cursor_api_client = CursorAPIClient()
            
            self._listen_task = asyncio.create_task(self._listen_loop())
            logger.info("✅ Command listener started")
        
        except Exception as e:
            logger.error(f"Error starting command listener: {e}", exc_info=True)
            self.running = False
            raise RuntimeError(f"Failed to start command listener: {e}") from e
        
    async def stop(self) -> None:
        """
        Detener listener.
        
        Cancela el loop de escucha y limpia recursos.
        """
        if not self.running:
            return
        
        logger.info("🛑 Stopping command listener...")
        self.running = False
        
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        
        # Limpiar cliente de API si existe
        if self._cursor_api_client:
            try:
                # TODO: Cerrar conexión cuando esté implementado
                # await self._cursor_api_client.close()
                pass
            except Exception as e:
                logger.error(f"Error closing API client: {e}", exc_info=True)
        
        logger.info("✅ Command listener stopped")
                
    async def _listen_loop(self) -> None:
        """
        Loop principal de escucha.
        
        Escucha comandos desde la API de Cursor y los procesa.
        """
        while self.running:
            try:
                # TODO: Integrar con Cursor API cuando esté disponible
                # Por ahora, simulamos escuchando comandos
                
                # Ejemplo de integración futura:
                # if self._cursor_api_client:
                #     command = await self._cursor_api_client.get_next_command()
                #     if command:
                #         await self._handle_command(command)
                
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                logger.debug("Listen loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in listen loop: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def _handle_command(self, command: str) -> None:
        """
        Manejar comando recibido.
        
        Args:
            command: Comando recibido.
        
        Raises:
            ValueError: Si el comando está vacío.
        """
        if not command or not command.strip():
            raise ValueError("Command cannot be empty")
        
        logger.info(f"📨 Command received: {command[:50]}...")
        
        if not self.callback:
            logger.warning("No callback set, ignoring command")
            return
        
        try:
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback(command.strip())
            else:
                # Ejecutar callback síncrono
                self.callback(command.strip())
        except Exception as e:
            logger.error(f"Error in command callback: {e}", exc_info=True)
    
    def set_callback(
        self,
        callback: Callable[[str], Union[None, Awaitable[None]]]
    ) -> None:
        """
        Establecer callback para comandos recibidos.
        
        Args:
            callback: Función o coroutine a llamar cuando se recibe un comando.
        
        Raises:
            ValueError: Si callback no es callable.
        """
        if not callable(callback):
            raise ValueError("Callback must be callable")
        self.callback = callback
