"""
Command Listener - Escucha comandos desde Cursor
=================================================

Escucha comandos desde la ventana de Cursor y los envía al agente.
"""

import asyncio
import logging
from typing import Optional, Callable, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class CommandListener:
    """Escucha comandos desde Cursor"""
    
    def __init__(self, callback: Optional[Callable] = None, use_mcp: bool = True):
        self.callback = callback
        self.running = False
        self._listen_task: Optional[asyncio.Task] = None
        self._cursor_api_client = None
        self.use_mcp = use_mcp
        
    async def start(self) -> None:
        """Iniciar listener"""
        if self.running:
            logger.warning("Listener is already running")
            return
            
        logger.info("👂 Starting command listener...")
        self.running = True
        
        if self.use_mcp:
            try:
                from .mcp_client import CursorAPIClient
                self._cursor_api_client = CursorAPIClient()
                await self._cursor_api_client.initialize()
                logger.info("✅ Cursor MCP client initialized")
            except ImportError:
                logger.warning("⚠️  MCP client not available, falling back to file-based commands")
                self.use_mcp = False
                self._cursor_api_client = None
        else:
            logger.info("📝 Using file-based command listening")
        
        self._listen_task = asyncio.create_task(self._listen_loop())
        
    async def stop(self) -> None:
        """Detener listener"""
        if not self.running:
            return
            
        logger.info("🛑 Stopping command listener...")
        self.running = False
        
        if self._cursor_api_client:
            try:
                await self._cursor_api_client.mcp_client.disconnect()
            except Exception as e:
                logger.debug(f"Error disconnecting MCP client: {e}")
        
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
                
    async def _listen_loop(self) -> None:
        """Loop principal de escucha"""
        while self.running:
            try:
                if self.use_mcp and self._cursor_api_client:
                    command_data = await self._cursor_api_client.get_next_command(timeout=0.5)
                    if command_data:
                        command, metadata = command_data
                        await self._handle_command(command, metadata)
                else:
                    await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in listen loop: {e}")
                await asyncio.sleep(5)
                
    async def _handle_command(self, command: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Manejar comando recibido"""
        logger.info(f"📨 Command received: {command[:50]}...")
        
        if self.callback:
            try:
                if asyncio.iscoroutinefunction(self.callback):
                    await self.callback(command)
                else:
                    self.callback(command)
            except Exception as e:
                logger.error(f"Error in command callback: {e}")
