"""
WebSocket Handler - Manejo de WebSockets para chat
"""
from typing import Callable, Optional
import asyncio


class WebSocketHandler:
    """Manejador de WebSockets para chat en tiempo real"""
    
    def __init__(self):
        self.connections: dict = {}
        self.message_handlers: list = []
    
    async def handle_connection(self, websocket, user_id: str):
        """Maneja una nueva conexión WebSocket"""
        self.connections[user_id] = websocket
        try:
            async for message in websocket:
                await self._handle_message(user_id, message)
        finally:
            if user_id in self.connections:
                del self.connections[user_id]
    
    async def _handle_message(self, user_id: str, message: str):
        """Maneja un mensaje recibido"""
        for handler in self.message_handlers:
            await handler(user_id, message)
    
    async def send_message(self, user_id: str, message: str):
        """Envía un mensaje a un usuario"""
        if user_id in self.connections:
            await self.connections[user_id].send(message)
    
    def register_handler(self, handler: Callable):
        """Registra un manejador de mensajes"""
        self.message_handlers.append(handler)

