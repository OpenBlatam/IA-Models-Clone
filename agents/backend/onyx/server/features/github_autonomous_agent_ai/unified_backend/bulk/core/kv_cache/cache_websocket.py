"""
Cache WebSocket interface.

Provides WebSocket interface for real-time cache operations.
"""
from __future__ import annotations

import logging
import json
import asyncio
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


class CacheWebSocket:
    """
    Cache WebSocket interface.
    
    Provides WebSocket support for cache.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize WebSocket interface.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.connections: set = set()
        self.subscriptions: Dict[str, set] = {}  # position -> set of connections
    
    async def handle_connection(self, websocket: Any) -> None:
        """
        Handle WebSocket connection.
        
        Args:
            websocket: WebSocket connection
        """
        self.connections.add(websocket)
        
        try:
            async for message in websocket:
                await self._handle_message(websocket, message)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.connections.discard(websocket)
            # Remove from all subscriptions
            for subscriptions in self.subscriptions.values():
                subscriptions.discard(websocket)
    
    async def _handle_message(self, websocket: Any, message: str) -> None:
        """
        Handle WebSocket message.
        
        Args:
            websocket: WebSocket connection
            message: Message string
        """
        try:
            data = json.loads(message)
            action = data.get("action")
            
            if action == "get":
                position = data.get("position")
                value = await self._async_get(position)
                await websocket.send(json.dumps({
                    "action": "get_response",
                    "position": position,
                    "value": str(value) if value else None
                }))
            
            elif action == "put":
                position = data.get("position")
                value = data.get("value")
                await self._async_put(position, value)
                await websocket.send(json.dumps({
                    "action": "put_response",
                    "success": True,
                    "position": position
                }))
            
            elif action == "subscribe":
                position = data.get("position")
                self._subscribe(websocket, position)
                await websocket.send(json.dumps({
                    "action": "subscribe_response",
                    "position": position,
                    "success": True
                }))
            
            elif action == "unsubscribe":
                position = data.get("position")
                self._unsubscribe(websocket, position)
                await websocket.send(json.dumps({
                    "action": "unsubscribe_response",
                    "position": position,
                    "success": True
                }))
            
            elif action == "stats":
                stats = self.cache.get_stats()
                await websocket.send(json.dumps({
                    "action": "stats_response",
                    "stats": stats
                }))
        
        except Exception as e:
            await websocket.send(json.dumps({
                "action": "error",
                "error": str(e)
            }))
    
    async def _async_get(self, position: int) -> Optional[Any]:
        """Async get operation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.cache.get, position)
    
    async def _async_put(self, position: int, value: Any) -> None:
        """Async put operation."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.cache.put, position, value)
    
    def _subscribe(self, websocket: Any, position: int) -> None:
        """
        Subscribe to position updates.
        
        Args:
            websocket: WebSocket connection
            position: Cache position
        """
        if position not in self.subscriptions:
            self.subscriptions[position] = set()
        self.subscriptions[position].add(websocket)
    
    def _unsubscribe(self, websocket: Any, position: int) -> None:
        """
        Unsubscribe from position updates.
        
        Args:
            websocket: WebSocket connection
            position: Cache position
        """
        if position in self.subscriptions:
            self.subscriptions[position].discard(websocket)
    
    async def notify_subscribers(self, position: int, value: Any) -> None:
        """
        Notify subscribers of position update.
        
        Args:
            position: Cache position
            value: New value
        """
        if position not in self.subscriptions:
            return
        
        message = json.dumps({
            "action": "update",
            "position": position,
            "value": str(value) if value else None
        })
        
        disconnected = set()
        
        for websocket in self.subscriptions[position]:
            try:
                await websocket.send(message)
            except Exception:
                disconnected.add(websocket)
        
        # Remove disconnected
        for ws in disconnected:
            self.subscriptions[position].discard(ws)

