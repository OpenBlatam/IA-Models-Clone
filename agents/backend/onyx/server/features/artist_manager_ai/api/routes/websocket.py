"""
WebSocket Routes
================

Rutas WebSocket para tiempo real.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional
from ..websocket import handle_connection

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: Optional[str] = Query(None)
):
    """
    Endpoint WebSocket principal.
    
    Args:
        websocket: Conexión WebSocket
        user_id: ID del usuario (opcional)
    """
    await handle_connection(websocket, user_id)


@router.websocket("/ws/{artist_id}")
async def artist_websocket_endpoint(
    websocket: WebSocket,
    artist_id: str
):
    """
    Endpoint WebSocket para artista específico.
    
    Args:
        websocket: Conexión WebSocket
        artist_id: ID del artista
    """
    await handle_connection(websocket, artist_id)




