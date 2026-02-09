"""
GraphQL API Endpoints
====================

Endpoints para API GraphQL y WebSocket.
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, Any, Optional
import logging
import json

from ..core.graphql_api import get_graphql_api
from ..core.websocket_manager import get_websocket_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/graphql", tags=["graphql"])


@router.post("/query")
async def execute_graphql_query(
    query: str,
    variables: Optional[Dict[str, Any]] = None,
    operation_name: Optional[str] = None
) -> Dict[str, Any]:
    """Ejecutar query GraphQL."""
    try:
        graphql_api = get_graphql_api()
        result = await graphql_api.execute_query(
            query=query,
            variables=variables or {},
            operation_name=operation_name
        )
        return result
    except Exception as e:
        logger.error(f"Error executing GraphQL query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resolvers")
async def list_resolvers() -> Dict[str, Any]:
    """Listar resolvers GraphQL."""
    try:
        graphql_api = get_graphql_api()
        resolvers = graphql_api.list_resolvers()
        return {
            "resolvers": [
                {
                    "field_name": r.field_name,
                    "description": r.description
                }
                for r in resolvers
            ],
            "count": len(resolvers)
        }
    except Exception as e:
        logger.error(f"Error listing resolvers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/{connection_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    connection_id: str,
    client_id: Optional[str] = None
):
    """Endpoint WebSocket."""
    await websocket.accept()
    
    manager = get_websocket_manager()
    manager.register_connection(
        connection_id=connection_id,
        websocket=websocket,
        client_id=client_id
    )
    
    try:
        while True:
            # Recibir mensaje
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type", "message")
                
                if message_type == "ping":
                    # Responder ping
                    await websocket.send_json({"type": "pong"})
                elif message_type == "subscribe":
                    # Suscribirse a eventos
                    events = message.get("events", [])
                    # Implementar suscripción
                    await websocket.send_json({
                        "type": "subscribed",
                        "events": events
                    })
                else:
                    # Procesar mensaje
                    await websocket.send_json({
                        "type": "ack",
                        "message_id": message.get("message_id")
                    })
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
    except WebSocketDisconnect:
        manager.unregister_connection(connection_id)
        logger.info(f"WebSocket disconnected: {connection_id}")


@router.get("/websocket/connections")
async def list_websocket_connections() -> Dict[str, Any]:
    """Listar conexiones WebSocket."""
    try:
        manager = get_websocket_manager()
        connections = manager.list_connections()
        return {
            "connections": [
                {
                    "connection_id": c.connection_id,
                    "client_id": c.client_id,
                    "status": c.status.value,
                    "connected_at": c.connected_at,
                    "last_message_at": c.last_message_at
                }
                for c in connections
            ],
            "count": len(connections)
        }
    except Exception as e:
        logger.error(f"Error listing connections: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/websocket/broadcast")
async def broadcast_message(
    message: Dict[str, Any],
    exclude: Optional[list] = None
) -> Dict[str, Any]:
    """Enviar mensaje a todas las conexiones WebSocket."""
    try:
        manager = get_websocket_manager()
        exclude_set = set(exclude) if exclude else None
        sent_count = await manager.broadcast_message(message, exclude=exclude_set)
        return {
            "message": message,
            "sent_count": sent_count
        }
    except Exception as e:
        logger.error(f"Error broadcasting message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






