"""
Stream Helpers
==============
Utilidades para manejo de streams.
"""

from typing import AsyncIterator, Optional, Callable, Any
import json
import asyncio
from datetime import datetime


def format_sse_event(event_type: str, data: Any) -> str:
    """
    Formatear evento SSE.
    
    Args:
        event_type: Tipo de evento
        data: Datos del evento
        
    Returns:
        String formateado SSE
    """
    return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"


def format_sse_data(data: Any) -> str:
    """
    Formatear datos SSE simples.
    
    Args:
        data: Datos a formatear
        
    Returns:
        String formateado SSE
    """
    return f"data: {json.dumps(data, default=str)}\n\n"


async def stream_with_heartbeat(
    stream: AsyncIterator[str],
    heartbeat_interval: float = 30.0,
    heartbeat_data: Optional[dict] = None
) -> AsyncIterator[str]:
    """
    Agregar heartbeat a un stream.
    
    Args:
        stream: Stream original
        heartbeat_interval: Intervalo de heartbeat en segundos
        heartbeat_data: Datos adicionales para heartbeat
        
    Yields:
        Chunks del stream con heartbeats
    """
    heartbeat_data = heartbeat_data or {}
    last_heartbeat = datetime.now()
    
    async def heartbeat_task():
        while True:
            await asyncio.sleep(heartbeat_interval)
            yield format_sse_data({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat(),
                **heartbeat_data
            })
    
    heartbeat_gen = heartbeat_task()
    
    try:
        async for chunk in stream:
            yield chunk
            last_heartbeat = datetime.now()
            
            # Enviar heartbeat si es necesario
            if (datetime.now() - last_heartbeat).total_seconds() >= heartbeat_interval:
                try:
                    heartbeat_chunk = await heartbeat_gen.__anext__()
                    yield heartbeat_chunk
                except StopAsyncIteration:
                    pass
    finally:
        # Enviar mensaje de finalización
        yield format_sse_data({"type": "done", "timestamp": datetime.now().isoformat()})


async def stream_with_error_handling(
    stream: AsyncIterator[str],
    on_error: Optional[Callable[[Exception], str]] = None
) -> AsyncIterator[str]:
    """
    Agregar manejo de errores a un stream.
    
    Args:
        stream: Stream original
        on_error: Función para manejar errores
        
    Yields:
        Chunks del stream con manejo de errores
    """
    try:
        async for chunk in stream:
            yield chunk
    except Exception as e:
        error_message = on_error(e) if on_error else str(e)
        yield format_sse_event("error", {"error": error_message, "timestamp": datetime.now().isoformat()})


async def stream_with_metadata(
    stream: AsyncIterator[str],
    metadata: dict,
    include_in_each: bool = False
) -> AsyncIterator[str]:
    """
    Agregar metadata a un stream.
    
    Args:
        stream: Stream original
        metadata: Metadata a agregar
        include_in_each: Incluir metadata en cada chunk
        
    Yields:
        Chunks del stream con metadata
    """
    # Enviar metadata inicial
    yield format_sse_event("metadata", metadata)
    
    async for chunk in stream:
        if include_in_each:
            # Parsear chunk y agregar metadata
            try:
                chunk_data = json.loads(chunk.replace("data: ", "").strip())
                chunk_data["metadata"] = metadata
                yield format_sse_data(chunk_data)
            except:
                yield chunk
        else:
            yield chunk


def create_stream_response(
    stream: AsyncIterator[str],
    media_type: str = "text/event-stream"
) -> dict:
    """
    Crear configuración de respuesta de stream.
    
    Args:
        stream: Stream a enviar
        media_type: Tipo de media
        
    Returns:
        Configuración de respuesta
    """
    return {
        "stream": stream,
        "media_type": media_type,
        "headers": {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": media_type
        }
    }

