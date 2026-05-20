"""
MCP Streaming - Soporte para respuestas streaming
==================================================
"""

import asyncio
import json
import logging
from typing import AsyncIterator, Any, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from .contracts import ContextFrame
from .exceptions import MCPError

logger = logging.getLogger(__name__)


class StreamChunk(BaseModel):
    """Chunk de datos en stream"""
    chunk_id: str
    data: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_complete: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StreamResponse:
    """
    Response streaming
    
    Permite enviar respuestas en chunks para operaciones largas.
    """
    
    def __init__(self, stream_id: str):
        """
        Args:
            stream_id: ID único del stream
        """
        self.stream_id = stream_id
        self._chunks: list[StreamChunk] = []
        self._complete = False
    
    async def send_chunk(self, data: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Envía un chunk de datos
        
        Args:
            data: Datos del chunk
            metadata: Metadata adicional
        """
        chunk = StreamChunk(
            chunk_id=f"{self.stream_id}_{len(self._chunks)}",
            data=data,
            metadata=metadata or {},
            is_complete=False,
        )
        self._chunks.append(chunk)
    
    async def send_final(self, data: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Envía chunk final y marca stream como completo
        
        Args:
            data: Datos finales
            metadata: Metadata adicional
        """
        chunk = StreamChunk(
            chunk_id=f"{self.stream_id}_final",
            data=data,
            metadata=metadata or {},
            is_complete=True,
        )
        self._chunks.append(chunk)
        self._complete = True
    
    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Genera chunks del stream
        
        Yields:
            Diccionario con datos del chunk
        """
        for chunk in self._chunks:
            yield {
                "stream_id": self.stream_id,
                "chunk_id": chunk.chunk_id,
                "data": chunk.data,
                "metadata": chunk.metadata,
                "is_complete": chunk.is_complete,
                "timestamp": chunk.timestamp.isoformat(),
            }
    
    def is_complete(self) -> bool:
        """Verifica si el stream está completo"""
        return self._complete


async def stream_query(
    query_func: callable,
    *args,
    chunk_size: int = 1000,
    **kwargs
) -> AsyncIterator[Dict[str, Any]]:
    """
    Ejecuta query y retorna resultados en streaming
    
    Args:
        query_func: Función que ejecuta la query
        *args: Argumentos posicionales
        chunk_size: Tamaño de cada chunk
        **kwargs: Argumentos nombrados
        
    Yields:
        Chunks de datos
    """
    import uuid
    stream_id = str(uuid.uuid4())
    stream = StreamResponse(stream_id)
    
    try:
        # Ejecutar query (puede ser async generator)
        if asyncio.iscoroutinefunction(query_func):
            result = await query_func(*args, **kwargs)
        else:
            result = query_func(*args, **kwargs)
        
        # Si es async generator, yield chunks
        if hasattr(result, '__aiter__'):
            async for item in result:
                await stream.send_chunk(item)
        # Si es lista, dividir en chunks
        elif isinstance(result, list):
            for i in range(0, len(result), chunk_size):
                chunk_data = result[i:i + chunk_size]
                await stream.send_chunk(chunk_data, metadata={"chunk_index": i // chunk_size})
        # Si es dict, enviar como chunk único
        elif isinstance(result, dict):
            await stream.send_chunk(result)
        else:
            await stream.send_chunk(result)
        
        # Chunk final
        await stream.send_final(None, metadata={"total_chunks": len(stream._chunks)})
        
    except Exception as e:
        logger.error(f"Error in stream query: {e}")
        await stream.send_final({"error": str(e)}, metadata={"error": True})
    
    # Yield todos los chunks
    async for chunk in stream.stream():
        yield chunk

