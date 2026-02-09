"""
Streaming Response
Respuestas streaming para mejor rendimiento con datos grandes
"""

import logging
from typing import AsyncIterator, Any
from fastapi.responses import StreamingResponse
import orjson

logger = logging.getLogger(__name__)


class StreamingResponseOptimizer:
    """Optimizador de respuestas streaming"""
    
    @staticmethod
    async def stream_json_array(
        items: AsyncIterator[Dict[str, Any]],
        chunk_size: int = 100
    ) -> AsyncIterator[bytes]:
        """
        Stream de array JSON optimizado
        
        Args:
            items: Iterator async de items
            chunk_size: Tamaño del chunk
            
        Yields:
            Chunks de bytes
        """
        # Abrir array
        yield b'['
        
        first = True
        chunk = []
        
        async for item in items:
            chunk.append(item)
            
            if len(chunk) >= chunk_size:
                # Serializar chunk
                chunk_json = orjson.dumps(chunk)
                
                if not first:
                    yield b','
                else:
                    first = False
                
                yield chunk_json
                chunk = []
        
        # Serializar chunk final
        if chunk:
            if not first:
                yield b','
            yield orjson.dumps(chunk)
        
        # Cerrar array
        yield b']'
    
    @staticmethod
    def create_streaming_response(
        items: AsyncIterator[Dict[str, Any]],
        media_type: str = "application/json",
        chunk_size: int = 100
    ) -> StreamingResponse:
        """
        Crea respuesta streaming optimizada
        
        Args:
            items: Iterator async de items
            media_type: Tipo de media
            chunk_size: Tamaño del chunk
            
        Returns:
            StreamingResponse
        """
        return StreamingResponse(
            StreamingResponseOptimizer.stream_json_array(items, chunk_size),
            media_type=media_type
        )















