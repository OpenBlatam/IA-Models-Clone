"""
Streaming API - API de streaming para datos en tiempo real
==========================================================
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class StreamFormat(Enum):
    """Formatos de stream"""
    JSON = "json"
    NDJSON = "ndjson"  # Newline Delimited JSON
    SSE = "sse"  # Server-Sent Events
    BINARY = "binary"


@dataclass
class StreamConfig:
    """Configuración de stream"""
    format: StreamFormat = StreamFormat.JSON
    chunk_size: int = 1024
    delimiter: str = "\n"
    include_metadata: bool = True


class StreamingAPI:
    """API de streaming"""
    
    def __init__(self):
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.stream_handlers: Dict[str, Callable] = {}
    
    def register_stream(
        self,
        stream_id: str,
        handler: Callable,
        config: Optional[StreamConfig] = None
    ):
        """Registra un stream"""
        self.stream_handlers[stream_id] = handler
        self.active_streams[stream_id] = {
            "config": config or StreamConfig(),
            "started_at": datetime.now(),
            "message_count": 0
        }
        logger.info(f"Stream {stream_id} registrado")
    
    async def stream_data(
        self,
        stream_id: str,
        data_source: AsyncIterator[Any],
        config: Optional[StreamConfig] = None
    ) -> AsyncIterator[str]:
        """Stream de datos"""
        if stream_id not in self.stream_handlers:
            raise ValueError(f"Stream {stream_id} no encontrado")
        
        stream_config = config or StreamConfig()
        stream_info = self.active_streams.get(stream_id, {})
        
        try:
            async for item in data_source:
                # Formatear según configuración
                if stream_config.format == StreamFormat.JSON:
                    yield json.dumps(item) + stream_config.delimiter
                elif stream_config.format == StreamFormat.NDJSON:
                    yield json.dumps(item) + "\n"
                elif stream_config.format == StreamFormat.SSE:
                    yield f"data: {json.dumps(item)}\n\n"
                elif stream_config.format == StreamFormat.BINARY:
                    yield str(item).encode()
                
                stream_info["message_count"] = stream_info.get("message_count", 0) + 1
        except Exception as e:
            logger.error(f"Error en stream {stream_id}: {e}")
            if stream_config.format == StreamFormat.SSE:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    
    async def stream_from_handler(
        self,
        stream_id: str,
        *args,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream desde un handler"""
        handler = self.stream_handlers.get(stream_id)
        if not handler:
            raise ValueError(f"Stream {stream_id} no encontrado")
        
        config = self.active_streams.get(stream_id, {}).get("config", StreamConfig())
        
        if asyncio.iscoroutinefunction(handler):
            data_source = await handler(*args, **kwargs)
        else:
            data_source = handler(*args, **kwargs)
        
        async for chunk in self.stream_data(stream_id, data_source, config):
            yield chunk
    
    def get_stream_info(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un stream"""
        if stream_id not in self.active_streams:
            return None
        
        info = self.active_streams[stream_id]
        return {
            "stream_id": stream_id,
            "started_at": info["started_at"].isoformat(),
            "message_count": info.get("message_count", 0),
            "config": {
                "format": info["config"].format.value,
                "chunk_size": info["config"].chunk_size
            }
        }
    
    def list_streams(self) -> List[str]:
        """Lista streams activos"""
        return list(self.active_streams.keys())




