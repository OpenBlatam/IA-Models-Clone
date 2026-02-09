"""
Streaming Optimizer
Optimizaciones para streaming de audio
"""

import logging
import asyncio
import numpy as np
from typing import AsyncIterator, Optional, Any, Dict
from collections import deque

logger = logging.getLogger(__name__)


class StreamingOptimizer:
    """Optimizador para streaming"""
    
    def __init__(self, buffer_size: int = 10, chunk_size: int = 8192):
        self.buffer_size = buffer_size
        self.chunk_size = chunk_size
        self._buffers: Dict[str, deque] = {}
    
    def create_buffered_stream(
        self,
        stream_id: str,
        source: AsyncIterator[bytes]
    ) -> AsyncIterator[bytes]:
        """
        Crea stream con buffer
        
        Args:
            stream_id: ID único del stream
            source: Stream fuente
            
        Returns:
            Stream con buffer
        """
        buffer = deque(maxlen=self.buffer_size)
        self._buffers[stream_id] = buffer
        
        async def buffered():
            # Llenar buffer en background
            fill_task = asyncio.create_task(
                self._fill_buffer(stream_id, source, buffer)
            )
            
            try:
                while True:
                    if buffer:
                        yield buffer.popleft()
                    else:
                        # Esperar un poco si el buffer está vacío
                        await asyncio.sleep(0.01)
                        if not fill_task.done() and not buffer:
                            continue
                        if fill_task.done() and not buffer:
                            break
            finally:
                fill_task.cancel()
                if stream_id in self._buffers:
                    del self._buffers[stream_id]
        
        return buffered()
    
    async def _fill_buffer(
        self,
        stream_id: str,
        source: AsyncIterator[bytes],
        buffer: deque
    ):
        """Llena el buffer en background"""
        try:
            async for chunk in source:
                buffer.append(chunk)
                # No llenar demasiado
                while len(buffer) >= self.buffer_size:
                    await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Error filling buffer for {stream_id}: {e}")
    
    def optimize_chunk_size(self, sample_rate: int, channels: int) -> int:
        """
        Calcula tamaño óptimo de chunk
        
        Args:
            sample_rate: Sample rate
            channels: Número de canales
            
        Returns:
            Tamaño óptimo en bytes
        """
        # Chunk de ~100ms de audio
        samples_per_chunk = int(sample_rate * 0.1)  # 100ms
        bytes_per_sample = 4  # float32
        chunk_size = samples_per_chunk * channels * bytes_per_sample
        
        # Redondear a potencia de 2 para mejor rendimiento
        return 2 ** int(np.log2(chunk_size))
    
    def create_adaptive_stream(
        self,
        source: AsyncIterator[bytes],
        initial_chunk_size: int = 8192
    ) -> AsyncIterator[bytes]:
        """
        Crea stream adaptativo que ajusta chunk size
        
        Args:
            source: Stream fuente
            initial_chunk_size: Tamaño inicial de chunk
            
        Returns:
            Stream adaptativo
        """
        chunk_size = initial_chunk_size
        last_yield_time = asyncio.get_event_loop().time()
        
        async def adaptive():
            nonlocal chunk_size, last_yield_time
            
            buffer = b''
            async for chunk in source:
                buffer += chunk
                
                # Ajustar chunk size basado en throughput
                current_time = asyncio.get_event_loop().time()
                elapsed = current_time - last_yield_time
                
                if elapsed > 0:
                    throughput = len(buffer) / elapsed
                    # Aumentar chunk size si throughput es alto
                    if throughput > 100000:  # 100KB/s
                        chunk_size = min(chunk_size * 2, 65536)
                    elif throughput < 10000:  # 10KB/s
                        chunk_size = max(chunk_size // 2, 4096)
                
                # Yield cuando tengamos suficiente
                while len(buffer) >= chunk_size:
                    yield buffer[:chunk_size]
                    buffer = buffer[chunk_size:]
                    last_yield_time = asyncio.get_event_loop().time()
            
            # Yield resto
            if buffer:
                yield buffer
        
        return adaptive()


# Instancia global
_streaming_optimizer: Optional[StreamingOptimizer] = None


def get_streaming_optimizer() -> StreamingOptimizer:
    """Obtiene el optimizador de streaming"""
    global _streaming_optimizer
    if _streaming_optimizer is None:
        _streaming_optimizer = StreamingOptimizer()
    return _streaming_optimizer

