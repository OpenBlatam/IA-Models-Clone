"""
Sistema de Streaming de Audio en Tiempo Real

Proporciona:
- Streaming de audio en tiempo real
- Buffering inteligente
- Calidad adaptativa
- Múltiples formatos
- Estadísticas de streaming
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import io
import numpy as np

logger = logging.getLogger(__name__)

try:
    import soundfile as sf
    import numpy as np
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    logger.warning("Audio libraries not available, streaming limited")


@dataclass
class StreamConfig:
    """Configuración de streaming"""
    sample_rate: int = 44100
    channels: int = 2
    format: str = "wav"  # wav, mp3, ogg
    bitrate: int = 192  # kbps para formatos comprimidos
    buffer_size: int = 4096  # Tamaño del buffer en frames
    adaptive_quality: bool = True


@dataclass
class StreamStats:
    """Estadísticas de streaming"""
    bytes_sent: int = 0
    chunks_sent: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    current_bitrate: float = 0.0
    buffer_level: float = 0.0
    dropped_chunks: int = 0


class AudioStreamer:
    """Streamer de audio en tiempo real"""
    
    def __init__(self):
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.stream_stats: Dict[str, StreamStats] = {}
        logger.info("AudioStreamer initialized")
    
    async def create_stream(
        self,
        stream_id: str,
        audio_path: str,
        config: Optional[StreamConfig] = None
    ) -> Dict[str, Any]:
        """
        Crea un stream de audio
        
        Args:
            stream_id: ID único del stream
            audio_path: Ruta del archivo de audio
            config: Configuración del stream
        
        Returns:
            Información del stream
        """
        if config is None:
            config = StreamConfig()
        
        # Cargar audio
        if not AUDIO_LIBS_AVAILABLE:
            raise Exception("Audio libraries not available")
        
        try:
            data, sample_rate = sf.read(audio_path)
            
            # Ajustar sample rate si es necesario
            if sample_rate != config.sample_rate:
                # Resample (simplificado, en producción usar librosa)
                ratio = config.sample_rate / sample_rate
                new_length = int(len(data) * ratio)
                data = np.interp(
                    np.linspace(0, len(data), new_length),
                    np.arange(len(data)),
                    data
                )
            
            # Ajustar canales
            if len(data.shape) == 1:
                # Mono a estéreo
                if config.channels == 2:
                    data = np.column_stack([data, data])
            elif data.shape[1] != config.channels:
                # Reducir o expandir canales
                if config.channels == 1:
                    data = np.mean(data, axis=1)
                else:
                    data = np.column_stack([data[:, 0], data[:, 0]])
            
            self.active_streams[stream_id] = {
                "audio_data": data,
                "config": config,
                "position": 0,
                "paused": False,
                "stopped": False
            }
            
            self.stream_stats[stream_id] = StreamStats()
            
            logger.info(f"Stream created: {stream_id}")
            
            return {
                "stream_id": stream_id,
                "duration": len(data) / config.sample_rate,
                "sample_rate": config.sample_rate,
                "channels": config.channels,
                "format": config.format
            }
        
        except Exception as e:
            logger.error(f"Error creating stream: {e}")
            raise
    
    async def stream_chunks(
        self,
        stream_id: str
    ) -> AsyncIterator[bytes]:
        """
        Genera chunks de audio para streaming
        
        Args:
            stream_id: ID del stream
        
        Yields:
            Chunks de audio en bytes
        """
        if stream_id not in self.active_streams:
            raise ValueError(f"Stream not found: {stream_id}")
        
        stream = self.active_streams[stream_id]
        stats = self.stream_stats[stream_id]
        config = stream["config"]
        audio_data = stream["audio_data"]
        
        chunk_size = config.buffer_size
        total_samples = len(audio_data)
        
        while stream["position"] < total_samples and not stream["stopped"]:
            if stream["paused"]:
                await asyncio.sleep(0.1)
                continue
            
            # Obtener chunk
            start_pos = stream["position"]
            end_pos = min(start_pos + chunk_size, total_samples)
            chunk_data = audio_data[start_pos:end_pos]
            
            # Convertir a bytes
            chunk_bytes = self._audio_to_bytes(chunk_data, config)
            
            # Actualizar posición
            stream["position"] = end_pos
            
            # Actualizar estadísticas
            stats.bytes_sent += len(chunk_bytes)
            stats.chunks_sent += 1
            
            yield chunk_bytes
            
            # Control de velocidad (simular tiempo real)
            samples_sent = end_pos - start_pos
            sleep_time = samples_sent / config.sample_rate
            await asyncio.sleep(sleep_time)
    
    def _audio_to_bytes(self, audio_data: np.ndarray, config: StreamConfig) -> bytes:
        """Convierte audio numpy a bytes según formato"""
        if config.format == "wav":
            # WAV sin comprimir
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, config.sample_rate, format='WAV')
            return buffer.getvalue()
        else:
            # Por ahora solo WAV, en producción agregar mp3/ogg
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, config.sample_rate, format='WAV')
            return buffer.getvalue()
    
    def pause_stream(self, stream_id: str):
        """Pausa un stream"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id]["paused"] = True
            logger.info(f"Stream paused: {stream_id}")
    
    def resume_stream(self, stream_id: str):
        """Reanuda un stream"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id]["paused"] = False
            logger.info(f"Stream resumed: {stream_id}")
    
    def stop_stream(self, stream_id: str):
        """Detiene un stream"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id]["stopped"] = True
            logger.info(f"Stream stopped: {stream_id}")
    
    def seek_stream(self, stream_id: str, position: float):
        """
        Busca una posición en el stream
        
        Args:
            stream_id: ID del stream
            position: Posición en segundos
        """
        if stream_id in self.active_streams:
            stream = self.active_streams[stream_id]
            config = stream["config"]
            sample_position = int(position * config.sample_rate)
            stream["position"] = min(sample_position, len(stream["audio_data"]))
            logger.info(f"Stream seeked to {position}s: {stream_id}")
    
    def get_stream_stats(self, stream_id: str) -> Dict[str, Any]:
        """Obtiene estadísticas de un stream"""
        if stream_id not in self.stream_stats:
            return {}
        
        stats = self.stream_stats[stream_id]
        stream = self.active_streams.get(stream_id, {})
        
        elapsed = (datetime.now() - stats.start_time).total_seconds()
        current_bitrate = (stats.bytes_sent * 8) / elapsed if elapsed > 0 else 0
        
        return {
            "stream_id": stream_id,
            "bytes_sent": stats.bytes_sent,
            "chunks_sent": stats.chunks_sent,
            "current_bitrate": current_bitrate / 1000,  # kbps
            "elapsed_time": elapsed,
            "position": stream.get("position", 0) / stream.get("config", {}).get("sample_rate", 44100) if stream else 0,
            "paused": stream.get("paused", False),
            "dropped_chunks": stats.dropped_chunks
        }


# Instancia global
_audio_streamer: Optional[AudioStreamer] = None


def get_audio_streamer() -> AudioStreamer:
    """Obtiene la instancia global del streamer de audio"""
    global _audio_streamer
    if _audio_streamer is None:
        _audio_streamer = AudioStreamer()
    return _audio_streamer


def process_s3_event(bucket: str, key: str) -> Dict[str, Any]:
    """
    Procesa un evento de S3 (cuando se sube un archivo)
    
    Args:
        bucket: Nombre del bucket de S3
        key: Clave (path) del archivo en S3
        
    Returns:
        Resultado del procesamiento
    """
    try:
        logger.info(f"Processing S3 event: s3://{bucket}/{key}")
        
        # Verificar si es un archivo de audio
        audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a']
        if not any(key.lower().endswith(ext) for ext in audio_extensions):
            logger.info(f"File {key} is not an audio file, skipping")
            return {"status": "skipped", "reason": "not_audio_file"}
        
        # Descargar archivo de S3
        from aws.services.s3_service import S3Service
        from config.settings import settings
        
        s3 = S3Service(
            bucket_name=bucket,
            region_name=settings.aws_region
        )
        
        audio_data = s3.download_file(key)
        
        # Procesar audio (ejemplo: generar metadatos, transcripción, etc.)
        # Aquí puedes agregar lógica específica según tus necesidades
        
        result = {
            "status": "success",
            "bucket": bucket,
            "key": key,
            "size": len(audio_data),
            "processed_at": datetime.now().isoformat()
        }
        
        logger.info(f"S3 event processed successfully: {key}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing S3 event: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}
