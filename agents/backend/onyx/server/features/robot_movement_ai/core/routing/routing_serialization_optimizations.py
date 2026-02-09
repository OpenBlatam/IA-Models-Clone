"""
Routing Serialization Optimizations
===================================

Optimizaciones de serialización y deserialización.
Incluye: Fast serialization, Compression, Protocol buffers, etc.
"""

import logging
import pickle
import json
import gzip
import time
from typing import Dict, Any, Optional
import threading

logger = logging.getLogger(__name__)

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    logger.warning("msgpack not available, using JSON fallback")

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False


class FastSerializer:
    """Serializador rápido con múltiples formatos."""
    
    def __init__(self, format: str = "auto"):
        """
        Inicializar serializador.
        
        Args:
            format: Formato ('auto', 'json', 'msgpack', 'pickle')
        """
        self.format = format
        self._determine_best_format()
    
    def _determine_best_format(self):
        """Determinar mejor formato disponible."""
        if self.format == "auto":
            if MSGPACK_AVAILABLE:
                self.format = "msgpack"
            elif ORJSON_AVAILABLE:
                self.format = "orjson"
            else:
                self.format = "json"
    
    def serialize(self, data: Any) -> bytes:
        """
        Serializar datos.
        
        Args:
            data: Datos a serializar
        
        Returns:
            Datos serializados
        """
        if self.format == "msgpack" and MSGPACK_AVAILABLE:
            return msgpack.packb(data)
        elif self.format == "orjson" and ORJSON_AVAILABLE:
            return orjson.dumps(data)
        elif self.format == "pickle":
            return pickle.dumps(data)
        else:
            return json.dumps(data).encode('utf-8')
    
    def deserialize(self, data: bytes) -> Any:
        """
        Deserializar datos.
        
        Args:
            data: Datos serializados
        
        Returns:
            Datos deserializados
        """
        if self.format == "msgpack" and MSGPACK_AVAILABLE:
            return msgpack.unpackb(data, raw=False)
        elif self.format == "orjson" and ORJSON_AVAILABLE:
            return orjson.loads(data)
        elif self.format == "pickle":
            return pickle.loads(data)
        else:
            return json.loads(data.decode('utf-8'))


class CompressedSerializer:
    """Serializador con compresión."""
    
    def __init__(self, serializer: FastSerializer, compression_level: int = 6):
        """
        Inicializar serializador comprimido.
        
        Args:
            serializer: Serializador base
            compression_level: Nivel de compresión (1-9)
        """
        self.serializer = serializer
        self.compression_level = compression_level
    
    def serialize(self, data: Any) -> bytes:
        """Serializar y comprimir datos."""
        serialized = self.serializer.serialize(data)
        return gzip.compress(serialized, compresslevel=self.compression_level)
    
    def deserialize(self, data: bytes) -> Any:
        """Descomprimir y deserializar datos."""
        decompressed = gzip.decompress(data)
        return self.serializer.deserialize(decompressed)


class SerializationOptimizer:
    """Optimizador completo de serialización."""
    
    def __init__(self, format: str = "auto", use_compression: bool = False):
        """
        Inicializar optimizador de serialización.
        
        Args:
            format: Formato de serialización
            use_compression: Usar compresión
        """
        self.serializer = FastSerializer(format)
        
        if use_compression:
            self.serializer = CompressedSerializer(self.serializer)
        
        self.stats = {
            'serializations': 0,
            'deserializations': 0,
            'total_serialization_time': 0.0,
            'total_deserialization_time': 0.0
        }
        self.lock = threading.Lock()
    
    def serialize(self, data: Any) -> bytes:
        """Serializar datos con tracking."""
        start_time = time.time()
        result = self.serializer.serialize(data)
        duration = time.time() - start_time
        
        with self.lock:
            self.stats['serializations'] += 1
            self.stats['total_serialization_time'] += duration
        
        return result
    
    def deserialize(self, data: bytes) -> Any:
        """Deserializar datos con tracking."""
        start_time = time.time()
        result = self.serializer.deserialize(data)
        duration = time.time() - start_time
        
        with self.lock:
            self.stats['deserializations'] += 1
            self.stats['total_deserialization_time'] += duration
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        with self.lock:
            stats = self.stats.copy()
            if stats['serializations'] > 0:
                stats['avg_serialization_time'] = (
                    stats['total_serialization_time'] / stats['serializations']
                )
            if stats['deserializations'] > 0:
                stats['avg_deserialization_time'] = (
                    stats['total_deserialization_time'] / stats['deserializations']
                )
            return stats

