"""
Serialization Optimizer
Optimización de serialización de datos
"""

import logging
from typing import Any, Dict, List, Optional
import orjson

logger = logging.getLogger(__name__)


class SerializationOptimizer:
    """Optimizador de serialización"""
    
    # Opciones optimizadas de orjson
    ORJSON_OPTIONS = (
        orjson.OPT_SERIALIZE_NUMPY |
        orjson.OPT_SERIALIZE_DATACLASS |
        orjson.OPT_NON_STR_KEYS |
        orjson.OPT_OMIT_MICROSECONDS
    )
    
    @classmethod
    def serialize(cls, data: Any) -> bytes:
        """
        Serializa datos de forma optimizada
        
        Args:
            data: Datos a serializar
            
        Returns:
            Bytes serializados
        """
        try:
            return orjson.dumps(data, option=cls.ORJSON_OPTIONS)
        except (TypeError, ValueError) as e:
            logger.warning(f"orjson serialization failed: {e}, using fallback")
            # Fallback a json estándar
            import json
            return json.dumps(data, default=str).encode()
    
    @classmethod
    def serialize_list(cls, items: List[Any]) -> bytes:
        """
        Serializa lista de forma optimizada
        
        Args:
            items: Lista de items
            
        Returns:
            Bytes serializados
        """
        return cls.serialize(items)
    
    @classmethod
    def serialize_dict(cls, data: Dict[str, Any]) -> bytes:
        """
        Serializa diccionario de forma optimizada
        
        Args:
            data: Diccionario
            
        Returns:
            Bytes serializados
        """
        return cls.serialize(data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> Any:
        """
        Deserializa datos
        
        Args:
            data: Bytes a deserializar
            
        Returns:
            Datos deserializados
        """
        try:
            return orjson.loads(data)
        except Exception as e:
            logger.warning(f"orjson deserialization failed: {e}")
            import json
            return json.loads(data.decode())


# Instancia global
_serialization_optimizer: Optional[SerializationOptimizer] = None


def get_serialization_optimizer() -> SerializationOptimizer:
    """Obtiene el optimizador de serialización"""
    global _serialization_optimizer
    if _serialization_optimizer is None:
        _serialization_optimizer = SerializationOptimizer()
    return _serialization_optimizer

