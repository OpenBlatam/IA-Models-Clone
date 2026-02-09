"""
Serialization Optimizations - Optimizaciones de serialización
=============================================================

Optimizaciones para serialización JSON y otros formatos.
"""

import structlog
from typing import Any, Dict
import orjson
from fastapi.responses import ORJSONResponse

logger = structlog.get_logger(__name__)


class FastJSONSerializer:
    """Serializador JSON rápido usando orjson"""
    
    @staticmethod
    def dumps(obj: Any, **kwargs) -> bytes:
        """
        Serializa objeto a JSON.
        
        Args:
            obj: Objeto a serializar
            **kwargs: Opciones adicionales
        
        Returns:
            JSON como bytes
        """
        # orjson es más rápido que json estándar
        return orjson.dumps(
            obj,
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS
        )
    
    @staticmethod
    def loads(data: bytes) -> Any:
        """
        Deserializa JSON.
        
        Args:
            data: JSON como bytes
        
        Returns:
            Objeto deserializado
        """
        return orjson.loads(data)


class ResponseSerializer:
    """Serializador optimizado de respuestas"""
    
    def __init__(self):
        self.serializer = FastJSONSerializer()
    
    def serialize_response(self, data: Any) -> bytes:
        """
        Serializa respuesta.
        
        Args:
            data: Datos a serializar
        
        Returns:
            Datos serializados
        """
        # Optimizar datos antes de serializar
        optimized = self._optimize_data(data)
        return self.serializer.dumps(optimized)
    
    def _optimize_data(self, data: Any) -> Any:
        """
        Optimiza datos antes de serializar.
        
        Args:
            data: Datos originales
        
        Returns:
            Datos optimizados
        """
        if isinstance(data, dict):
            # Remover None values
            return {k: self._optimize_data(v) for k, v in data.items() if v is not None}
        elif isinstance(data, list):
            return [self._optimize_data(item) for item in data]
        else:
            return data


def enable_fast_json(app):
    """Habilita serialización JSON rápida en la app"""
    app.default_response_class = ORJSONResponse
    logger.info("Fast JSON serialization enabled")




