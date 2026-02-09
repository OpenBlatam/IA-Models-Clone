"""
Response Optimizer
Optimización de respuestas HTTP
"""

import logging
from typing import Any, Dict
from fastapi.responses import Response
import orjson

logger = logging.getLogger(__name__)


class ResponseOptimizer:
    """Optimizador de respuestas HTTP"""
    
    @staticmethod
    def optimize_json_response(data: Any) -> bytes:
        """
        Optimiza respuesta JSON usando orjson
        
        Args:
            data: Datos a serializar
            
        Returns:
            Bytes serializados
        """
        try:
            return orjson.dumps(
                data,
                option=orjson.OPT_SERIALIZE_NUMPY | 
                       orjson.OPT_SERIALIZE_DATACLASS |
                       orjson.OPT_NON_STR_KEYS
            )
        except Exception as e:
            logger.warning(f"orjson failed, falling back to json: {e}")
            import json
            return json.dumps(data).encode()
    
    @staticmethod
    def create_optimized_response(
        data: Any,
        status_code: int = 200,
        headers: Dict[str, str] = None
    ) -> Response:
        """
        Crea respuesta optimizada
        
        Args:
            data: Datos de respuesta
            status_code: Código de estado
            headers: Headers adicionales
            
        Returns:
            Response optimizada
        """
        content = ResponseOptimizer.optimize_json_response(data)
        
        response_headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(content)),
            **{k: v for k, v in (headers or {}).items()}
        }
        
        return Response(
            content=content,
            status_code=status_code,
            headers=response_headers,
            media_type="application/json"
        )
    
    @staticmethod
    def paginate_response(
        items: list,
        page: int,
        page_size: int,
        total: int = None
    ) -> Dict[str, Any]:
        """
        Crea respuesta paginada optimizada
        
        Args:
            items: Items de la página
            page: Número de página
            page_size: Tamaño de página
            total: Total de items (opcional)
            
        Returns:
            Respuesta paginada
        """
        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total": total or len(items),
            "has_next": total is None or (page * page_size) < total,
            "has_prev": page > 1
        }
