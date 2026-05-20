"""
MCP Advanced Transforms - Transformaciones avanzadas
=====================================================
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class TransformPipeline:
    """
    Pipeline de transformaciones
    
    Aplica múltiples transformaciones en secuencia.
    """
    
    def __init__(self):
        self._transforms: List[Callable] = []
    
    def add_transform(self, transform: Callable):
        """
        Agrega transformación al pipeline
        
        Args:
            transform: Función de transformación
        """
        self._transforms.append(transform)
        logger.info(f"Added transform to pipeline: {transform.__name__}")
    
    def apply(self, data: Any) -> Any:
        """
        Aplica todas las transformaciones
        
        Args:
            data: Datos a transformar
            
        Returns:
            Datos transformados
        """
        result = data
        for transform in self._transforms:
            try:
                if isinstance(result, dict):
                    result = transform(result)
                else:
                    result = transform(result)
            except Exception as e:
                logger.error(f"Error in transform {transform.__name__}: {e}")
        return result


class AdvancedRequestTransformer:
    """
    Transformador avanzado de requests
    
    Proporciona transformaciones complejas de requests.
    """
    
    def __init__(self):
        self._pipelines: Dict[str, TransformPipeline] = {}
    
    def register_pipeline(self, name: str, pipeline: TransformPipeline):
        """
        Registra pipeline de transformaciones
        
        Args:
            name: Nombre del pipeline
            pipeline: Pipeline de transformaciones
        """
        self._pipelines[name] = pipeline
        logger.info(f"Registered transform pipeline: {name}")
    
    def transform_request(
        self,
        request: Dict[str, Any],
        pipeline_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transforma un request
        
        Args:
            request: Request a transformar
            pipeline_name: Nombre del pipeline (opcional)
            
        Returns:
            Request transformado
        """
        if pipeline_name:
            pipeline = self._pipelines.get(pipeline_name)
            if pipeline:
                return pipeline.apply(request)
        
        return request


class AdvancedResponseTransformer:
    """
    Transformador avanzado de responses
    
    Proporciona transformaciones complejas de responses.
    """
    
    def __init__(self):
        self._pipelines: Dict[str, TransformPipeline] = {}
    
    def register_pipeline(self, name: str, pipeline: TransformPipeline):
        """
        Registra pipeline de transformaciones
        
        Args:
            name: Nombre del pipeline
            pipeline: Pipeline de transformaciones
        """
        self._pipelines[name] = pipeline
        logger.info(f"Registered transform pipeline: {name}")
    
    def transform_response(
        self,
        response: Dict[str, Any],
        pipeline_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transforma un response
        
        Args:
            response: Response a transformar
            pipeline_name: Nombre del pipeline (opcional)
            
        Returns:
            Response transformado
        """
        if pipeline_name:
            pipeline = self._pipelines.get(pipeline_name)
            if pipeline:
                return pipeline.apply(response)
        
        return response


# Transformaciones comunes

def add_correlation_id(data: Dict[str, Any]) -> Dict[str, Any]:
    """Agrega correlation ID"""
    import uuid
    if "correlation_id" not in data:
        data["correlation_id"] = str(uuid.uuid4())
    return data


def add_timestamps(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agrega timestamps timezone-aware a los datos.
    
    Args:
        data: Diccionario de datos
        
    Returns:
        Diccionario con timestamps agregados
    """
    from datetime import timezone
    now = datetime.now(timezone.utc).isoformat()
    data["timestamp"] = now
    data["processed_at"] = now
    return data


def normalize_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normaliza keys del diccionario"""
    return {k.lower().replace("_", "-"): v for k, v in data.items()}


def filter_sensitive_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Filtra campos sensibles"""
    sensitive = ["password", "token", "secret", "api_key"]
    return {
        k: "***" if any(s in k.lower() for s in sensitive) else v
        for k, v in data.items()
    }

