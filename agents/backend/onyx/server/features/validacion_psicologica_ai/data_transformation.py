"""
Sistema de Transformación de Datos
====================================
Transformación y normalización de datos
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import structlog
import json

logger = structlog.get_logger()


class DataTransformer:
    """Transformador de datos"""
    
    def __init__(self):
        """Inicializar transformador"""
        self._transformers: Dict[str, Callable] = {}
        self._load_default_transformers()
        logger.info("DataTransformer initialized")
    
    def _load_default_transformers(self) -> None:
        """Cargar transformadores por defecto"""
        # Normalizar texto
        self._transformers["normalize_text"] = lambda x: (
            str(x).strip().lower() if isinstance(x, str) else x
        )
        
        # Normalizar fecha
        self._transformers["normalize_date"] = lambda x: (
            datetime.fromisoformat(str(x)) if isinstance(x, str) else x
        )
        
        # Normalizar número
        self._transformers["normalize_number"] = lambda x: (
            float(x) if isinstance(x, (int, str)) and str(x).replace('.', '').isdigit() else x
        )
        
        # Sanitizar HTML
        self._transformers["sanitize_html"] = lambda x: (
            str(x).replace('<', '&lt;').replace('>', '&gt;') if isinstance(x, str) else x
        )
    
    def register_transformer(
        self,
        name: str,
        transformer: Callable
    ) -> None:
        """
        Registrar transformador
        
        Args:
            name: Nombre del transformador
            transformer: Función transformadora
        """
        self._transformers[name] = transformer
        logger.info("Transformer registered", name=name)
    
    def transform(
        self,
        data: Any,
        transformer_name: str
    ) -> Any:
        """
        Transformar dato
        
        Args:
            data: Dato a transformar
            transformer_name: Nombre del transformador
            
        Returns:
            Dato transformado
        """
        transformer = self._transformers.get(transformer_name)
        if not transformer:
            raise ValueError(f"Transformer {transformer_name} not found")
        
        try:
            return transformer(data)
        except Exception as e:
            logger.error("Error transforming data", transformer=transformer_name, error=str(e))
            return data
    
    def transform_dict(
        self,
        data: Dict[str, Any],
        transformations: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Transformar diccionario
        
        Args:
            data: Diccionario a transformar
            transformations: Mapeo de campo -> transformador
            
        Returns:
            Diccionario transformado
        """
        result = {}
        
        for field, value in data.items():
            transformer_name = transformations.get(field)
            if transformer_name:
                result[field] = self.transform(value, transformer_name)
            else:
                result[field] = value
        
        return result
    
    def normalize_validation_data(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Normalizar datos de validación
        
        Args:
            data: Datos a normalizar
            
        Returns:
            Datos normalizados
        """
        transformations = {
            "user_id": "normalize_text",
            "platforms": lambda x: [str(p).lower() for p in x] if isinstance(x, list) else x,
            "metadata": lambda x: json.dumps(x) if isinstance(x, dict) else x
        }
        
        return self.transform_dict(data, transformations)
    
    def denormalize_validation_data(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Desnormalizar datos de validación
        
        Args:
            data: Datos a desnormalizar
            
        Returns:
            Datos desnormalizados
        """
        result = data.copy()
        
        # Desnormalizar metadata
        if "metadata" in result and isinstance(result["metadata"], str):
            try:
                result["metadata"] = json.loads(result["metadata"])
            except:
                pass
        
        return result


# Instancia global del transformador
data_transformer = DataTransformer()




