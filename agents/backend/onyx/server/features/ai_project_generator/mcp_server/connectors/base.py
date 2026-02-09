"""
Base Connector - Clase base para conectores MCP
===============================================

Clase base abstracta que define la interfaz común para todos los conectores MCP.
Proporciona métodos opcionales para health checks y gestión de recursos.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Set
from ..contracts import ContextFrame

logger = logging.getLogger(__name__)


class BaseConnector(ABC):
    """
    Clase base abstracta para todos los conectores MCP.
    
    Cada conector debe implementar:
    - execute(): ejecuta operaciones sobre el recurso
    - validate_operation(): valida si una operación es soportada
    - get_supported_operations(): retorna lista de operaciones soportadas
    
    Los conectores deben ser thread-safe si se usan en entornos concurrentes.
    """
    
    @abstractmethod
    async def execute(
        self,
        resource_id: str,
        operation: str,
        parameters: Dict[str, Any],
        context: Optional[ContextFrame] = None,
    ) -> Any:
        """
        Ejecuta una operación sobre el recurso.
        
        Args:
            resource_id: ID del recurso (debe ser string no vacío)
            operation: Operación a realizar (read, write, query, etc.) (debe ser string no vacío)
            parameters: Parámetros de la operación (debe ser dict)
            context: Frame de contexto adicional (opcional)
            
        Returns:
            Resultado de la operación (tipo depende de la operación)
            
        Raises:
            ValueError: Si los parámetros son inválidos (resource_id, operation vacíos, parameters no es dict)
            TypeError: Si los tipos de parámetros son incorrectos
            RuntimeError: Si la operación falla
            NotImplementedError: Si la operación no está implementada
        """
        pass
    
    def _validate_execute_params(
        self,
        resource_id: str,
        operation: str,
        parameters: Dict[str, Any],
    ) -> None:
        """
        Valida parámetros para execute().
        
        Args:
            resource_id: ID del recurso a validar
            operation: Operación a validar
            parameters: Parámetros a validar
            
        Raises:
            ValueError: Si resource_id o operation están vacíos
            TypeError: Si los tipos son incorrectos
        """
        if not isinstance(resource_id, str):
            raise TypeError(f"resource_id must be a string, got {type(resource_id)}")
        if not resource_id or not resource_id.strip():
            raise ValueError("resource_id cannot be empty or whitespace")
        
        if not isinstance(operation, str):
            raise TypeError(f"operation must be a string, got {type(operation)}")
        if not operation or not operation.strip():
            raise ValueError("operation cannot be empty or whitespace")
        
        if not isinstance(parameters, dict):
            raise TypeError(f"parameters must be a dictionary, got {type(parameters)}")
    
    @abstractmethod
    def validate_operation(self, operation: str) -> bool:
        """
        Valida si una operación es soportada por este conector.
        
        Args:
            operation: Nombre de la operación (debe ser string no vacío)
            
        Returns:
            True si la operación es soportada, False en caso contrario
            
        Raises:
            ValueError: Si operation está vacío
            TypeError: Si operation no es string
        """
        pass
    
    def _validate_operation_param(self, operation: str) -> None:
        """
        Valida el parámetro operation.
        
        Args:
            operation: Operación a validar
            
        Raises:
            ValueError: Si operation está vacío
            TypeError: Si operation no es string
        """
        if not isinstance(operation, str):
            raise TypeError(f"operation must be a string, got {type(operation)}")
        if not operation or not operation.strip():
            raise ValueError("operation cannot be empty or whitespace")
    
    @abstractmethod
    def get_supported_operations(self) -> List[str]:
        """
        Retorna lista de operaciones soportadas por este conector.
        
        Returns:
            Lista de nombres de operaciones soportadas
        """
        pass
    
    def get_connector_type(self) -> str:
        """
        Retorna el tipo del conector (opcional, para logging/debugging).
        
        Returns:
            Tipo del conector (default: nombre de la clase)
        """
        return self.__class__.__name__
    
    async def health_check(self) -> bool:
        """
        Verifica el estado de salud del conector (opcional).
        
        Returns:
            True si el conector está saludable, False en caso contrario
            
        Nota:
            Los conectores pueden sobrescribir este método para implementar
            verificaciones específicas (conexión a DB, API, etc.)
        """
        return True
    
    async def close(self) -> None:
        """
        Cierra recursos del conector (opcional).
        
        Nota:
            Los conectores que usan recursos (clientes HTTP, conexiones DB, etc.)
            deben sobrescribir este método para limpiar recursos.
        """
        pass

