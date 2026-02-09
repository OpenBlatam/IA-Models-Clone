"""
Scope Service - Determines required scopes for operations
==========================================================

Servicio para determinar el scope requerido para diferentes operaciones,
incluyendo validación y clasificación de operaciones.
"""

import logging
from typing import Set
from ..security import Scope

logger = logging.getLogger(__name__)


class ScopeService:
    """
    Service for determining required scopes for operations.
    
    Proporciona métodos estáticos para clasificar operaciones
    y determinar el scope de seguridad requerido.
    """
    
    # Operaciones de lectura (no modifican datos)
    READ_OPERATIONS: Set[str] = {
        "read", "query", "list", "get", "search", "fetch",
        "retrieve", "find", "select", "read_all", "count"
    }
    
    # Operaciones de escritura (modifican datos)
    WRITE_OPERATIONS: Set[str] = {
        "write", "create", "update", "delete", "modify", "remove",
        "post", "put", "patch", "insert", "upsert", "replace",
        "edit", "save", "add", "append", "clear", "truncate"
    }
    
    @staticmethod
    def get_scope_for_operation(operation: str) -> Scope:
        """
        Determina el scope requerido para una operación.
        
        Args:
            operation: Nombre de la operación
            
        Returns:
            Scope requerido (READ por defecto para seguridad)
            
        Raises:
            ValueError: Si operation es inválido
        """
        if not operation or not isinstance(operation, str):
            raise ValueError("operation must be a non-empty string")
        
        operation_lower = operation.lower().strip()
        
        if operation_lower in ScopeService.READ_OPERATIONS:
            return Scope.READ
        elif operation_lower in ScopeService.WRITE_OPERATIONS:
            return Scope.WRITE
        else:
            # Default seguro: asumir operación de lectura
            logger.debug(
                f"Unknown operation '{operation}', defaulting to READ scope for safety"
            )
            return Scope.READ
    
    @staticmethod
    def is_read_operation(operation: str) -> bool:
        """
        Verifica si una operación es de solo lectura.
        
        Args:
            operation: Nombre de la operación
            
        Returns:
            True si es operación de lectura, False en caso contrario
        """
        if not operation or not isinstance(operation, str):
            return False
        
        return operation.lower().strip() in ScopeService.READ_OPERATIONS
    
    @staticmethod
    def is_write_operation(operation: str) -> bool:
        """
        Verifica si una operación modifica datos.
        
        Args:
            operation: Nombre de la operación
            
        Returns:
            True si es operación de escritura, False en caso contrario
        """
        if not operation or not isinstance(operation, str):
            return False
        
        return operation.lower().strip() in ScopeService.WRITE_OPERATIONS
    
    @staticmethod
    def get_all_read_operations() -> Set[str]:
        """
        Retorna todas las operaciones de lectura conocidas.
        
        Returns:
            Set de nombres de operaciones de lectura
        """
        return ScopeService.READ_OPERATIONS.copy()
    
    @staticmethod
    def get_all_write_operations() -> Set[str]:
        """
        Retorna todas las operaciones de escritura conocidas.
        
        Returns:
            Set de nombres de operaciones de escritura
        """
        return ScopeService.WRITE_OPERATIONS.copy()
    
    @staticmethod
    def add_read_operation(operation: str) -> None:
        """
        Agrega una operación a la lista de operaciones de lectura.
        
        Args:
            operation: Nombre de la operación a agregar
        """
        if operation and isinstance(operation, str):
            ScopeService.READ_OPERATIONS.add(operation.lower().strip())
            logger.debug(f"Added read operation: {operation}")
    
    @staticmethod
    def add_write_operation(operation: str) -> None:
        """
        Agrega una operación a la lista de operaciones de escritura.
        
        Args:
            operation: Nombre de la operación a agregar
        """
        if operation and isinstance(operation, str):
            ScopeService.WRITE_OPERATIONS.add(operation.lower().strip())
            logger.debug(f"Added write operation: {operation}")

