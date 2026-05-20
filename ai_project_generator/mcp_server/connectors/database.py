"""
Database Connector - Conector para bases de datos
=================================================
"""

import logging
from typing import Any, Dict, Optional, List

from .base import BaseConnector
from ..contracts import ContextFrame
from ..exceptions import MCPConnectorError, MCPOperationError

logger = logging.getLogger(__name__)


class DatabaseConnector(BaseConnector):
    """
    Conector para acceso a bases de datos
    
    Operaciones soportadas:
    - query: ejecutar query SQL
    - execute: ejecutar comando (INSERT, UPDATE, DELETE)
    - schema: obtener esquema de tabla
    - list_tables: listar tablas disponibles
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Inicializa el conector de base de datos.
        
        Args:
            connection_string: String de conexión (opcional, puede configurarse después)
            
        Raises:
            ValueError: Si connection_string no es None y no es un string válido
        """
        if connection_string is not None:
            if not isinstance(connection_string, str) or not connection_string.strip():
                raise ValueError("connection_string must be a non-empty string or None")
            connection_string = connection_string.strip()
        
        self.connection_string = connection_string
        self._connection = None
        self._supported_operations = {
            "query", "execute", "schema", "list_tables", "describe"
        }
    
    async def execute(
        self,
        resource_id: str,
        operation: str,
        parameters: Dict[str, Any],
        context: Optional[ContextFrame] = None,
    ) -> Any:
        """
        Ejecuta operación sobre base de datos.
        
        Args:
            resource_id: ID del recurso
            operation: Operación a ejecutar
            parameters: Parámetros de la operación
            context: Frame de contexto (opcional)
            
        Returns:
            Resultado de la operación
            
        Raises:
            MCPOperationError: Si la operación no está soportada o falla
            MCPConnectorError: Si hay error del conector
            ValueError: Si los parámetros son inválidos
        """
        if not self.validate_operation(operation):
            raise MCPOperationError(f"Operation {operation} not supported by DatabaseConnector")
        
        if not self.connection_string:
            raise MCPConnectorError("Database connection not configured")
        
        if not parameters or not isinstance(parameters, dict):
            raise ValueError("parameters must be a non-empty dictionary")
        
        try:
            # Lazy connection (puede mejorarse con connection pooling)
            if not self._connection:
                await self._connect()
            
            if operation == "query":
                return await self._query(parameters)
            elif operation == "execute":
                return await self._execute_command(parameters)
            elif operation == "schema":
                return await self._get_schema(parameters)
            elif operation == "list_tables":
                return await self._list_tables(parameters)
            elif operation == "describe":
                return await self._describe_table(parameters)
            else:
                raise MCPOperationError(f"Operation {operation} not implemented")
        except (MCPOperationError, MCPConnectorError):
            raise
        except ValueError as e:
            logger.error(f"Validation error in DatabaseConnector: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in DatabaseConnector: {e}", exc_info=True)
            raise MCPConnectorError(f"Database operation failed: {e}") from e
    
    async def _connect(self):
        """Establece conexión a la base de datos"""
        # Implementación genérica - puede extenderse para diferentes DBs
        # Por ahora, solo registramos que se necesita conexión
        logger.warning("Database connection not fully implemented - use mock for testing")
        self._connection = "mock_connection"
    
    async def _query(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta query SELECT.
        
        Args:
            parameters: Debe contener 'query' con la consulta SQL
            
        Returns:
            Diccionario con rows, columns, count y query
            
        Raises:
            ValueError: Si query está ausente o no es un SELECT
        """
        query = parameters.get("query")
        if not query or not isinstance(query, str):
            raise ValueError("Query parameter must be a non-empty string")
        
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty or whitespace")
        
        # Validación básica de seguridad (solo SELECT)
        query_upper = query.upper().strip()
        if not query_upper.startswith("SELECT"):
            raise ValueError("Only SELECT queries allowed in query operation")
        
        # Implementación real dependería del driver de DB
        # Por ahora retornamos estructura esperada
        return {
            "rows": [],
            "columns": [],
            "count": 0,
            "query": query,
        }
    
    async def _execute_command(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta comando (INSERT, UPDATE, DELETE).
        
        Args:
            parameters: Debe contener 'command' con el comando SQL
            
        Returns:
            Diccionario con success, rows_affected y command
            
        Raises:
            ValueError: Si command está ausente o contiene operaciones peligrosas
        """
        command = parameters.get("command")
        if not command or not isinstance(command, str):
            raise ValueError("Command parameter must be a non-empty string")
        
        command = command.strip()
        if not command:
            raise ValueError("Command cannot be empty or whitespace")
        
        # Validación de seguridad
        command_upper = command.upper().strip()
        dangerous_ops = ["DROP", "TRUNCATE", "ALTER", "CREATE", "DELETE"]
        if any(op in command_upper for op in dangerous_ops):
            raise ValueError(f"Dangerous operation not allowed: {command_upper[:20]}")
        
        # Implementación real dependería del driver de DB
        return {
            "success": True,
            "rows_affected": 0,
            "command": command,
        }
    
    async def _get_schema(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Obtiene esquema de tabla"""
        table = parameters.get("table")
        if not table:
            raise ValueError("Table parameter required")
        
        # Implementación real dependería del driver de DB
        return {
            "table": table,
            "columns": [],
            "primary_keys": [],
            "foreign_keys": [],
        }
    
    async def _list_tables(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Lista tablas disponibles"""
        # Implementación real dependería del driver de DB
        return {
            "tables": [],
            "count": 0,
        }
    
    async def _describe_table(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Describe estructura de tabla (alias de schema)"""
        return await self._get_schema(parameters)
    
    def validate_operation(self, operation: str) -> bool:
        """Valida si operación es soportada"""
        return operation.lower() in self._supported_operations
    
    def get_supported_operations(self) -> List[str]:
        """
        Retorna operaciones soportadas.
        
        Returns:
            Lista de nombres de operaciones soportadas
        """
        return list(self._supported_operations)
    
    async def close(self) -> None:
        """
        Cierra conexión y libera recursos.
        
        Debe llamarse cuando el conector ya no se use para evitar
        conexiones abiertas.
        """
        if self._connection:
            try:
                # Implementación real de cierre dependería del driver
                if hasattr(self._connection, 'close'):
                    if hasattr(self._connection.close, '__call__'):
                        if hasattr(self._connection.close, '__code__'):
                            # Es una función, verificar si es async
                            import asyncio
                            if asyncio.iscoroutinefunction(self._connection.close):
                                await self._connection.close()
                            else:
                                self._connection.close()
            except Exception as e:
                logger.warning(f"Error closing database connection: {e}")
            finally:
                self._connection = None

