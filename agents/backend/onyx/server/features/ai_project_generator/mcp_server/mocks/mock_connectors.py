"""
Mock Connectors - Conectores mock para testing
===============================================
"""

from typing import Any, Dict, Optional
from ..connectors.base import BaseConnector
from ..contracts import ContextFrame


class MockFileSystemConnector(BaseConnector):
    """Mock del conector de sistema de archivos"""
    
    def __init__(self):
        self._files: Dict[str, str] = {}
        self._directories: Dict[str, list[str]] = {}
        self._supported_operations = {"read", "write", "list", "exists", "metadata", "search"}
    
    async def execute(
        self,
        resource_id: str,
        operation: str,
        parameters: Dict[str, Any],
        context: Optional[ContextFrame] = None,
    ) -> Any:
        """Ejecuta operación mock"""
        path = parameters.get("path", resource_id)
        
        if operation == "read":
            if path not in self._files:
                raise FileNotFoundError(f"File not found: {path}")
            return {
                "content": self._files[path],
                "path": path,
                "size": len(self._files[path]),
                "encoding": "utf-8",
            }
        elif operation == "write":
            content = parameters.get("content", "")
            self._files[path] = content
            return {"success": True, "path": path, "size": len(content)}
        elif operation == "list":
            items = self._directories.get(path, [])
            return {"items": items, "count": len(items), "path": path}
        elif operation == "exists":
            return {"exists": path in self._files or path in self._directories}
        elif operation == "metadata":
            if path in self._files:
                return {
                    "path": path,
                    "exists": True,
                    "type": "file",
                    "size": len(self._files[path]),
                }
            elif path in self._directories:
                return {
                    "path": path,
                    "exists": True,
                    "type": "directory",
                }
            else:
                raise FileNotFoundError(f"Path not found: {path}")
        elif operation == "search":
            pattern = parameters.get("pattern", "*")
            matches = [p for p in self._files.keys() if pattern in p]
            return {"matches": matches, "count": len(matches), "pattern": pattern}
        else:
            raise ValueError(f"Operation {operation} not supported")
    
    def validate_operation(self, operation: str) -> bool:
        return operation.lower() in self._supported_operations
    
    def get_supported_operations(self) -> list[str]:
        return list(self._supported_operations)
    
    def add_file(self, path: str, content: str):
        """Helper para agregar archivo al mock"""
        self._files[path] = content
    
    def add_directory(self, path: str, items: list[str]):
        """Helper para agregar directorio al mock"""
        self._directories[path] = items


class MockDatabaseConnector(BaseConnector):
    """Mock del conector de base de datos"""
    
    def __init__(self):
        self._tables: Dict[str, list[Dict[str, Any]]] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._supported_operations = {"query", "execute", "schema", "list_tables", "describe"}
    
    async def execute(
        self,
        resource_id: str,
        operation: str,
        parameters: Dict[str, Any],
        context: Optional[ContextFrame] = None,
    ) -> Any:
        """Ejecuta operación mock"""
        if operation == "query":
            query = parameters.get("query", "")
            # Mock simple: retornar datos de tabla si existe
            table = parameters.get("table", "default")
            rows = self._tables.get(table, [])
            return {"rows": rows, "columns": list(rows[0].keys()) if rows else [], "count": len(rows)}
        elif operation == "execute":
            return {"success": True, "rows_affected": 1}
        elif operation == "schema":
            table = parameters.get("table", "default")
            return self._schemas.get(table, {"table": table, "columns": []})
        elif operation == "list_tables":
            return {"tables": list(self._tables.keys()), "count": len(self._tables)}
        elif operation == "describe":
            table = parameters.get("table", "default")
            return self._schemas.get(table, {"table": table, "columns": []})
        else:
            raise ValueError(f"Operation {operation} not supported")
    
    def validate_operation(self, operation: str) -> bool:
        return operation.lower() in self._supported_operations
    
    def get_supported_operations(self) -> list[str]:
        return list(self._supported_operations)
    
    def add_table(self, name: str, rows: list[Dict[str, Any]], schema: Optional[Dict[str, Any]] = None):
        """Helper para agregar tabla al mock"""
        self._tables[name] = rows
        if schema:
            self._schemas[name] = schema


class MockAPIConnector(BaseConnector):
    """Mock del conector de API"""
    
    def __init__(self):
        self._responses: Dict[str, Dict[str, Any]] = {}
        self._supported_operations = {"get", "post", "put", "delete", "patch", "request"}
    
    async def execute(
        self,
        resource_id: str,
        operation: str,
        parameters: Dict[str, Any],
        context: Optional[ContextFrame] = None,
    ) -> Any:
        """Ejecuta operación mock"""
        url = parameters.get("url", resource_id)
        
        # Buscar respuesta mock
        response_key = f"{operation.upper()}:{url}"
        if response_key in self._responses:
            return self._responses[response_key]
        
        # Respuesta por defecto
        return {
            "status_code": 200,
            "headers": {},
            "content": {"message": "Mock response"},
            "url": url,
        }
    
    def validate_operation(self, operation: str) -> bool:
        return operation.lower() in self._supported_operations
    
    def get_supported_operations(self) -> list[str]:
        return list(self._supported_operations)
    
    def set_response(self, method: str, url: str, response: Dict[str, Any]):
        """Helper para configurar respuesta mock"""
        self._responses[f"{method.upper()}:{url}"] = response

