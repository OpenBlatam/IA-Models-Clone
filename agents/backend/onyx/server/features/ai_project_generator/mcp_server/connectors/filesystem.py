"""
File System Connector - Conector para sistema de archivos
==========================================================
"""

import os
import logging
import aiofiles
from pathlib import Path
from typing import Any, Dict, Optional, List

from .base import BaseConnector
from ..contracts import ContextFrame
from ..exceptions import MCPConnectorError, MCPOperationError

logger = logging.getLogger(__name__)


class FileSystemConnector(BaseConnector):
    """
    Conector para acceso al sistema de archivos
    
    Operaciones soportadas:
    - read: leer contenido de archivo
    - write: escribir contenido a archivo
    - list: listar directorio
    - exists: verificar existencia
    - metadata: obtener metadatos de archivo
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Inicializa el conector de sistema de archivos
        
        Args:
            base_path: Ruta base para operaciones (opcional, para sandboxing)
        """
        self.base_path = Path(base_path) if base_path else None
        self._supported_operations = {
            "read", "write", "list", "exists", "metadata", "search"
        }
    
    def _resolve_path(self, path: str) -> Path:
        """
        Resuelve y valida una ruta.
        
        Args:
            path: Ruta a resolver
            
        Returns:
            Path resuelto y validado
            
        Raises:
            ValueError: Si la ruta está fuera del sandbox permitido
        """
        if not path or not isinstance(path, str):
            raise ValueError("path must be a non-empty string")
        
        resolved = Path(path)
        
        # Si hay base_path, asegurar que está dentro del sandbox
        if self.base_path:
            resolved = (self.base_path / resolved).resolve()
            base_resolved = self.base_path.resolve()
            if not str(resolved).startswith(str(base_resolved)):
                raise ValueError(f"Path {path} is outside allowed base path {base_resolved}")
        
        return resolved
    
    async def execute(
        self,
        resource_id: str,
        operation: str,
        parameters: Dict[str, Any],
        context: Optional[ContextFrame] = None,
    ) -> Any:
        """
        Ejecuta operación sobre sistema de archivos.
        
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
            raise MCPOperationError(f"Operation {operation} not supported by FileSystemConnector")
        
        if not parameters or not isinstance(parameters, dict):
            raise ValueError("parameters must be a non-empty dictionary")
        
        try:
            path = parameters.get("path", resource_id)
            resolved_path = self._resolve_path(path)
            
            if operation == "read":
                return await self._read_file(resolved_path, parameters)
            elif operation == "write":
                return await self._write_file(resolved_path, parameters)
            elif operation == "list":
                return await self._list_directory(resolved_path, parameters)
            elif operation == "exists":
                return {"exists": resolved_path.exists()}
            elif operation == "metadata":
                return await self._get_metadata(resolved_path)
            elif operation == "search":
                return await self._search_files(resolved_path, parameters)
            else:
                raise MCPOperationError(f"Operation {operation} not implemented")
        except (ValueError, FileNotFoundError, PermissionError) as e:
            logger.error(f"Error executing {operation} on {resource_id}: {e}")
            raise MCPConnectorError(f"File system operation failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in FileSystemConnector: {e}", exc_info=True)
            raise MCPConnectorError(f"Unexpected error: {e}") from e
    
    async def _read_file(self, path: Path, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lee contenido de archivo.
        
        Args:
            path: Ruta del archivo
            parameters: Parámetros con encoding y max_size
            
        Returns:
            Diccionario con contenido y metadatos
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el path no es un archivo o es muy grande
            PermissionError: Si no hay permisos de lectura
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        
        encoding = parameters.get("encoding", "utf-8")
        if not isinstance(encoding, str):
            encoding = "utf-8"
        
        max_size = parameters.get("max_size", 10 * 1024 * 1024)  # 10MB default
        if not isinstance(max_size, int) or max_size <= 0:
            max_size = 10 * 1024 * 1024
        
        try:
            file_size = path.stat().st_size
        except OSError as e:
            raise PermissionError(f"Cannot access file {path}: {e}") from e
        
        if file_size > max_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {max_size})")
        
        try:
            async with aiofiles.open(path, "r", encoding=encoding) as f:
                content = await f.read()
        except UnicodeDecodeError as e:
            raise ValueError(f"Cannot decode file with encoding {encoding}: {e}") from e
        except PermissionError as e:
            raise PermissionError(f"Permission denied reading {path}: {e}") from e
        
        return {
            "content": content,
            "path": str(path),
            "size": file_size,
            "encoding": encoding,
        }
    
    async def _write_file(self, path: Path, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Escribe contenido a archivo"""
        content = parameters.get("content", "")
        encoding = parameters.get("encoding", "utf-8")
        create_dirs = parameters.get("create_dirs", True)
        
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(path, "w", encoding=encoding) as f:
            await f.write(content)
        
        return {
            "success": True,
            "path": str(path),
            "size": len(content.encode(encoding)),
        }
    
    async def _list_directory(self, path: Path, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Lista contenido de directorio"""
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        
        recursive = parameters.get("recursive", False)
        pattern = parameters.get("pattern", "*")
        
        items = []
        if recursive:
            for item in path.rglob(pattern):
                items.append({
                    "name": item.name,
                    "path": str(item.relative_to(path)),
                    "type": "file" if item.is_file() else "directory",
                    "size": item.stat().st_size if item.is_file() else None,
                })
        else:
            for item in path.glob(pattern):
                items.append({
                    "name": item.name,
                    "path": str(item.relative_to(path)),
                    "type": "file" if item.is_file() else "directory",
                    "size": item.stat().st_size if item.is_file() else None,
                })
        
        return {
            "items": items,
            "count": len(items),
            "path": str(path),
        }
    
    async def _get_metadata(self, path: Path) -> Dict[str, Any]:
        """Obtiene metadatos de archivo/directorio"""
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        stat = path.stat()
        return {
            "path": str(path),
            "exists": True,
            "type": "file" if path.is_file() else "directory",
            "size": stat.st_size if path.is_file() else None,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "permissions": oct(stat.st_mode)[-3:],
        }
    
    async def _search_files(self, path: Path, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Busca archivos por patrón o contenido"""
        pattern = parameters.get("pattern", "*")
        content_search = parameters.get("content_search")
        recursive = parameters.get("recursive", True)
        
        matches = []
        search_path = path if path.is_dir() else path.parent
        
        for item in search_path.rglob(pattern) if recursive else search_path.glob(pattern):
            if item.is_file():
                if content_search:
                    # Búsqueda simple en contenido (puede mejorarse)
                    try:
                        async with aiofiles.open(item, "r", encoding="utf-8") as f:
                            content = await f.read()
                            if content_search.lower() in content.lower():
                                matches.append(str(item))
                    except Exception:
                        pass
                else:
                    matches.append(str(item))
        
        return {
            "matches": matches,
            "count": len(matches),
            "pattern": pattern,
        }
    
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

