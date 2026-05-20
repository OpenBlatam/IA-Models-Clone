"""
Tool Manager
============

Gestor de herramientas que permite al agente usar herramientas reales
del sistema de manera similar a Devin.
"""

import logging
import subprocess
import sys
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ToolInfo:
    """Información de una herramienta"""
    name: str
    description: str
    available: bool
    version: Optional[str] = None
    path: Optional[str] = None


class ToolManager:
    """
    Gestor de herramientas del sistema.
    
    Detecta y gestiona herramientas disponibles en el sistema,
    similar a cómo Devin puede usar herramientas reales.
    """
    
    def __init__(self) -> None:
        """Inicializar gestor de herramientas"""
        self.tools: Dict[str, ToolInfo] = {}
        self._detect_tools()
        logger.info(f"🔧 Tool manager initialized with {len(self.tools)} tools")
    
    def _detect_tools(self) -> None:
        """Detectar herramientas disponibles en el sistema"""
        tools_to_check = [
            ("git", "Git version control"),
            ("python", "Python interpreter"),
            ("pip", "Python package manager"),
            ("node", "Node.js runtime"),
            ("npm", "Node.js package manager"),
            ("docker", "Docker container platform"),
            ("curl", "Command-line HTTP client"),
            ("wget", "Command-line HTTP client"),
        ]
        
        for tool_name, description in tools_to_check:
            info = self._check_tool_availability(tool_name, description)
            self.tools[tool_name] = info
    
    def _check_tool_availability(
        self,
        tool_name: str,
        description: str
    ) -> ToolInfo:
        """
        Verificar disponibilidad de una herramienta.
        
        Args:
            tool_name: Nombre de la herramienta.
            description: Descripción de la herramienta.
        
        Returns:
            Información de la herramienta.
        """
        try:
            if sys.platform == "win32":
                result = subprocess.run(
                    ["where", tool_name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            else:
                result = subprocess.run(
                    ["which", tool_name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            
            if result.returncode == 0:
                path = result.stdout.strip().split('\n')[0]
                
                version = None
                try:
                    if tool_name == "git":
                        version_result = subprocess.run(
                            [tool_name, "--version"],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if version_result.returncode == 0:
                            version = version_result.stdout.strip()
                    elif tool_name == "python":
                        version_result = subprocess.run(
                            [tool_name, "--version"],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if version_result.returncode == 0:
                            version = version_result.stdout.strip()
                except Exception:
                    pass
                
                return ToolInfo(
                    name=tool_name,
                    description=description,
                    available=True,
                    version=version,
                    path=path
                )
        except Exception as e:
            logger.debug(f"Tool {tool_name} not available: {e}")
        
        return ToolInfo(
            name=tool_name,
            description=description,
            available=False
        )
    
    def is_available(self, tool_name: str) -> bool:
        """
        Verificar si una herramienta está disponible.
        
        Args:
            tool_name: Nombre de la herramienta.
        
        Returns:
            True si está disponible.
        """
        if tool_name not in self.tools:
            return False
        return self.tools[tool_name].available
    
    def get_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        """
        Obtener información de una herramienta.
        
        Args:
            tool_name: Nombre de la herramienta.
        
        Returns:
            Información de la herramienta o None.
        """
        return self.tools.get(tool_name)
    
    def get_available_tools(self) -> List[ToolInfo]:
        """
        Obtener lista de herramientas disponibles.
        
        Returns:
            Lista de herramientas disponibles.
        """
        return [tool for tool in self.tools.values() if tool.available]
    
    def get_all_tools(self) -> List[ToolInfo]:
        """
        Obtener todas las herramientas (disponibles y no disponibles).
        
        Returns:
            Lista de todas las herramientas.
        """
        return list(self.tools.values())
    
    def check_library_available(self, library_name: str) -> bool:
        """
        Verificar si una librería Python está disponible.
        
        Args:
            library_name: Nombre de la librería.
        
        Returns:
            True si está disponible.
        """
        try:
            __import__(library_name)
            return True
        except ImportError:
            return False
    
    def check_library_in_project(self, library_name: str, workspace_root: Optional[str] = None) -> bool:
        """
        Verificar si una librería está en requirements.txt o pyproject.toml.
        
        Siguiendo la regla de Devin: nunca asumir librerías, verificar primero.
        
        Args:
            library_name: Nombre de la librería.
            workspace_root: Raíz del workspace (opcional).
        
        Returns:
            True si está en el proyecto.
        """
        if not workspace_root:
            workspace_root = Path.cwd()
        else:
            workspace_root = Path(workspace_root)
        
        requirements_files = [
            workspace_root / "requirements.txt",
            workspace_root / "requirements-dev.txt",
            workspace_root / "pyproject.toml",
            workspace_root / "setup.py"
        ]
        
        for req_file in requirements_files:
            if req_file.exists():
                try:
                    content = req_file.read_text(encoding='utf-8')
                    if library_name.lower() in content.lower():
                        return True
                except Exception as e:
                    logger.debug(f"Error reading {req_file}: {e}")
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado del gestor de herramientas.
        
        Returns:
            Diccionario con estado.
        """
        available = self.get_available_tools()
        return {
            "total_tools": len(self.tools),
            "available_tools": len(available),
            "tools": {
                tool.name: {
                    "available": tool.available,
                    "version": tool.version,
                    "path": tool.path
                }
                for tool in self.tools.values()
            }
        }

