"""
MCP Connectors - Conectores para recursos
==========================================

Conectores que implementan acceso a diferentes tipos de recursos:
- FileSystemConnector: acceso a sistema de archivos
- DatabaseConnector: acceso a bases de datos
- APIConnector: acceso a APIs externas
"""

from .base import BaseConnector
from .filesystem import FileSystemConnector
from .database import DatabaseConnector
from .api import APIConnector
from .registry import ConnectorRegistry

__all__ = [
    "BaseConnector",
    "FileSystemConnector",
    "DatabaseConnector",
    "APIConnector",
    "ConnectorRegistry",
]

