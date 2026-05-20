"""
Connection Manager (Legacy)
===========================

Este archivo mantiene compatibilidad hacia atrás.
Nuevo código debe usar core.connection.connection_manager.ConnectionManager
"""

from .connection.connection_manager import ConnectionManager, get_connection_manager

__all__ = ["ConnectionManager", "get_connection_manager"]
