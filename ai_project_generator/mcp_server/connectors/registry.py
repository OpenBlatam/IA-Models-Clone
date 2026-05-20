"""
Connector Registry - Registro de conectores disponibles
========================================================

Registro centralizado y thread-safe para gestionar conectores MCP.
Permite registrar, obtener, y gestionar conectores de forma segura
en entornos concurrentes.
"""

import logging
import threading
from typing import Dict, Optional, List, Set

from .base import BaseConnector

logger = logging.getLogger(__name__)


class ConnectorRegistry:
    """
    Registro centralizado de conectores MCP (thread-safe).
    
    Permite registrar y obtener conectores por tipo de forma segura
    en entornos concurrentes. Utiliza RLock para garantizar thread-safety
    en todas las operaciones.
    
    Attributes:
        _connectors: Diccionario que mapea tipos de conectores a instancias.
        _lock: Lock reentrante para thread-safety.
    """
    
    def __init__(self) -> None:
        """
        Inicializa el registry de conectores.
        
        Crea un diccionario vacío para almacenar conectores y un lock
        reentrante para garantizar thread-safety.
        """
        self._connectors: Dict[str, BaseConnector] = {}
        self._lock = threading.RLock()  # Thread safety
        logger.debug("ConnectorRegistry initialized")
    
    def register(self, connector_type: str, connector: BaseConnector) -> None:
        """
        Registra un conector (thread-safe).
        
        Args:
            connector_type: Tipo del conector (ej: "filesystem", "database", "api").
            connector: Instancia del conector a registrar.
        
        Raises:
            ValueError: Si connector_type es inválido o connector es None.
            TypeError: Si connector no es una instancia de BaseConnector.
        """
        if not connector_type or not isinstance(connector_type, str):
            raise ValueError("connector_type must be a non-empty string")
        if connector is None:
            raise ValueError("connector cannot be None")
        if not isinstance(connector, BaseConnector):
            raise TypeError(
                f"connector must be an instance of BaseConnector, "
                f"got {type(connector)}"
            )
        
        connector_type_lower = connector_type.lower().strip()
        with self._lock:
            if connector_type_lower in self._connectors:
                logger.warning(
                    f"Overwriting existing connector: {connector_type_lower}"
                )
            self._connectors[connector_type_lower] = connector
            logger.info(f"Registered connector: {connector_type_lower}")
    
    def get(self, connector_type: str) -> Optional[BaseConnector]:
        """
        Obtiene un conector por tipo (thread-safe).
        
        Args:
            connector_type: Tipo del conector a obtener.
        
        Returns:
            Conector si existe, None en caso contrario.
        """
        if not connector_type:
            return None
        
        connector_type_lower = connector_type.lower().strip()
        with self._lock:
            return self._connectors.get(connector_type_lower)
    
    def list_connectors(self) -> List[str]:
        """
        Lista todos los tipos de conectores registrados (thread-safe).
        
        Returns:
            Lista de tipos de conectores (copia para thread safety).
        """
        with self._lock:
            return list(self._connectors.keys())
    
    def has_connector(self, connector_type: str) -> bool:
        """
        Verifica si un conector está registrado (thread-safe).
        
        Args:
            connector_type: Tipo del conector a verificar.
        
        Returns:
            True si existe, False en caso contrario.
        """
        if not connector_type:
            return False
        
        connector_type_lower = connector_type.lower().strip()
        with self._lock:
            return connector_type_lower in self._connectors
    
    def get_supported_operations(
        self,
        connector_type: str
    ) -> Optional[Set[str]]:
        """
        Obtiene operaciones soportadas por un conector.
        
        Args:
            connector_type: Tipo del conector.
        
        Returns:
            Set de operaciones soportadas o None si el conector no existe.
        """
        connector = self.get(connector_type)
        if connector:
            try:
                operations = connector.get_supported_operations()
                return set(operations) if operations else set()
            except Exception as e:
                logger.error(
                    f"Error getting supported operations for "
                    f"{connector_type}: {e}",
                    exc_info=True
                )
                return None
        return None
    
    def unregister(self, connector_type: str) -> bool:
        """
        Elimina un conector del registro (thread-safe).
        
        Args:
            connector_type: Tipo del conector a eliminar.
        
        Returns:
            True si se eliminó, False si no existía.
        """
        if not connector_type:
            return False
        
        connector_type_lower = connector_type.lower().strip()
        with self._lock:
            if connector_type_lower in self._connectors:
                del self._connectors[connector_type_lower]
                logger.info(f"Unregistered connector: {connector_type_lower}")
                return True
        return False
    
    def clear(self) -> int:
        """
        Limpia todos los conectores registrados (thread-safe).
        
        Returns:
            Número de conectores eliminados.
        """
        with self._lock:
            count = len(self._connectors)
            if count > 0:
                self._connectors.clear()
                logger.info(f"Cleared {count} connectors")
            return count
    
    def count(self) -> int:
        """
        Retorna el número de conectores registrados (thread-safe).
        
        Returns:
            Número de conectores registrados.
        """
        with self._lock:
            return len(self._connectors)
    
    def get_all_connectors(self) -> Dict[str, BaseConnector]:
        """
        Obtiene una copia de todos los conectores registrados (thread-safe).
        
        Returns:
            Diccionario con todos los conectores (copia para thread safety).
        """
        with self._lock:
            return self._connectors.copy()
