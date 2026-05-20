"""
Manifest Registry - Registro de manifiestos de recursos
========================================================
"""

import logging
import threading
from typing import Dict, Optional, List, Set
from .models import ResourceManifest

logger = logging.getLogger(__name__)


class ManifestRegistry:
    """
    Registro centralizado de manifiestos de recursos (thread-safe).
    
    Permite registrar, obtener y listar recursos disponibles de forma segura
    en entornos concurrentes.
    """
    
    def __init__(self):
        """Inicializa el registry de manifests."""
        self._manifests: Dict[str, ResourceManifest] = {}
        self._lock = threading.RLock()  # Thread safety
    
    def register(self, manifest: ResourceManifest) -> None:
        """
        Registra un manifest de recurso (thread-safe).
        
        Args:
            manifest: Manifest del recurso
            
        Raises:
            ValueError: Si manifest es None o resource_id es inválido
            TypeError: Si manifest no es una instancia de ResourceManifest
        """
        if manifest is None:
            raise ValueError("manifest cannot be None")
        if not isinstance(manifest, ResourceManifest):
            raise TypeError(f"manifest must be an instance of ResourceManifest, got {type(manifest)}")
        if not manifest.resource_id or not isinstance(manifest.resource_id, str):
            raise ValueError("manifest.resource_id must be a non-empty string")
        
        with self._lock:
            if manifest.resource_id in self._manifests:
                logger.warning(f"Overwriting existing manifest: {manifest.resource_id}")
            self._manifests[manifest.resource_id] = manifest
            logger.debug(f"Registered manifest: {manifest.resource_id}")
    
    def get(self, resource_id: str) -> Optional[ResourceManifest]:
        """
        Obtiene un manifest por resource_id (thread-safe).
        
        Args:
            resource_id: ID del recurso
            
        Returns:
            Manifest o None si no existe
        """
        if not resource_id or not isinstance(resource_id, str):
            return None
        
        with self._lock:
            return self._manifests.get(resource_id)
    
    def has_resource(self, resource_id: str) -> bool:
        """
        Verifica si un recurso está registrado (thread-safe).
        
        Args:
            resource_id: ID del recurso
            
        Returns:
            True si existe, False en caso contrario
        """
        if not resource_id:
            return False
        
        with self._lock:
            return resource_id in self._manifests
    
    def get_all(self) -> List[ResourceManifest]:
        """
        Obtiene todos los manifests registrados (thread-safe).
        
        Returns:
            Lista de todos los manifests (copia para thread safety)
        """
        with self._lock:
            return list(self._manifests.values())
    
    def list_resource_ids(self) -> List[str]:
        """
        Lista todos los resource_ids registrados (thread-safe).
        
        Returns:
            Lista de resource_ids (copia para thread safety)
        """
        with self._lock:
            return list(self._manifests.keys())
    
    def get_by_connector_type(self, connector_type: str) -> List[ResourceManifest]:
        """
        Obtiene todos los manifests de un tipo de conector específico (thread-safe).
        
        Args:
            connector_type: Tipo del conector
            
        Returns:
            Lista de manifests que usan ese conector
        """
        if not connector_type:
            return []
        
        connector_type_lower = connector_type.lower().strip()
        with self._lock:
            return [
                manifest for manifest in self._manifests.values()
                if manifest.connector_type.lower() == connector_type_lower
            ]
    
    def get_connector_types(self) -> Set[str]:
        """
        Obtiene todos los tipos de conectores únicos usados por los manifests (thread-safe).
        
        Returns:
            Set de tipos de conectores
        """
        with self._lock:
            return {manifest.connector_type.lower() for manifest in self._manifests.values()}
    
    def unregister(self, resource_id: str) -> bool:
        """
        Elimina un manifest del registro (thread-safe).
        
        Args:
            resource_id: ID del recurso
            
        Returns:
            True si se eliminó, False si no existía
        """
        if not resource_id:
            return False
        
        with self._lock:
            if resource_id in self._manifests:
                del self._manifests[resource_id]
                logger.debug(f"Unregistered manifest: {resource_id}")
                return True
        return False
    
    def clear(self) -> int:
        """
        Limpia todos los manifests (thread-safe).
        
        Returns:
            Número de manifests eliminados
        """
        with self._lock:
            count = len(self._manifests)
            self._manifests.clear()
            logger.debug(f"Cleared {count} manifests")
            return count
    
    def count(self) -> int:
        """
        Retorna el número de manifests registrados (thread-safe).
        
        Returns:
            Número de manifests
        """
        with self._lock:
            return len(self._manifests)

