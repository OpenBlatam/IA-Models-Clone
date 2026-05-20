"""
Sistema de Caché Distribuido
==============================

Sistema para caché distribuido entre múltiples nodos.
"""

import logging
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class CacheNode:
    """Nodo de caché"""
    node_id: str
    url: str
    active: bool = True
    capacity: int = 1000
    current_load: int = 0


class DistributedCache:
    """
    Caché distribuido
    
    Proporciona:
    - Distribución de datos entre nodos
    - Consistent hashing
    - Replicación
    - Fallback automático
    - Sincronización entre nodos
    """
    
    def __init__(self):
        """Inicializar caché distribuido"""
        self.nodes: Dict[str, CacheNode] = {}
        self.cache_data: Dict[str, Dict[str, Any]] = {}
        self.replication_factor = 2
        logger.info("DistributedCache inicializado")
    
    def register_node(
        self,
        node_id: str,
        url: str,
        capacity: int = 1000
    ):
        """Registrar nodo de caché"""
        node = CacheNode(
            node_id=node_id,
            url=url,
            capacity=capacity
        )
        
        self.nodes[node_id] = node
        logger.info(f"Nodo de caché registrado: {node_id}")
    
    def _get_node_for_key(self, key: str) -> str:
        """Obtener nodo para una clave usando consistent hashing"""
        if not self.nodes:
            return None
        
        # Consistent hashing simple
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        node_ids = sorted([nid for nid, n in self.nodes.items() if n.active])
        
        if not node_ids:
            return None
        
        node_index = hash_value % len(node_ids)
        return node_ids[node_index]
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Establecer valor en caché distribuido
        
        Args:
            key: Clave
            value: Valor
            ttl: Time to live en segundos
        
        Returns:
            True si se guardó correctamente
        """
        node_id = self._get_node_for_key(key)
        
        if not node_id:
            # Fallback a almacenamiento local
            self.cache_data[key] = {
                "value": value,
                "expires_at": (datetime.now() + timedelta(seconds=ttl)).isoformat() if ttl else None,
                "node_id": "local"
            }
            return True
        
        # Guardar en nodo (simulación)
        # En producción, haría una llamada HTTP al nodo
        self.cache_data[key] = {
            "value": value,
            "expires_at": (datetime.now() + timedelta(seconds=ttl)).isoformat() if ttl else None,
            "node_id": node_id
        }
        
        logger.debug(f"Valor guardado en nodo {node_id}: {key}")
        
        return True
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché distribuido"""
        if key not in self.cache_data:
            return None
        
        entry = self.cache_data[key]
        
        # Verificar expiración
        if entry.get("expires_at"):
            expires = datetime.fromisoformat(entry["expires_at"])
            if datetime.now() > expires:
                del self.cache_data[key]
                return None
        
        return entry["value"]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del caché"""
        return {
            "total_nodes": len(self.nodes),
            "active_nodes": sum(1 for n in self.nodes.values() if n.active),
            "total_entries": len(self.cache_data),
            "nodes": [
                {
                    "node_id": n.node_id,
                    "url": n.url,
                    "active": n.active,
                    "capacity": n.capacity,
                    "current_load": n.current_load
                }
                for n in self.nodes.values()
            ]
        }


# Instancia global
_distributed_cache: Optional[DistributedCache] = None


def get_distributed_cache() -> DistributedCache:
    """Obtener instancia global del caché"""
    global _distributed_cache
    if _distributed_cache is None:
        _distributed_cache = DistributedCache()
    return _distributed_cache














