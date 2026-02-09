"""
Memcached Client - Cliente para caché de alta performance
=========================================================

Soporta:
- Caché distribuido
- TTL (Time To Live)
- Operaciones batch
"""

import logging
from typing import Optional, Dict, Any, List
import hashlib
import json

logger = logging.getLogger(__name__)


class MemcachedClient:
    """Cliente para Memcached"""
    
    def __init__(self, servers: List[str] = None):
        self.servers = servers or ["localhost:11211"]
        self.client = None
        self._setup()
    
    def _setup(self):
        """Configura cliente de Memcached"""
        try:
            import pymemcache.client
            
            self.client = pymemcache.client.HashClient(
                self.servers,
                connect_timeout=5,
                timeout=5,
                retry_attempts=3
            )
            logger.info("Memcached client configured")
        except ImportError:
            logger.warning("pymemcache not available. Install with: pip install pymemcache")
        except Exception as e:
            logger.error(f"Failed to setup Memcached: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor del caché"""
        if not self.client:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                # Intentar deserializar JSON
                try:
                    return json.loads(value.decode('utf-8'))
                except:
                    return value.decode('utf-8')
            return None
        except Exception as e:
            logger.error(f"Memcached get error: {e}")
            return None
    
    def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """Guarda un valor en el caché"""
        if not self.client:
            return False
        
        try:
            # Serializar si es necesario
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            elif not isinstance(value, (str, bytes)):
                value = str(value)
            
            if isinstance(value, str):
                value = value.encode('utf-8')
            
            return self.client.set(key, value, expire=expire)
        except Exception as e:
            logger.error(f"Memcached set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Elimina un valor del caché"""
        if not self.client:
            return False
        
        try:
            return self.client.delete(key)
        except Exception as e:
            logger.error(f"Memcached delete error: {e}")
            return False
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Obtiene múltiples valores"""
        if not self.client:
            return {}
        
        try:
            values = self.client.get_many(keys)
            result = {}
            for key, value in values.items():
                if value:
                    try:
                        result[key] = json.loads(value.decode('utf-8'))
                    except:
                        result[key] = value.decode('utf-8')
            return result
        except Exception as e:
            logger.error(f"Memcached get_many error: {e}")
            return {}
    
    def set_many(self, mapping: Dict[str, Any], expire: int = 3600) -> bool:
        """Guarda múltiples valores"""
        if not self.client:
            return False
        
        try:
            # Serializar valores
            serialized = {}
            for key, value in mapping.items():
                if isinstance(value, (dict, list)):
                    serialized[key] = json.dumps(value).encode('utf-8')
                elif isinstance(value, str):
                    serialized[key] = value.encode('utf-8')
                else:
                    serialized[key] = str(value).encode('utf-8')
            
            return self.client.set_many(serialized, expire=expire)
        except Exception as e:
            logger.error(f"Memcached set_many error: {e}")
            return False
    
    def flush_all(self) -> bool:
        """Limpia todo el caché"""
        if not self.client:
            return False
        
        try:
            return self.client.flush_all()
        except Exception as e:
            logger.error(f"Memcached flush_all error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del servidor"""
        if not self.client:
            return {}
        
        try:
            stats = self.client.stats()
            return {k.decode('utf-8'): v.decode('utf-8') if isinstance(v, bytes) else v 
                   for k, v in stats.items()}
        except Exception as e:
            logger.error(f"Memcached stats error: {e}")
            return {}


# Instancia global
memcached_client = MemcachedClient()




