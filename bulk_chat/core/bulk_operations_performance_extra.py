"""
Optimizaciones adicionales de máximo rendimiento.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import json

# Importar funciones base
try:
    from .bulk_operations_performance import (
        fast_json_dumps,
        fast_json_loads,
        fast_serialize,
        fast_deserialize,
        HAS_ORJSON,
        HAS_MSGPACK,
        HAS_NUMPY
    )
    if HAS_NUMPY:
        import numpy as np
except ImportError:
    HAS_ORJSON = False
    HAS_MSGPACK = False
    HAS_NUMPY = False


class BulkMultiLevelCache:
    """Cache multi-nivel con L1 (memoria) y L2 (disco)."""
    
    def __init__(self, l1_size: int = 1000, l2_enabled: bool = True):
        self.l1_cache: Dict[str, Tuple[Any, float]] = {}  # Memoria
        self.l1_size = l1_size
        self.l2_enabled = l2_enabled
        self.l2_cache: Dict[str, str] = {}  # Paths a disco
        self.access_count: Dict[str, int] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtener de cache (L1 primero, luego L2)."""
        # L1 cache (memoria)
        if key in self.l1_cache:
            value, timestamp = self.l1_cache[key]
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return value
        
        # L2 cache (disco)
        if self.l2_enabled and key in self.l2_cache:
            try:
                import aiofiles
                filepath = self.l2_cache[key]
                async with aiofiles.open(filepath, 'rb') as f:
                    data = await f.read()
                    # Promover a L1
                    if len(self.l1_cache) < self.l1_size:
                        self.l1_cache[key] = (data, time.time())
                    return data
            except Exception:
                pass
        
        return None
    
    async def set(self, key: str, value: Any):
        """Guardar en cache (L1 y opcionalmente L2)."""
        # L1 cache
        if len(self.l1_cache) >= self.l1_size:
            # Evict LRU
            if self.access_count:
                lru_key = min(self.access_count.items(), key=lambda x: x[1])[0]
                del self.l1_cache[lru_key]
                del self.access_count[lru_key]
        
        self.l1_cache[key] = (value, time.time())
        self.access_count[key] = 0
        
        # L2 cache (disco) para valores grandes
        if self.l2_enabled and isinstance(value, bytes) and len(value) > 1024:
            try:
                import aiofiles
                import os
                cache_dir = ".cache"
                os.makedirs(cache_dir, exist_ok=True)
                filepath = os.path.join(cache_dir, f"{hash(key)}.cache")
                async with aiofiles.open(filepath, 'wb') as f:
                    await f.write(value)
                self.l2_cache[key] = filepath
            except Exception:
                pass


class BulkMemoryPool:
    """Pool de memoria para reducir allocations."""
    
    def __init__(self, pool_size: int = 10):
        self.pool_size = pool_size
        self.list_pools: deque = deque(maxlen=pool_size)
        self.dict_pools: deque = deque(maxlen=pool_size)
    
    def get_list(self, initial_size: int = 0) -> List[Any]:
        """Obtener lista del pool."""
        if self.list_pools:
            lst = self.list_pools.popleft()
            lst.clear()
            return lst
        return [None] * initial_size if initial_size > 0 else []
    
    def return_list(self, lst: List[Any]):
        """Devolver lista al pool."""
        if len(self.list_pools) < self.pool_size:
            lst.clear()
            self.list_pools.append(lst)
    
    def get_dict(self) -> Dict[str, Any]:
        """Obtener dict del pool."""
        if self.dict_pools:
            dct = self.dict_pools.popleft()
            dct.clear()
            return dct
        return {}
    
    def return_dict(self, dct: Dict[str, Any]):
        """Devolver dict al pool."""
        if len(self.dict_pools) < self.pool_size:
            dct.clear()
            self.dict_pools.append(dct)


class BulkFastSerializer:
    """Serializador ultra-rápido con múltiples formatos."""
    
    def __init__(self):
        self.has_pickle = True
        self.has_msgpack = HAS_MSGPACK
        self.has_orjson = HAS_ORJSON
    
    def serialize(self, obj: Any, format: str = "auto") -> bytes:
        """Serializar objeto."""
        if format == "auto":
            # Elegir mejor formato
            if self.has_orjson and isinstance(obj, (dict, list)):
                return fast_json_dumps(obj)
            elif self.has_msgpack:
                return fast_serialize(obj, format="msgpack")
            else:
                import pickle
                return pickle.dumps(obj)
        
        elif format == "json":
            return fast_json_dumps(obj) if HAS_ORJSON else json.dumps(obj).encode()
        
        elif format == "msgpack" and self.has_msgpack:
            return fast_serialize(obj, format="msgpack")
        
        elif format == "pickle":
            import pickle
            return pickle.dumps(obj)
        
        else:
            return str(obj).encode()
    
    def deserialize(self, data: bytes, format: str = "auto") -> Any:
        """Deserializar objeto."""
        if format == "auto":
            # Detectar formato
            try:
                if self.has_orjson:
                    return fast_json_loads(data)
            except:
                pass
            
            try:
                if self.has_msgpack:
                    return fast_deserialize(data, format="msgpack")
            except:
                pass
            
            try:
                import pickle
                return pickle.loads(data)
            except:
                return data.decode()
        
        elif format == "json":
            return fast_json_loads(data) if HAS_ORJSON else json.loads(data.decode())
        
        elif format == "msgpack" and self.has_msgpack:
            return fast_deserialize(data, format="msgpack")
        
        elif format == "pickle":
            import pickle
            return pickle.loads(data)
        
        else:
            return data.decode()


class BulkBatchAggregator:
    """Agregador de batches para operaciones masivas."""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.batches: Dict[str, List[Any]] = {}
    
    def add(self, batch_id: str, item: Any):
        """Agregar item a batch."""
        if batch_id not in self.batches:
            self.batches[batch_id] = []
        
        self.batches[batch_id].append(item)
    
    def get_batch(self, batch_id: str, flush: bool = False) -> Optional[List[Any]]:
        """Obtener batch si está lleno."""
        if batch_id not in self.batches:
            return None
        
        batch = self.batches[batch_id]
        
        if len(batch) >= self.batch_size or flush:
            if flush:
                self.batches[batch_id] = []
            else:
                self.batches[batch_id] = batch[self.batch_size:]
                batch = batch[:self.batch_size]
            return batch
        
        return None
    
    def flush_all(self) -> Dict[str, List[Any]]:
        """Vaciar todos los batches."""
        result = {}
        for batch_id, batch in self.batches.items():
            if batch:
                result[batch_id] = batch
                self.batches[batch_id] = []
        return result


class BulkPerformanceTracker:
    """Tracker de rendimiento con métricas avanzadas."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.counts: Dict[str, int] = {}
        self.totals: Dict[str, float] = {}
    
    def record(self, metric_name: str, value: float):
        """Registrar métrica."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
            self.counts[metric_name] = 0
            self.totals[metric_name] = 0.0
        
        self.metrics[metric_name].append(value)
        self.counts[metric_name] += 1
        self.totals[metric_name] += value
        
        # Mantener solo últimos 1000
        if len(self.metrics[metric_name]) > 1000:
            self.metrics[metric_name].pop(0)
    
    def get_stats(self, metric_name: str) -> Dict[str, Any]:
        """Obtener estadísticas de métrica."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}
        
        values = self.metrics[metric_name]
        
        if HAS_NUMPY:
            arr = np.array(values)
            return {
                "count": len(values),
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "std": float(np.std(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "total": float(np.sum(arr))
            }
        else:
            return {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "total": sum(values)
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Obtener todas las estadísticas."""
        return {
            name: self.get_stats(name)
            for name in self.metrics.keys()
        }


__all__ = [
    "BulkMultiLevelCache",
    "BulkMemoryPool",
    "BulkFastSerializer",
    "BulkBatchAggregator",
    "BulkPerformanceTracker"
]
















