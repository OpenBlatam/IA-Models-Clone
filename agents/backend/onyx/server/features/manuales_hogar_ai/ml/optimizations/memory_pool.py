"""
Memory Pool
===========

Pool de memoria para reutilización y evitar allocaciones.
"""

import logging
import torch
from typing import Dict, Optional, List
from collections import deque

logger = logging.getLogger(__name__)


class MemoryPool:
    """Pool de memoria para reutilización."""
    
    def __init__(self, max_pools: int = 10):
        """
        Inicializar pool de memoria.
        
        Args:
            max_pools: Número máximo de pools
        """
        self.max_pools = max_pools
        self.pools: Dict[tuple, deque] = {}  # (shape, dtype) -> deque of tensors
        self._logger = logger
    
    def get_tensor(
        self,
        shape: tuple,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Obtener tensor del pool o crear nuevo.
        
        Args:
            shape: Forma del tensor
            dtype: Tipo de datos
            device: Dispositivo
        
        Returns:
            Tensor
        """
        key = (shape, dtype, device)
        
        if key in self.pools and len(self.pools[key]) > 0:
            tensor = self.pools[key].popleft()
            tensor.zero_()  # Limpiar
            return tensor
        
        # Crear nuevo tensor
        return torch.zeros(shape, dtype=dtype, device=device)
    
    def return_tensor(self, tensor: torch.Tensor):
        """
        Devolver tensor al pool.
        
        Args:
            tensor: Tensor a devolver
        """
        key = (tuple(tensor.shape), tensor.dtype, str(tensor.device))
        
        if key not in self.pools:
            self.pools[key] = deque(maxlen=self.max_pools)
        
        if len(self.pools[key]) < self.max_pools:
            self.pools[key].append(tensor.detach())
    
    def clear(self):
        """Limpiar todos los pools."""
        self.pools.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._logger.info("Memory pool limpiado")
    
    def get_stats(self) -> Dict[str, any]:
        """Obtener estadísticas del pool."""
        total_tensors = sum(len(pool) for pool in self.pools.values())
        return {
            "num_pools": len(self.pools),
            "total_tensors": total_tensors,
            "pool_keys": list(self.pools.keys())
        }




