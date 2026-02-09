"""
Estrategias avanzadas de caching
"""

import torch
import hashlib
import pickle
from typing import Any, Optional, Dict
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class ModelCache:
    """Cache avanzado para modelos"""
    
    def __init__(self, cache_dir: str = "./cache/models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, model_config: Dict[str, Any]) -> str:
        """Genera key de cache"""
        config_str = json.dumps(model_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get(
        self,
        model_config: Dict[str, Any]
    ) -> Optional[torch.nn.Module]:
        """
        Obtiene modelo del cache
        
        Args:
            model_config: Configuración del modelo
            
        Returns:
            Modelo cacheado o None
        """
        cache_key = self._get_cache_key(model_config)
        cache_path = self.cache_dir / f"{cache_key}.pt"
        
        if cache_path.exists():
            try:
                model = torch.load(cache_path, map_location="cpu")
                logger.info(f"Modelo cargado del cache: {cache_key}")
                return model
            except Exception as e:
                logger.warning(f"Error cargando del cache: {e}")
                return None
        
        return None
    
    def set(
        self,
        model: torch.nn.Module,
        model_config: Dict[str, Any]
    ):
        """
        Guarda modelo en cache
        
        Args:
            model: Modelo a cachear
            model_config: Configuración del modelo
        """
        cache_key = self._get_cache_key(model_config)
        cache_path = self.cache_dir / f"{cache_key}.pt"
        
        try:
            torch.save(model.state_dict(), cache_path)
            logger.info(f"Modelo guardado en cache: {cache_key}")
        except Exception as e:
            logger.error(f"Error guardando en cache: {e}")


class PredictionCache:
    """Cache para predicciones"""
    
    def __init__(self, cache_dir: str = "./cache/predictions", max_size: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
    
    def _get_key(self, inputs: Any) -> str:
        """Genera key de cache"""
        if isinstance(inputs, dict):
            inputs_str = json.dumps(inputs, sort_keys=True, default=str)
        else:
            inputs_str = str(inputs)
        return hashlib.md5(inputs_str.encode()).hexdigest()
    
    def get(self, inputs: Any) -> Optional[Any]:
        """Obtiene predicción del cache"""
        key = self._get_key(inputs)
        
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        
        return None
    
    def set(self, inputs: Any, prediction: Any):
        """Guarda predicción en cache"""
        key = self._get_key(inputs)
        
        # Limpiar si está lleno
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = prediction
        self.access_times[key] = time.time()
    
    def _evict_oldest(self):
        """Elimina entrada más antigua"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times, key=self.access_times.get)
        del self.cache[oldest_key]
        del self.access_times[oldest_key]


import time




