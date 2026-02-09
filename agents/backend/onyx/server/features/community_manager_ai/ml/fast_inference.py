"""
Fast Inference - Inferencia Rápida
===================================

Sistema de inferencia optimizado para máxima velocidad.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from functools import lru_cache
import threading

logger = logging.getLogger(__name__)


class FastInferenceEngine:
    """Motor de inferencia rápida con optimizaciones"""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        use_compile: bool = True,
        use_quantization: bool = False,
        batch_size: int = 1
    ):
        """
        Inicializar motor de inferencia
        
        Args:
            model: Modelo a usar
            device: Dispositivo
            use_compile: Usar torch.compile
            use_quantization: Usar cuantización
            batch_size: Tamaño de batch
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Optimizar modelo
        model = model.to(self.device)
        model.eval()
        
        if use_compile and hasattr(torch, "compile"):
            from ..optimization.model_optimizer import ModelOptimizer
            model = ModelOptimizer.compile_model(model, mode="reduce-overhead")
        
        if use_quantization:
            from ..optimization.quantization import QuantizedModel
            model = QuantizedModel.quantize_dynamic(model)
        
        self.model = model
        
        # Thread pool para procesamiento paralelo
        self.lock = threading.Lock()
        
        logger.info(f"Fast Inference Engine inicializado en {self.device}")
    
    @torch.inference_mode()
    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predicción rápida
        
        Args:
            inputs: Inputs del modelo
            
        Returns:
            Predicción
        """
        # Mover inputs a dispositivo
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Inferencia con autocast para mixed precision
        with torch.cuda.amp.autocast() if self.device == "cuda" else torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs
    
    def predict_batch(self, batch_inputs: List[Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
        """
        Predicción en batch
        
        Args:
            batch_inputs: Lista de inputs
            
        Returns:
            Lista de predicciones
        """
        results = []
        
        # Procesar en batches
        for i in range(0, len(batch_inputs), self.batch_size):
            batch = batch_inputs[i:i + self.batch_size]
            
            # Combinar batch
            combined_inputs = self._combine_batch(batch)
            
            # Predecir
            batch_outputs = self.predict(combined_inputs)
            
            # Separar resultados
            results.extend(self._split_batch(batch_outputs))
        
        return results
    
    def _combine_batch(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Combinar batch de inputs"""
        combined = {}
        for key in batch[0].keys():
            combined[key] = torch.stack([item[key] for item in batch])
        return combined
    
    def _split_batch(self, batch_output: torch.Tensor) -> List[torch.Tensor]:
        """Separar batch de outputs"""
        return [batch_output[i] for i in range(batch_output.size(0))]


class CachedInference:
    """Inferencia con cache para evitar recomputación"""
    
    def __init__(self, max_size: int = 1000):
        """
        Inicializar cache
        
        Args:
            max_size: Tamaño máximo del cache
        """
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    @lru_cache(maxsize=1000)
    def cached_predict(self, cache_key: str, model_func, *args, **kwargs):
        """
        Predicción con cache
        
        Args:
            cache_key: Clave única para cache
            model_func: Función del modelo
            *args: Argumentos posicionales
            **kwargs: Argumentos keyword
            
        Returns:
            Resultado cacheado o nuevo
        """
        if cache_key in self.cache:
            # Mover al final (LRU)
            self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            return self.cache[cache_key]
        
        # Calcular nuevo resultado
        result = model_func(*args, **kwargs)
        
        # Agregar al cache
        if len(self.cache) >= self.max_size:
            # Remover LRU
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[cache_key] = result
        self.access_order.append(cache_key)
        
        return result
    
    def clear_cache(self):
        """Limpiar cache"""
        self.cache.clear()
        self.access_order.clear()




