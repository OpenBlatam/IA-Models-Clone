"""
Production Utils - Utilidades de Producción
===========================================

Utilidades para servir modelos en producción y A/B testing.
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List, Callable, Any
import time
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Versión de modelo."""
    name: str
    model: nn.Module
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelServer:
    """
    Servidor de modelos para producción.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        batch_size: int = 32
    ):
        """
        Inicializar servidor.
        
        Args:
            model: Modelo
            device: Dispositivo
            batch_size: Tamaño de batch
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        self.stats = {
            'requests': 0,
            'total_time': 0.0,
            'errors': 0
        }
    
    def predict(
        self,
        inputs: torch.Tensor,
        return_probs: bool = False
    ) -> torch.Tensor:
        """
        Predecir.
        
        Args:
            inputs: Inputs
            return_probs: Retornar probabilidades
            
        Returns:
            Predicciones
        """
        start_time = time.time()
        
        try:
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs)
                
                if return_probs:
                    results = torch.softmax(outputs, dim=1)
                else:
                    results = outputs.argmax(dim=1)
            
            elapsed = time.time() - start_time
            self.stats['requests'] += 1
            self.stats['total_time'] += elapsed
            
            return results
        
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Prediction error: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas.
        
        Returns:
            Estadísticas
        """
        avg_time = (
            self.stats['total_time'] / self.stats['requests']
            if self.stats['requests'] > 0 else 0.0
        )
        
        return {
            'total_requests': self.stats['requests'],
            'total_errors': self.stats['errors'],
            'avg_inference_time': avg_time,
            'throughput': 1.0 / avg_time if avg_time > 0 else 0.0
        }


class ABTestManager:
    """
    Gestor de A/B testing para modelos.
    """
    
    def __init__(self):
        """Inicializar gestor."""
        self.versions: Dict[str, ModelVersion] = {}
        self.results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def register_version(
        self,
        name: str,
        model: nn.Module,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Registrar versión de modelo.
        
        Args:
            name: Nombre de versión
            model: Modelo
            weight: Peso para routing
            metadata: Metadata (opcional)
        """
        self.versions[name] = ModelVersion(
            name=name,
            model=model,
            weight=weight,
            metadata=metadata or {}
        )
    
    def route_request(
        self,
        inputs: torch.Tensor,
        user_id: Optional[str] = None
    ) -> str:
        """
        Enrutar request a versión.
        
        Args:
            inputs: Inputs
            user_id: ID de usuario (opcional)
            
        Returns:
            Nombre de versión seleccionada
        """
        if not self.versions:
            raise ValueError("No versions registered")
        
        # Selección basada en peso (simplificado)
        total_weight = sum(v.weight for v in self.versions.values())
        import random
        rand = random.random() * total_weight
        
        cumulative = 0.0
        for name, version in self.versions.items():
            cumulative += version.weight
            if rand <= cumulative:
                return name
        
        # Fallback
        return list(self.versions.keys())[0]
    
    def test(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        metric_fn: Callable
    ) -> Dict[str, float]:
        """
        Probar todas las versiones.
        
        Args:
            inputs: Inputs
            targets: Targets
            metric_fn: Función de métrica
            
        Returns:
            Métricas por versión
        """
        results = {}
        
        for name, version in self.versions.items():
            version.model.eval()
            with torch.no_grad():
                outputs = version.model(inputs)
                metric = metric_fn(targets, outputs)
                results[name] = metric.item() if hasattr(metric, 'item') else metric
                
                self.results[name].append({
                    'metric': metric.item() if hasattr(metric, 'item') else metric,
                    'timestamp': time.time()
                })
        
        return results
    
    def get_winner(
        self,
        metric_name: str = "accuracy",
        higher_is_better: bool = True
    ) -> Optional[str]:
        """
        Obtener versión ganadora.
        
        Args:
            metric_name: Nombre de métrica
            higher_is_better: Mayor es mejor
            
        Returns:
            Nombre de versión ganadora
        """
        if not self.results:
            return None
        
        avg_metrics = {}
        for name, results in self.results.items():
            if results:
                avg_metrics[name] = np.mean([r['metric'] for r in results])
        
        if not avg_metrics:
            return None
        
        if higher_is_better:
            return max(avg_metrics, key=avg_metrics.get)
        else:
            return min(avg_metrics, key=avg_metrics.get)


class ModelCache:
    """
    Caché de modelos para producción.
    """
    
    def __init__(self, max_size: int = 10):
        """
        Inicializar caché.
        
        Args:
            max_size: Tamaño máximo
        """
        self.max_size = max_size
        self.cache: Dict[str, nn.Module] = {}
        self.access_times: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[nn.Module]:
        """
        Obtener modelo del caché.
        
        Args:
            key: Clave
            
        Returns:
            Modelo o None
        """
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, key: str, model: nn.Module):
        """
        Agregar modelo al caché.
        
        Args:
            key: Clave
            model: Modelo
        """
        if len(self.cache) >= self.max_size:
            # Remover menos usado
            lru_key = min(self.access_times, key=self.access_times.get)
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[key] = model
        self.access_times[key] = time.time()
    
    def clear(self):
        """Limpiar caché."""
        self.cache.clear()
        self.access_times.clear()

