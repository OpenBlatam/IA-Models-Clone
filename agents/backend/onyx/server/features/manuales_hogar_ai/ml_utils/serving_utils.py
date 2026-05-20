"""
Serving Utils - Advanced Model Serving Utilities
================================================

Utilidades avanzadas para servir modelos en producción.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Callable, Any
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

# Intentar importar FastAPI
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    _has_fastapi = True
except ImportError:
    _has_fastapi = False
    logger.warning("FastAPI not available, REST API features will be limited")


@dataclass
class ServingConfig:
    """Configuración de serving."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    batch_size: int = 32
    max_queue_size: int = 100
    timeout: float = 30.0


class RESTModelServer:
    """
    Servidor REST para modelos.
    """
    
    def __init__(
        self,
        model: nn.Module,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
        device: str = "cuda"
    ):
        """
        Inicializar servidor REST.
        
        Args:
            model: Modelo
            preprocess_fn: Función de preprocesamiento
            postprocess_fn: Función de postprocesamiento
            device: Dispositivo
        """
        if not _has_fastapi:
            raise ImportError("FastAPI is required for RESTModelServer")
        
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        
        self.app = FastAPI()
        self.stats = defaultdict(int)
        self._setup_routes()
    
    def _setup_routes(self):
        """Configurar rutas."""
        @self.app.post("/predict")
        async def predict(request: dict):
            """Endpoint de predicción."""
            try:
                start_time = time.time()
                
                # Preprocesar
                if self.preprocess_fn:
                    inputs = self.preprocess_fn(request['data'])
                else:
                    inputs = torch.tensor(request['data'])
                
                inputs = inputs.to(self.device)
                
                # Predecir
                with torch.no_grad():
                    outputs = self.model(inputs)
                
                # Postprocesar
                if self.postprocess_fn:
                    results = self.postprocess_fn(outputs)
                else:
                    results = outputs.cpu().numpy().tolist()
                
                elapsed = time.time() - start_time
                self.stats['requests'] += 1
                self.stats['total_time'] += elapsed
                
                return {
                    'predictions': results,
                    'inference_time': elapsed
                }
            
            except Exception as e:
                self.stats['errors'] += 1
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health():
            """Health check."""
            return {
                'status': 'healthy',
                'model_loaded': True
            }
        
        @self.app.get("/stats")
        async def stats():
            """Estadísticas."""
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
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Ejecutar servidor.
        
        Args:
            host: Host
            port: Puerto
        """
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


class BatchPredictor:
    """
    Predictor por batches optimizado.
    """
    
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 32,
        device: str = "cuda",
        use_amp: bool = True
    ):
        """
        Inicializar predictor.
        
        Args:
            model: Modelo
            batch_size: Tamaño de batch
            device: Dispositivo
            use_amp: Usar mixed precision
        """
        self.model = model.to(device)
        self.model.eval()
        self.batch_size = batch_size
        self.device = device
        self.use_amp = use_amp
    
    def predict_batch(
        self,
        inputs: List[torch.Tensor],
        return_probs: bool = False
    ) -> List[torch.Tensor]:
        """
        Predecir batch.
        
        Args:
            inputs: Lista de inputs
            return_probs: Retornar probabilidades
            
        Returns:
            Lista de predicciones
        """
        all_predictions = []
        
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            batch_tensor = torch.stack(batch).to(self.device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast() if self.use_amp else torch.no_grad():
                    outputs = self.model(batch_tensor)
                
                if return_probs:
                    outputs = torch.softmax(outputs, dim=1)
                else:
                    outputs = outputs.argmax(dim=1)
            
            all_predictions.extend(outputs.cpu())
        
        return all_predictions


class ModelVersionManager:
    """
    Gestor de versiones de modelos para serving.
    """
    
    def __init__(self):
        """Inicializar gestor."""
        self.versions: Dict[str, nn.Module] = {}
        self.active_version: Optional[str] = None
    
    def register_version(
        self,
        version: str,
        model: nn.Module
    ):
        """
        Registrar versión.
        
        Args:
            version: Versión
            model: Modelo
        """
        self.versions[version] = model
        if self.active_version is None:
            self.active_version = version
    
    def set_active_version(self, version: str):
        """
        Establecer versión activa.
        
        Args:
            version: Versión
        """
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        self.active_version = version
    
    def get_active_model(self) -> Optional[nn.Module]:
        """
        Obtener modelo activo.
        
        Returns:
            Modelo activo
        """
        if self.active_version:
            return self.versions[self.active_version]
        return None




