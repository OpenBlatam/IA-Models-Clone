"""
Model Serving Module
=====================

Sistema profesional para servir modelos en producción.
Soporta FastAPI, TorchServe, y optimizaciones de inferencia.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    HTTPException = None
    JSONResponse = None
    BaseModel = None
    logging.warning("FastAPI not available. Install with: pip install fastapi uvicorn")

logger = logging.getLogger(__name__)


class ModelServer:
    """
    Servidor profesional para modelos de deep learning.
    
    Características:
    - API REST con FastAPI
    - Batch processing
    - Caching de predicciones
    - Health checks
    - Metrics endpoint
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str = "model",
        version: str = "1.0.0",
        device: str = "cuda"
    ):
        """
        Inicializar servidor de modelo.
        
        Args:
            model: Modelo PyTorch
            model_name: Nombre del modelo
            version: Versión del modelo
            device: Dispositivo
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn")
        
        self.model = model
        self.model_name = model_name
        self.version = version
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        self.app = FastAPI(
            title=f"{model_name} API",
            version=version,
            description=f"API for serving {model_name} model"
        )
        
        self._setup_routes()
        self.request_count = 0
        self.cache: Dict[str, Any] = {}
        
        logger.info(f"ModelServer initialized: {model_name} v{version} on {device}")
    
    def _setup_routes(self):
        """Configurar rutas de la API."""
        
        @self.app.get("/")
        async def root():
            return {
                "model": self.model_name,
                "version": self.version,
                "status": "running"
            }
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "device": str(self.device)}
        
        @self.app.get("/metrics")
        async def metrics():
            return {
                "requests": self.request_count,
                "cache_size": len(self.cache)
            }
    
    def add_predict_endpoint(
        self,
        endpoint: str = "/predict",
        input_schema: Optional[BaseModel] = None,
        preprocess_fn: Optional[callable] = None,
        postprocess_fn: Optional[callable] = None
    ):
        """
        Agregar endpoint de predicción.
        
        Args:
            endpoint: Ruta del endpoint
            input_schema: Schema Pydantic para validación
            preprocess_fn: Función de preprocesamiento
            postprocess_fn: Función de postprocesamiento
        """
        class PredictionRequest(BaseModel):
            data: List[List[float]]
            batch_size: Optional[int] = 32
        
        request_schema = input_schema or PredictionRequest
        
        @self.app.post(endpoint)
        async def predict(request: request_schema):
            try:
                self.request_count += 1
                
                # Preprocesar
                if preprocess_fn:
                    inputs = preprocess_fn(request.data)
                else:
                    inputs = torch.FloatTensor(request.data).to(self.device)
                
                # Predecir
                with torch.no_grad():
                    predictions = self.model(inputs)
                
                # Postprocesar
                if postprocess_fn:
                    results = postprocess_fn(predictions)
                else:
                    results = predictions.cpu().numpy().tolist()
                
                return {"predictions": results}
            
            except Exception as e:
                logger.error(f"Prediction error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1
    ):
        """
        Ejecutar servidor.
        
        Args:
            host: Host
            port: Puerto
            workers: Número de workers
        """
        try:
            import uvicorn
            uvicorn.run(self.app, host=host, port=port, workers=workers)
        except ImportError:
            raise ImportError("uvicorn is required. Install with: pip install uvicorn")


class InferenceOptimizer:
    """
    Optimizador de inferencia para producción.
    
    Incluye:
    - Model compilation (torch.compile)
    - Batch optimization
    - Memory optimization
    - Caching
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Inicializar optimizador.
        
        Args:
            model: Modelo PyTorch
            device: Dispositivo
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        # Compilar modelo si está disponible
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        self.cache: Dict[str, torch.Tensor] = {}
        logger.info("InferenceOptimizer initialized")
    
    def optimize_batch_size(
        self,
        input_shape: Tuple[int, ...],
        max_batch_size: int = 128,
        target_latency: Optional[float] = None
    ) -> int:
        """
        Encontrar batch size óptimo.
        
        Args:
            input_shape: Forma de entrada (sin batch)
            max_batch_size: Batch size máximo
            target_latency: Latencia objetivo en segundos
            
        Returns:
            Batch size óptimo
        """
        import time
        
        best_batch_size = 1
        best_throughput = 0
        
        for batch_size in [1, 2, 4, 8, 16, 32, 64, max_batch_size]:
            if batch_size > max_batch_size:
                continue
            
            try:
                dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
                
                # Warmup
                with torch.no_grad():
                    _ = self.model(dummy_input)
                
                # Medir
                start = time.time()
                with torch.no_grad():
                    _ = self.model(dummy_input)
                latency = time.time() - start
                
                throughput = batch_size / latency
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size
                
                if target_latency and latency > target_latency:
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                raise
        
        logger.info(f"Optimal batch size: {best_batch_size} (throughput: {best_throughput:.2f} samples/s)")
        return best_batch_size
    
    def predict_batch(
        self,
        inputs: Union[np.ndarray, torch.Tensor, List],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Predecir en batch optimizado.
        
        Args:
            inputs: Inputs para el modelo
            batch_size: Batch size (None para usar óptimo)
            
        Returns:
            Predicciones
        """
        if isinstance(inputs, list):
            inputs = np.array(inputs)
        if isinstance(inputs, np.ndarray):
            inputs = torch.FloatTensor(inputs)
        
        inputs = inputs.to(self.device)
        
        if batch_size is None:
            batch_size = self.optimize_batch_size(inputs.shape[1:])
        
        predictions = []
        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                pred = self.model(batch)
                predictions.append(pred.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)

