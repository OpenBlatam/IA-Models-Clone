"""
Model Server - Modular Model Serving
====================================

Servidor modular para servir modelos de deep learning.
"""

import logging
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from pathlib import Path
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from ..dl_inference import InferenceEngine

logger = logging.getLogger(__name__)


class ModelServer:
    """
    Servidor para servir modelos.
    
    Proporciona API REST para inferencia de modelos.
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str = "model",
        host: str = "0.0.0.0",
        port: int = 8000,
        device: Optional[torch.device] = None
    ):
        """
        Inicializar servidor.
        
        Args:
            model: Modelo a servir
            model_name: Nombre del modelo
            host: Host del servidor
            port: Puerto del servidor
            device: Dispositivo
        """
        self.model = model
        self.model_name = model_name
        self.host = host
        self.port = port
        
        # Crear inference engine
        self.inference_engine = InferenceEngine(model, device=device)
        
        # Crear FastAPI app
        self.app = FastAPI(title=f"{model_name} Model Server")
        self._setup_routes()
        
        logger.info(f"Model Server initialized: {model_name}")
    
    def _setup_routes(self):
        """Configurar rutas de API."""
        
        @self.app.get("/")
        async def root():
            return {
                "model": self.model_name,
                "status": "running",
                "endpoints": ["/predict", "/health", "/info"]
            }
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}
        
        @self.app.get("/info")
        async def info():
            return {
                "model_name": self.model_name,
                "device": str(self.inference_engine.device_manager.device),
                "model_info": self.inference_engine.device_manager.get_device_info()
            }
        
        @self.app.post("/predict")
        async def predict(request: Dict[str, Any]):
            """
            Endpoint de predicción.
            
            Body:
                - data: Datos de entrada (lista o array)
                - batch_size: Tamaño de batch (opcional)
            """
            try:
                data = request.get('data')
                if data is None:
                    raise HTTPException(status_code=400, detail="Missing 'data' field")
                
                # Convertir a tensor
                if isinstance(data, list):
                    import numpy as np
                    data = np.array(data)
                
                batch_size = request.get('batch_size', None)
                
                # Realizar predicción
                predictions = self.inference_engine.predict(data, batch_size=batch_size)
                
                # Convertir a lista para JSON
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.cpu().numpy().tolist()
                
                return {
                    "predictions": predictions,
                    "shape": list(predictions.shape) if hasattr(predictions, 'shape') else None
                }
            except Exception as e:
                logger.error(f"Error in prediction: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict_batch")
        async def predict_batch(request: Dict[str, Any]):
            """
            Endpoint de predicción por batch.
            
            Body:
                - batches: Lista de batches
            """
            try:
                batches = request.get('batches', [])
                if not batches:
                    raise HTTPException(status_code=400, detail="Missing 'batches' field")
                
                all_predictions = []
                for batch in batches:
                    predictions = self.inference_engine.predict(batch)
                    if isinstance(predictions, torch.Tensor):
                        predictions = predictions.cpu().numpy().tolist()
                    all_predictions.append(predictions)
                
                return {
                    "predictions": all_predictions,
                    "num_batches": len(all_predictions)
                }
            except Exception as e:
                logger.error(f"Error in batch prediction: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, **kwargs):
        """
        Ejecutar servidor.
        
        Args:
            **kwargs: Argumentos adicionales para uvicorn
        """
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            **kwargs
        )
    
    async def run_async(self, **kwargs):
        """Ejecutar servidor de forma asíncrona."""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            **kwargs
        )
        server = uvicorn.Server(config)
        await server.serve()


class AsyncModelServer(ModelServer):
    """Servidor asíncrono para modelos."""
    
    def __init__(self, *args, **kwargs):
        """Inicializar servidor asíncrono."""
        super().__init__(*args, **kwargs)
        self._setup_async_routes()
    
    def _setup_async_routes(self):
        """Configurar rutas asíncronas adicionales."""
        
        @self.app.post("/predict_async")
        async def predict_async(request: Dict[str, Any]):
            """Predicción asíncrona."""
            # Ejecutar en thread pool para no bloquear
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.inference_engine.predict(request.get('data'))
            )
            
            if isinstance(result, torch.Tensor):
                result = result.cpu().numpy().tolist()
            
            return {"predictions": result}








