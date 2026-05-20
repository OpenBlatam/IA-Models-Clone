"""
Model Server
============

Servidor de modelos para producción.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from ..models.manual_generator_model import ManualGeneratorModel
from ..embeddings.embedding_service import EmbeddingService
from ..config.ml_config import get_ml_config

logger = logging.getLogger(__name__)


class ModelServer:
    """Servidor de modelos para producción."""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8001,
        workers: int = 1
    ):
        """
        Inicializar servidor.
        
        Args:
            host: Host
            port: Puerto
            workers: Número de workers
        """
        self.host = host
        self.port = port
        self.workers = workers
        self.app = FastAPI(title="Manuales Hogar AI - Model Server")
        self._setup_routes()
        self._logger = logger
    
    def _setup_routes(self):
        """Configurar rutas."""
        
        @self.app.get("/health")
        async def health():
            """Health check."""
            return {"status": "healthy"}
        
        @self.app.post("/generate")
        async def generate(request: Dict[str, Any]):
            """Generar texto."""
            try:
                config = get_ml_config()
                model = ManualGeneratorModel(
                    model_name=config.generation_model,
                    use_lora=config.use_lora,
                    device=config.device
                )
                
                prompt = request.get("prompt", "")
                result = model.generate(prompt)
                
                return {"result": result}
            
            except Exception as e:
                self._logger.error(f"Error generando: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/embeddings")
        async def embeddings(request: Dict[str, Any]):
            """Generar embeddings."""
            try:
                config = get_ml_config()
                service = EmbeddingService(
                    model_name=config.embedding_model,
                    device=config.device
                )
                
                texts = request.get("texts", [])
                if isinstance(texts, str):
                    texts = [texts]
                
                embeddings = service.encode(texts)
                
                return {
                    "embeddings": embeddings.tolist(),
                    "dimension": service.get_embedding_dimension()
                }
            
            except Exception as e:
                self._logger.error(f"Error generando embeddings: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self):
        """Ejecutar servidor."""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            workers=self.workers,
            log_level="info"
        )




