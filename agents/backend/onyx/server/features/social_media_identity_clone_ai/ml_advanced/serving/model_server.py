"""
Model serving para inferencia en producción
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import asyncio
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ModelServer:
    """Servidor de modelos para inferencia"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        device: str = "cuda",
        max_batch_size: int = 32
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_batch_size = max_batch_size
        self.request_queue = asyncio.Queue()
        self.processing = False
    
    async def predict(
        self,
        inputs: Dict[str, Any],
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Predicción asíncrona
        
        Args:
            inputs: Inputs del modelo
            max_length: Longitud máxima (para generación)
            
        Returns:
            Predicciones
        """
        try:
            # Mover inputs a device
            device_inputs = self._move_to_device(inputs)
            
            with torch.inference_mode():
                if self.tokenizer and "text" in inputs:
                    # Generación de texto
                    encoded = self.tokenizer(
                        inputs["text"],
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.device)
                    
                    outputs = self.model.generate(
                        **encoded,
                        max_length=max_length or 100,
                        do_sample=True,
                        temperature=0.7
                    )
                    
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    return {"generated_text": generated_text}
                else:
                    # Forward pass normal
                    outputs = self.model(**device_inputs)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    probabilities = torch.softmax(logits, dim=-1)
                    
                    return {
                        "logits": logits.cpu().tolist(),
                        "probabilities": probabilities.cpu().tolist()
                    }
        except Exception as e:
            logger.error(f"Error en predicción: {e}", exc_info=True)
            raise
    
    def _move_to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Mueve inputs a device"""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
    
    async def batch_predict(
        self,
        batch_inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Predicción por batch"""
        results = []
        
        for i in range(0, len(batch_inputs), self.max_batch_size):
            batch = batch_inputs[i:i + self.max_batch_size]
            batch_result = await self.predict_batch(batch)
            results.extend(batch_result)
        
        return results
    
    async def predict_batch(
        self,
        batch_inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Predicción de batch"""
        # Implementar batching eficiente
        # Por ahora, procesar secuencialmente
        results = []
        for inputs in batch_inputs:
            result = await self.predict(inputs)
            results.append(result)
        return results


def create_model_server_app(
    model: nn.Module,
    tokenizer: Optional[Any] = None,
    device: str = "cuda"
) -> FastAPI:
    """Crea FastAPI app para model serving"""
    server = ModelServer(model, tokenizer, device)
    app = FastAPI(title="Model Serving API")
    
    @app.post("/predict")
    async def predict_endpoint(request: Dict[str, Any]):
        """Endpoint de predicción"""
        try:
            result = await server.predict(request)
            return JSONResponse(content={"success": True, "result": result})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/batch_predict")
    async def batch_predict_endpoint(request: Dict[str, Any]):
        """Endpoint de batch prediction"""
        try:
            batch_inputs = request.get("inputs", [])
            results = await server.batch_predict(batch_inputs)
            return JSONResponse(content={"success": True, "results": results})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health():
        """Health check"""
        return {"status": "healthy", "device": device}
    
    return app




