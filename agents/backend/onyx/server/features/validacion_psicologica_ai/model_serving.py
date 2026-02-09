"""
Model Serving
=============
Production-ready model serving
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from pathlib import Path
import structlog
import asyncio
from functools import lru_cache

logger = structlog.get_logger()


class ModelServer:
    """
    Production model server
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize model server
        
        Args:
            model: Model to serve
            model_path: Path to model file (optional)
            device: Device (optional, auto-detect)
        """
        from .deep_learning_models import get_device
        
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.model.eval()
        
        if model_path:
            self.load_model(model_path)
        
        logger.info("ModelServer initialized", device=str(self.device))
    
    def load_model(self, model_path: str) -> None:
        """
        Load model from file
        
        Args:
            model_path: Path to model file
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            logger.info("Model loaded", path=model_path)
        except Exception as e:
            logger.error("Error loading model", path=model_path, error=str(e))
            raise
    
    @torch.no_grad()
    def predict(
        self,
        inputs: Dict[str, torch.Tensor],
        return_probs: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction
        
        Args:
            inputs: Model inputs
            return_probs: Return probabilities
            
        Returns:
            Predictions
        """
        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        
        if isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get("predictions"))
        else:
            logits = outputs
        
        predictions = torch.argmax(logits, dim=-1) if logits.dim() > 1 else logits
        
        result = {
            "predictions": predictions.cpu().numpy().tolist()
        }
        
        if return_probs:
            probs = torch.softmax(logits, dim=-1) if logits.dim() > 1 else logits
            result["probabilities"] = probs.cpu().numpy().tolist()
            result["confidence"] = probs.max(dim=-1)[0].cpu().numpy().tolist()
        
        return result
    
    @torch.no_grad()
    async def predict_async(
        self,
        inputs: Dict[str, torch.Tensor],
        return_probs: bool = False
    ) -> Dict[str, Any]:
        """
        Async prediction
        
        Args:
            inputs: Model inputs
            return_probs: Return probabilities
            
        Returns:
            Predictions
        """
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, inputs, return_probs)


class ModelRegistry:
    """Registry for managing multiple models"""
    
    def __init__(self):
        """Initialize registry"""
        self._models: Dict[str, ModelServer] = {}
        logger.info("ModelRegistry initialized")
    
    def register_model(
        self,
        name: str,
        model: nn.Module,
        model_path: Optional[str] = None
    ) -> None:
        """
        Register model
        
        Args:
            name: Model name
            model: Model instance
            model_path: Path to model file
        """
        server = ModelServer(model, model_path)
        self._models[name] = server
        logger.info("Model registered", name=name)
    
    def get_model(self, name: str) -> Optional[ModelServer]:
        """
        Get model by name
        
        Args:
            name: Model name
            
        Returns:
            Model server or None
        """
        return self._models.get(name)
    
    def list_models(self) -> List[str]:
        """
        List all registered models
        
        Returns:
            List of model names
        """
        return list(self._models.keys())


# Global model registry
model_registry = ModelRegistry()




