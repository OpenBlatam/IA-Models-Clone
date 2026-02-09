"""
Optimización de inferencia para producción
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class InferenceOptimizer:
    """Optimizador de inferencia"""
    
    def __init__(self):
        pass
    
    def optimize_for_inference(
        self,
        model: nn.Module,
        example_input: Dict[str, torch.Tensor]
    ) -> nn.Module:
        """
        Optimiza modelo para inferencia
        
        Args:
            model: Modelo a optimizar
            example_input: Input de ejemplo
            
        Returns:
            Modelo optimizado
        """
        model.eval()
        
        # TorchScript compilation
        try:
            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_input)
                optimized_model = torch.jit.optimize_for_inference(traced_model)
            logger.info("Modelo optimizado con TorchScript")
            return optimized_model
        except Exception as e:
            logger.warning(f"No se pudo optimizar con TorchScript: {e}")
            return model
    
    def enable_inference_mode(self, model: nn.Module):
        """Habilita inference mode (más rápido que eval)"""
        return torch.inference_mode()
    
    def fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fusiona módulos para inferencia más rápida"""
        try:
            # Fusionar BatchNorm y Conv
            torch.quantization.fuse_modules(
                model,
                [['conv', 'bn', 'relu']],
                inplace=True
            )
            logger.info("Módulos fusionados")
        except Exception as e:
            logger.warning(f"No se pudieron fusionar módulos: {e}")
        
        return model
    
    def optimize_memory(self, model: nn.Module):
        """Optimiza uso de memoria"""
        # Empty cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Set eval mode
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False


class BatchInference:
    """Inferencia por batches optimizada"""
    
    def __init__(self, model: nn.Module, batch_size: int = 32, device: str = "cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.batch_size = batch_size
        self.device = device
    
    def predict_batch(
        self,
        inputs: List[Dict[str, torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Predicción por batches
        
        Args:
            inputs: Lista de inputs
            
        Returns:
            Lista de predicciones
        """
        predictions = []
        
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            
            # Batch inputs
            batched = self._batch_inputs(batch)
            
            with torch.inference_mode():
                outputs = self.model(**batched)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                predictions.extend([logits[j] for j in range(len(batch))])
        
        return predictions
    
    def _batch_inputs(self, inputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Agrupa inputs en batch"""
        batched = {}
        
        for key in inputs[0].keys():
            tensors = [inp[key] for inp in inputs]
            batched[key] = torch.stack(tensors).to(self.device)
        
        return batched




