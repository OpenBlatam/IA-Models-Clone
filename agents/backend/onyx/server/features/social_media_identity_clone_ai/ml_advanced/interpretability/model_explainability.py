"""
Interpretabilidad y explicabilidad de modelos
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Explicador de modelos"""
    
    def __init__(self, model: nn.Module, tokenizer: Optional[Any] = None):
        self.model = model
        self.tokenizer = tokenizer
    
    def gradient_based_importance(
        self,
        inputs: Dict[str, torch.Tensor],
        target_class: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Importancia basada en gradientes
        
        Args:
            inputs: Inputs del modelo
            target_class: Clase objetivo (None para usar predicción)
            
        Returns:
            Importancia de cada token/feature
        """
        self.model.eval()
        inputs = {k: v.requires_grad_(True) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=-1)
        
        # Backward
        self.model.zero_grad()
        logits[0, target_class].backward()
        
        # Obtener gradientes
        importance = {}
        for key, value in inputs.items():
            if value.grad is not None:
                importance[key] = value.grad.abs()
        
        return importance
    
    def integrated_gradients(
        self,
        inputs: Dict[str, torch.Tensor],
        baseline: Optional[Dict[str, torch.Tensor]] = None,
        steps: int = 50,
        target_class: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Integrated Gradients para explicabilidad
        
        Args:
            inputs: Inputs del modelo
            baseline: Baseline (None para usar ceros)
            steps: Número de pasos
            target_class: Clase objetivo
            
        Returns:
            Atribuciones
        """
        if baseline is None:
            baseline = {k: torch.zeros_like(v) for k, v in inputs.items()}
        
        # Interpolar entre baseline y inputs
        alphas = torch.linspace(0, 1, steps)
        attributions = {k: torch.zeros_like(v) for k, v in inputs.items()}
        
        for alpha in alphas:
            interpolated = {
                k: baseline[k] + alpha * (inputs[k] - baseline[k])
                for k in inputs.keys()
            }
            
            interpolated = {k: v.requires_grad_(True) for k, v in interpolated.items()}
            
            outputs = self.model(**interpolated)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            if target_class is None:
                target_class = torch.argmax(logits, dim=-1)
            
            self.model.zero_grad()
            logits[0, target_class].backward()
            
            for key in attributions.keys():
                if interpolated[key].grad is not None:
                    attributions[key] += interpolated[key].grad / steps
        
        # Multiplicar por diferencia
        for key in attributions.keys():
            attributions[key] *= (inputs[key] - baseline[key])
        
        return attributions
    
    def attention_visualization(
        self,
        inputs: Dict[str, torch.Tensor],
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Visualiza attention weights
        
        Args:
            inputs: Inputs del modelo
            layer_idx: Índice de capa
            
        Returns:
            Attention weights
        """
        # Hook para capturar attention
        attention_weights = []
        
        def hook_fn(module, input, output):
            if hasattr(output, 'attentions') and output.attentions is not None:
                attention_weights.append(output.attentions[layer_idx])
        
        # Registrar hook
        handles = []
        for module in self.model.modules():
            if hasattr(module, 'attention'):
                handle = module.register_forward_hook(hook_fn)
                handles.append(handle)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Remover hooks
        for handle in handles:
            handle.remove()
        
        if attention_weights:
            return attention_weights[0]
        return None




