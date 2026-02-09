"""
Model Interpretability System - Sistema de interpretabilidad de modelos
========================================================================
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Resultado de atribución"""
    input_tokens: List[str]
    attributions: np.ndarray
    method: str
    top_tokens: List[Tuple[str, float]] = field(default_factory=list)


class ModelInterpretability:
    """Sistema de interpretabilidad de modelos"""
    
    def __init__(self):
        self.attribution_history: List[AttributionResult] = []
    
    def gradient_based_attribution(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        device: str = "cuda"
    ) -> np.ndarray:
        """Atribución basada en gradientes (Gradient * Input)"""
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad = True
        
        # Forward pass
        output = model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=-1)
        
        # Backward
        output[0, target_class].backward()
        
        # Gradient * Input
        attribution = (input_tensor.grad * input_tensor).abs().cpu().numpy()
        
        return attribution
    
    def integrated_gradients(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
        steps: int = 50,
        device: str = "cuda"
    ) -> np.ndarray:
        """Integrated Gradients"""
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        input_tensor = input_tensor.to(device)
        
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        else:
            baseline = baseline.to(device)
        
        # Interpolate between baseline and input
        alphas = torch.linspace(0, 1, steps).to(device)
        gradients = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad = True
            
            output = model(interpolated)
            
            if target_class is None:
                target_class = output.argmax(dim=-1)
            
            output[0, target_class].backward()
            gradients.append(interpolated.grad.clone())
            model.zero_grad()
        
        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # Integrated gradients
        attribution = (avg_gradients * (input_tensor - baseline)).abs().cpu().numpy()
        
        return attribution
    
    def attention_visualization(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        layer_idx: int = -1,
        device: str = "cuda"
    ) -> np.ndarray:
        """Visualiza atención de transformers"""
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        attention_weights = []
        
        def attention_hook(module, input, output):
            if hasattr(output, 'attentions') and output.attentions is not None:
                attention_weights.append(output.attentions[-1].detach().cpu())
        
        # Register hook
        hooks = []
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                hooks.append(module.register_forward_hook(attention_hook))
        
        with torch.no_grad():
            _ = model(input_tensor.to(device))
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        if attention_weights:
            return attention_weights[-1].numpy()
        return np.array([])
    
    def shap_values(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        background_samples: torch.Tensor,
        target_class: Optional[int] = None,
        device: str = "cuda"
    ) -> np.ndarray:
        """SHAP values (simplified implementation)"""
        device = torch.device(device)
        model = model.to(device)
        model.eval()
        
        input_tensor = input_tensor.to(device)
        background_samples = background_samples.to(device)
        
        # Baseline prediction
        with torch.no_grad():
            baseline_output = model(background_samples).mean(dim=0)
        
        # Input prediction
        with torch.no_grad():
            input_output = model(input_tensor)
        
        if target_class is None:
            target_class = input_output.argmax(dim=-1)
        
        # Simplified SHAP: difference in predictions
        shap_values = (input_output[0, target_class] - baseline_output[target_class]).abs()
        
        return shap_values.cpu().numpy()
    
    def explain_prediction(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        method: str = "gradient",
        tokenizer: Optional[Any] = None,
        device: str = "cuda"
    ) -> AttributionResult:
        """Explica una predicción"""
        if method == "gradient":
            attributions = self.gradient_based_attribution(model, input_tensor, device=device)
        elif method == "integrated_gradients":
            attributions = self.integrated_gradients(model, input_tensor, device=device)
        elif method == "attention":
            attributions = self.attention_visualization(model, input_tensor, device=device)
        else:
            raise ValueError(f"Método {method} no soportado")
        
        # Convert to tokens if tokenizer provided
        input_tokens = []
        if tokenizer:
            if hasattr(tokenizer, 'decode'):
                input_tokens = tokenizer.decode(input_tensor[0].cpu().numpy().tolist()).split()
            elif hasattr(tokenizer, 'convert_ids_to_tokens'):
                input_tokens = tokenizer.convert_ids_to_tokens(input_tensor[0].cpu().numpy().tolist())
        
        # Get top tokens
        if attributions.size > 0:
            flat_attributions = attributions.flatten()
            top_indices = np.argsort(flat_attributions)[-10:][::-1]
            top_tokens = [
                (input_tokens[i] if i < len(input_tokens) else f"token_{i}", float(flat_attributions[i]))
                for i in top_indices
            ]
        else:
            top_tokens = []
        
        result = AttributionResult(
            input_tokens=input_tokens,
            attributions=attributions,
            method=method,
            top_tokens=top_tokens
        )
        
        self.attribution_history.append(result)
        return result




