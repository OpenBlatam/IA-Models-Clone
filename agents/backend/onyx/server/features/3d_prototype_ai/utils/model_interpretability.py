"""
Model Interpretability - Interpretabilidad de modelos
======================================================
SHAP, LIME, attention visualization, y más
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available")

try:
    from lime import lime_text
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available")

logger = logging.getLogger(__name__)


class ModelInterpreter:
    """Sistema de interpretabilidad de modelos"""
    
    def __init__(self):
        self.attention_weights: Dict[str, List[torch.Tensor]] = defaultdict(list)
    
    def extract_attention_weights(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        layer_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Extrae pesos de atención"""
        attention_weights = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    # Transformer output
                    if len(output) > 1:
                        attention_weights[name] = output[1]  # Attention weights
                elif hasattr(output, 'attentions'):
                    attention_weights[name] = output.attentions
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_ids)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_weights
    
    def visualize_attention(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        head_idx: int = 0
    ) -> Dict[str, Any]:
        """Visualiza atención"""
        # Promediar sobre todas las capas si es 3D
        if len(attention_weights.shape) == 4:
            # [batch, heads, seq, seq]
            attn = attention_weights[0, head_idx].cpu().numpy()
        elif len(attention_weights.shape) == 3:
            # [batch, seq, seq]
            attn = attention_weights[0].cpu().numpy()
        else:
            attn = attention_weights.cpu().numpy()
        
        # Normalizar
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
        
        return {
            "attention_matrix": attn.tolist(),
            "tokens": tokens,
            "head_idx": head_idx
        }
    
    def explain_with_shap(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        background_data: torch.Tensor,
        tokenizer: Any
    ) -> Dict[str, Any]:
        """Explicación con SHAP"""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available")
        
        def model_wrapper(x):
            model.eval()
            with torch.no_grad():
                outputs = model(torch.tensor(x))
                if isinstance(outputs, tuple):
                    return outputs[0].cpu().numpy()
                return outputs.cpu().numpy()
        
        explainer = shap.Explainer(model_wrapper, background_data.cpu().numpy())
        shap_values = explainer(input_data.cpu().numpy())
        
        return {
            "shap_values": shap_values.values.tolist(),
            "base_values": shap_values.base_values.tolist(),
            "data": shap_values.data.tolist()
        }
    
    def explain_with_lime(
        self,
        model: nn.Module,
        text: str,
        tokenizer: Any,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """Explicación con LIME"""
        if not LIME_AVAILABLE:
            raise ImportError("LIME not available")
        
        def predict_proba(texts):
            model.eval()
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
            return probs.cpu().numpy()
        
        explainer = lime_text.LimeTextExplainer(class_names=["negative", "positive"])
        explanation = explainer.explain_instance(text, predict_proba, num_features=num_features)
        
        return {
            "explanation": explanation.as_list(),
            "score": explanation.score
        }
    
    def gradient_based_importance(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        """Importancia basada en gradientes"""
        input_tensor.requires_grad = True
        model.eval()
        
        output = model(input_tensor)
        if output.dim() > 1:
            output = output[:, target_class]
        
        output.backward()
        gradients = input_tensor.grad
        
        # Saliency map
        saliency = torch.abs(gradients)
        return saliency
    
    def integrated_gradients(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: int,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> torch.Tensor:
        """Integrated Gradients"""
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        input_tensor.requires_grad = True
        alphas = torch.linspace(0, 1, steps)
        
        integrated_grads = torch.zeros_like(input_tensor)
        
        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad = True
            
            output = model(interpolated)
            if output.dim() > 1:
                output = output[:, target_class]
            
            output.backward()
            integrated_grads += interpolated.grad
        
        integrated_grads = (input_tensor - baseline) * integrated_grads / steps
        return integrated_grads




