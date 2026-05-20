"""
Interpretability Utils - Utilidades de Interpretabilidad
=========================================================

Utilidades para interpretar y visualizar modelos.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Intentar importar bibliotecas opcionales
try:
    import captum
    from captum.attr import IntegratedGradients, GradientShap, Saliency, InputXGradient
    _has_captum = True
except ImportError:
    _has_captum = False
    logger.warning("captum not available, some interpretability functions will be limited")


class AttentionVisualizer:
    """
    Visualizador de atención para modelos transformer.
    """
    
    def __init__(self, model: nn.Module, tokenizer):
        """
        Inicializar visualizador de atención.
        
        Args:
            model: Modelo transformer
            tokenizer: Tokenizador
        """
        self.model = model
        self.tokenizer = tokenizer
    
    def get_attention_weights(
        self,
        text: str,
        layer: Optional[int] = None,
        head: Optional[int] = None
    ) -> torch.Tensor:
        """
        Obtener pesos de atención.
        
        Args:
            text: Texto de entrada
            layer: Capa específica (opcional)
            head: Head específico (opcional)
            
        Returns:
            Pesos de atención
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions
        
        # Seleccionar capa y head
        if layer is not None:
            attentions = [attentions[layer]]
        
        # Promediar sobre heads si no se especifica
        if head is None:
            attentions = [att.mean(dim=1) for att in attentions]
        else:
            attentions = [att[:, head, :, :] for att in attentions]
        
        return attentions[0] if len(attentions) == 1 else attentions
    
    def visualize_attention(
        self,
        text: str,
        layer: int = 0,
        head: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Visualizar atención.
        
        Args:
            text: Texto de entrada
            layer: Capa a visualizar
            head: Head a visualizar
            save_path: Ruta para guardar (opcional)
        """
        tokens = self.tokenizer.tokenize(text)
        attention_weights = self.get_attention_weights(text, layer=layer, head=head)
        
        if isinstance(attention_weights, list):
            attention_weights = attention_weights[0]
        
        attention_matrix = attention_weights[0].cpu().numpy()
        
        plt.figure(figsize=(12, 8))
        plt.imshow(attention_matrix, cmap='Blues')
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.yticks(range(len(tokens)), tokens)
        plt.colorbar()
        plt.title(f'Attention Weights - Layer {layer}, Head {head}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class GradientAnalyzer:
    """
    Analizador de gradientes para interpretabilidad.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar analizador de gradientes.
        
        Args:
            model: Modelo PyTorch
        """
        self.model = model
    
    def compute_gradients(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        loss_fn: Callable
    ) -> torch.Tensor:
        """
        Calcular gradientes.
        
        Args:
            inputs: Inputs del modelo
            target: Target
            loss_fn: Función de pérdida
            
        Returns:
            Gradientes
        """
        inputs.requires_grad_(True)
        outputs = self.model(inputs)
        loss = loss_fn(outputs, target)
        loss.backward()
        
        return inputs.grad
    
    def integrated_gradients(
        self,
        inputs: torch.Tensor,
        baseline: torch.Tensor,
        target: torch.Tensor,
        loss_fn: Callable,
        steps: int = 50
    ) -> torch.Tensor:
        """
        Calcular Integrated Gradients.
        
        Args:
            inputs: Inputs
            baseline: Baseline
            target: Target
            loss_fn: Función de pérdida
            steps: Número de pasos
            
        Returns:
            Atribuciones
        """
        attributions = torch.zeros_like(inputs)
        
        for step in range(steps + 1):
            alpha = step / steps
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated.requires_grad_(True)
            
            outputs = self.model(interpolated)
            loss = loss_fn(outputs, target)
            loss.backward()
            
            attributions += interpolated.grad
        
        attributions = attributions * (inputs - baseline) / (steps + 1)
        return attributions


class FeatureImportance:
    """
    Calculador de importancia de features.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar calculador de importancia.
        
        Args:
            model: Modelo PyTorch
        """
        self.model = model
    
    def compute_importance(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        method: str = "gradient"
    ) -> torch.Tensor:
        """
        Calcular importancia de features.
        
        Args:
            inputs: Inputs
            target: Target
            method: Método ('gradient', 'integrated_gradients')
            
        Returns:
            Importancia de features
        """
        if method == "gradient":
            inputs.requires_grad_(True)
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, target)
            loss.backward()
            importance = torch.abs(inputs.grad)
        
        elif method == "integrated_gradients":
            baseline = torch.zeros_like(inputs)
            analyzer = GradientAnalyzer(self.model)
            importance = analyzer.integrated_gradients(
                inputs, baseline, target, F.cross_entropy
            )
            importance = torch.abs(importance)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return importance


class CaptumWrapper:
    """
    Wrapper para Captum (si está disponible).
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar wrapper de Captum.
        
        Args:
            model: Modelo PyTorch
        """
        if not _has_captum:
            raise ImportError("captum is required for CaptumWrapper")
        
        self.model = model
        self.ig = IntegratedGradients(model)
        self.gs = GradientShap(model)
        self.saliency = Saliency(model)
        self.ixg = InputXGradient(model)
    
    def integrated_gradients(
        self,
        inputs: torch.Tensor,
        target: int,
        baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calcular Integrated Gradients.
        
        Args:
            inputs: Inputs
            target: Target class
            baseline: Baseline (opcional)
            
        Returns:
            Atribuciones
        """
        if baseline is None:
            baseline = torch.zeros_like(inputs)
        
        attributions = self.ig.attribute(inputs, baseline, target=target)
        return attributions
    
    def gradient_shap(
        self,
        inputs: torch.Tensor,
        target: int,
        baselines: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcular Gradient SHAP.
        
        Args:
            inputs: Inputs
            target: Target class
            baselines: Baselines
            
        Returns:
            Atribuciones
        """
        attributions = self.gs.attribute(inputs, baselines, target=target)
        return attributions
    
    def saliency_map(
        self,
        inputs: torch.Tensor,
        target: int
    ) -> torch.Tensor:
        """
        Calcular Saliency Map.
        
        Args:
            inputs: Inputs
            target: Target class
            
        Returns:
            Saliency map
        """
        attributions = self.saliency.attribute(inputs, target=target)
        return attributions




