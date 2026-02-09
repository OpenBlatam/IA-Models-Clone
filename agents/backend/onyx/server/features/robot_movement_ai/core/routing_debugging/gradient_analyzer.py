"""
Gradient Analyzer
=================

Análisis de gradientes para debugging.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GradientAnalyzer:
    """
    Analizador de gradientes.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar analizador.
        
        Args:
            model: Modelo
        """
        self.model = model
        self.gradient_history = []
    
    def analyze_gradients(self) -> Dict[str, Any]:
        """
        Analizar gradientes del modelo.
        
        Returns:
            Análisis de gradientes
        """
        analysis = {
            "gradient_norms": {},
            "gradient_stats": {},
            "vanishing_gradients": [],
            "exploding_gradients": [],
            "zero_gradients": []
        }
        
        total_norm = 0.0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                norm = grad.norm().item()
                total_norm += norm ** 2
                
                analysis["gradient_norms"][name] = norm
                analysis["gradient_stats"][name] = {
                    "mean": grad.mean().item(),
                    "std": grad.std().item(),
                    "min": grad.min().item(),
                    "max": grad.max().item()
                }
                
                # Detectar problemas
                if norm < 1e-6:
                    analysis["vanishing_gradients"].append(name)
                elif norm > 100.0:
                    analysis["exploding_gradients"].append(name)
                elif norm == 0.0:
                    analysis["zero_gradients"].append(name)
        
        analysis["total_norm"] = total_norm ** 0.5
        
        return analysis
    
    def log_gradient_info(self, step: int = 0):
        """
        Loggear información de gradientes.
        
        Args:
            step: Paso actual
        """
        analysis = self.analyze_gradients()
        
        logger.info(f"Step {step} - Gradient Norm: {analysis['total_norm']:.4f}")
        
        if analysis["vanishing_gradients"]:
            logger.warning(f"Vanishing gradients en: {analysis['vanishing_gradients']}")
        
        if analysis["exploding_gradients"]:
            logger.warning(f"Exploding gradients en: {analysis['exploding_gradients']}")
        
        if analysis["zero_gradients"]:
            logger.warning(f"Zero gradients en: {analysis['zero_gradients']}")


def analyze_gradients(model: nn.Module) -> Dict[str, Any]:
    """
    Función helper para analizar gradientes.
    
    Args:
        model: Modelo
        
    Returns:
        Análisis
    """
    analyzer = GradientAnalyzer(model)
    return analyzer.analyze_gradients()

