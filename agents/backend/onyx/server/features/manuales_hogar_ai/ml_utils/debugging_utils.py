"""
Debugging Utils - Utilidades de Debugging Avanzado
===================================================

Utilidades para debugging y análisis de modelos.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Callable, Any
import numpy as np
from contextlib import contextmanager
import traceback

logger = logging.getLogger(__name__)


class GradientChecker:
    """
    Verificador de gradientes.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar verificador.
        
        Args:
            model: Modelo a verificar
        """
        self.model = model
    
    def check_gradients(
        self,
        threshold: float = 1e-6
    ) -> Dict[str, Dict[str, Any]]:
        """
        Verificar gradientes.
        
        Args:
            threshold: Umbral para gradientes pequeños
            
        Returns:
            Diccionario con información de gradientes
        """
        grad_info = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                grad_info[name] = {
                    'norm': grad.norm().item(),
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'min': grad.min().item(),
                    'max': grad.max().item(),
                    'has_nan': torch.isnan(grad).any().item(),
                    'has_inf': torch.isinf(grad).any().item(),
                    'is_small': grad.norm().item() < threshold
                }
            else:
                grad_info[name] = {
                    'has_grad': False
                }
        
        return grad_info
    
    def detect_vanishing_gradients(
        self,
        threshold: float = 1e-6
    ) -> List[str]:
        """
        Detectar gradientes que desaparecen.
        
        Args:
            threshold: Umbral
            
        Returns:
            Lista de parámetros con gradientes pequeños
        """
        vanishing = []
        grad_info = self.check_gradients(threshold)
        
        for name, info in grad_info.items():
            if 'norm' in info and info['norm'] < threshold:
                vanishing.append(name)
        
        return vanishing
    
    def detect_exploding_gradients(
        self,
        threshold: float = 100.0
    ) -> List[str]:
        """
        Detectar gradientes que explotan.
        
        Args:
            threshold: Umbral
            
        Returns:
            Lista de parámetros con gradientes grandes
        """
        exploding = []
        grad_info = self.check_gradients()
        
        for name, info in grad_info.items():
            if 'norm' in info and info['norm'] > threshold:
                exploding.append(name)
        
        return exploding


class ActivationMonitor:
    """
    Monitor de activaciones.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar monitor.
        
        Args:
            model: Modelo a monitorear
        """
        self.model = model
        self.activations: Dict[str, List[torch.Tensor]] = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Registrar hooks para capturar activaciones."""
        def get_activation_hook(name):
            def hook(module, input, output):
                if name not in self.activations:
                    self.activations[name] = []
                self.activations[name].append(output.detach())
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.GELU)):
                hook = module.register_forward_hook(get_activation_hook(name))
                self.hooks.append(hook)
    
    def get_activation_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Obtener estadísticas de activaciones.
        
        Returns:
            Estadísticas por capa
        """
        stats = {}
        
        for name, activations in self.activations.items():
            if len(activations) > 0:
                all_activations = torch.cat([a.flatten() for a in activations])
                stats[name] = {
                    'mean': all_activations.mean().item(),
                    'std': all_activations.std().item(),
                    'min': all_activations.min().item(),
                    'max': all_activations.max().item(),
                    'dead_ratio': (all_activations == 0).float().mean().item()
                }
        
        return stats
    
    def detect_dead_neurons(
        self,
        threshold: float = 1e-6
    ) -> Dict[str, float]:
        """
        Detectar neuronas muertas.
        
        Args:
            threshold: Umbral
            
        Returns:
            Ratio de neuronas muertas por capa
        """
        dead_ratios = {}
        stats = self.get_activation_stats()
        
        for name, stat in stats.items():
            dead_ratios[name] = stat['dead_ratio']
        
        return dead_ratios
    
    def clear(self):
        """Limpiar activaciones."""
        self.activations.clear()
    
    def remove_hooks(self):
        """Remover hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


@contextmanager
def detect_anomaly():
    """
    Context manager para detectar anomalías en autograd.
    """
    torch.autograd.set_detect_anomaly(True)
    try:
        yield
    finally:
        torch.autograd.set_detect_anomaly(False)


class ModelDebugger:
    """
    Debugger completo de modelos.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar debugger.
        
        Args:
            model: Modelo a debuggear
        """
        self.model = model
        self.gradient_checker = GradientChecker(model)
        self.activation_monitor = ActivationMonitor(model)
    
    def debug_forward(
        self,
        inputs: torch.Tensor,
        check_nan_inf: bool = True,
        check_shapes: bool = True
    ) -> Dict[str, Any]:
        """
        Debuggear forward pass.
        
        Args:
            inputs: Inputs
            check_nan_inf: Verificar NaN/Inf
            check_shapes: Verificar formas
            
        Returns:
            Información de debug
        """
        debug_info = {
            'errors': [],
            'warnings': []
        }
        
        try:
            with detect_anomaly():
                outputs = self.model(inputs)
            
            # Verificar NaN/Inf
            if check_nan_inf:
                if torch.isnan(outputs).any():
                    debug_info['errors'].append("Output contains NaN")
                if torch.isinf(outputs).any():
                    debug_info['warnings'].append("Output contains Inf")
            
            # Verificar formas
            if check_shapes:
                debug_info['input_shape'] = list(inputs.shape)
                debug_info['output_shape'] = list(outputs.shape)
            
            debug_info['output_stats'] = {
                'mean': outputs.mean().item(),
                'std': outputs.std().item(),
                'min': outputs.min().item(),
                'max': outputs.max().item()
            }
        
        except Exception as e:
            debug_info['errors'].append(f"Forward pass failed: {str(e)}")
            debug_info['traceback'] = traceback.format_exc()
        
        return debug_info
    
    def debug_backward(
        self,
        loss: torch.Tensor,
        check_gradients: bool = True
    ) -> Dict[str, Any]:
        """
        Debuggear backward pass.
        
        Args:
            loss: Loss
            check_gradients: Verificar gradientes
            
        Returns:
            Información de debug
        """
        debug_info = {
            'errors': [],
            'warnings': []
        }
        
        try:
            with detect_anomaly():
                loss.backward()
            
            # Verificar gradientes
            if check_gradients:
                grad_info = self.gradient_checker.check_gradients()
                vanishing = self.gradient_checker.detect_vanishing_gradients()
                exploding = self.gradient_checker.detect_exploding_gradients()
                
                if vanishing:
                    debug_info['warnings'].append(
                        f"Vanishing gradients in: {vanishing}"
                    )
                if exploding:
                    debug_info['errors'].append(
                        f"Exploding gradients in: {exploding}"
                    )
                
                debug_info['gradient_info'] = grad_info
        
        except Exception as e:
            debug_info['errors'].append(f"Backward pass failed: {str(e)}")
            debug_info['traceback'] = traceback.format_exc()
        
        return debug_info
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen del modelo.
        
        Returns:
            Resumen del modelo
        """
        summary = {
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            'layers': []
        }
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                summary['layers'].append({
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': sum(p.numel() for p in module.parameters())
                })
        
        return summary




