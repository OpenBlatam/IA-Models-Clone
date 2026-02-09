"""
Model Debugger - Modular Debugging
==================================

Debugging modular para modelos de deep learning.
"""

import logging
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class ModelDebugger:
    """
    Debugger modular para modelos.
    
    Proporciona herramientas de debugging para
    identificar problemas en modelos.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar debugger.
        
        Args:
            model: Modelo a debuggear
        """
        self.model = model
        self.hooks = []
        self.activations = {}
        self.gradients = {}
    
    def register_hooks(self):
        """Registrar hooks para activaciones y gradientes."""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = {
                    'input': input[0].detach() if isinstance(input, tuple) else input.detach(),
                    'output': output.detach() if isinstance(output, torch.Tensor) else output
                }
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.gradients[name] = grad_output[0].detach()
            return hook
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Solo módulos hoja
                hook_f = module.register_forward_hook(forward_hook(name))
                hook_b = module.register_full_backward_hook(backward_hook(name))
                self.hooks.append((hook_f, hook_b))
        
        logger.info(f"Registered hooks for {len(self.hooks)} modules")
    
    def remove_hooks(self):
        """Remover hooks."""
        for hook_f, hook_b in self.hooks:
            hook_f.remove()
            hook_b.remove()
        self.hooks = []
        logger.info("Hooks removed")
    
    def check_activations(
        self,
        input_tensor: torch.Tensor,
        check_nan: bool = True,
        check_inf: bool = True,
        check_zero: bool = False
    ) -> Dict[str, Any]:
        """
        Verificar activaciones.
        
        Args:
            input_tensor: Tensor de entrada
            check_nan: Verificar NaN
            check_inf: Verificar Inf
            check_zero: Verificar ceros
            
        Returns:
            Resultados de verificación
        """
        self.activations.clear()
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        issues = {}
        for name, activation in self.activations.items():
            output = activation.get('output')
            if isinstance(output, torch.Tensor):
                layer_issues = []
                
                if check_nan and torch.isnan(output).any():
                    layer_issues.append("Contains NaN")
                
                if check_inf and torch.isinf(output).any():
                    layer_issues.append("Contains Inf")
                
                if check_zero and (output == 0).all():
                    layer_issues.append("All zeros")
                
                if layer_issues:
                    issues[name] = {
                        'issues': layer_issues,
                        'shape': tuple(output.shape),
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'min': output.min().item(),
                        'max': output.max().item()
                    }
        
        return {
            'has_issues': len(issues) > 0,
            'issues': issues
        }
    
    def check_gradients(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        loss_fn: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Verificar gradientes.
        
        Args:
            input_tensor: Tensor de entrada
            target_tensor: Tensor objetivo
            loss_fn: Función de pérdida
            
        Returns:
            Resultados de verificación
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        self.gradients.clear()
        self.model.train()
        
        output = self.model(input_tensor)
        loss = loss_fn(output, target_tensor)
        loss.backward()
        
        issues = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                layer_issues = []
                
                if torch.isnan(grad).any():
                    layer_issues.append("Contains NaN")
                
                if torch.isinf(grad).any():
                    layer_issues.append("Contains Inf")
                
                if grad.abs().sum() == 0:
                    layer_issues.append("Zero gradients")
                
                if grad.abs().max() > 100:
                    layer_issues.append("Exploding gradients")
                
                if layer_issues:
                    issues[name] = {
                        'issues': layer_issues,
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'max': grad.abs().max().item()
                    }
        
        return {
            'has_issues': len(issues) > 0,
            'issues': issues,
            'loss': loss.item()
        }
    
    def detect_anomalies(
        self,
        input_tensor: torch.Tensor,
        target_tensor: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Detectar anomalías usando autograd.
        
        Args:
            input_tensor: Tensor de entrada
            target_tensor: Tensor objetivo (opcional)
            
        Returns:
            Resultados de detección
        """
        try:
            with torch.autograd.detect_anomaly():
                if target_tensor is not None:
                    self.model.train()
                    output = self.model(input_tensor)
                    loss = nn.MSELoss()(output, target_tensor)
                    loss.backward()
                    return {'anomaly_detected': False, 'loss': loss.item()}
                else:
                    self.model.eval()
                    with torch.no_grad():
                        output = self.model(input_tensor)
                    return {'anomaly_detected': False, 'output_shape': output.shape}
        except RuntimeError as e:
            if "anomaly detected" in str(e).lower():
                return {'anomaly_detected': True, 'error': str(e)}
            raise
    
    def visualize_computation_graph(
        self,
        input_tensor: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """
        Visualizar grafo de computación.
        
        Args:
            input_tensor: Tensor de entrada
            save_path: Ruta para guardar (opcional)
        """
        try:
            from torchviz import make_dot
            
            output = self.model(input_tensor)
            dot = make_dot(output, params=dict(self.model.named_parameters()))
            
            if save_path:
                dot.render(save_path, format='png')
                logger.info(f"Computation graph saved to {save_path}")
            else:
                dot.view()
        except ImportError:
            logger.warning("torchviz not available")
        except Exception as e:
            logger.error(f"Error visualizing computation graph: {e}")








