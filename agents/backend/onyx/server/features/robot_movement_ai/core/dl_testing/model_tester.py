"""
Model Tester - Modular Model Testing
====================================

Testing modular para modelos de deep learning.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

logger = logging.getLogger(__name__)


class ModelTester:
    """
    Tester modular para modelos.
    
    Proporciona tests unitarios y de integración
    para modelos de deep learning.
    """
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Inicializar tester.
        
        Args:
            model: Modelo a testear
            device: Dispositivo
        """
        self.model = model
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def test_forward_pass(
        self,
        input_shape: tuple,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        Test de forward pass.
        
        Args:
            input_shape: Forma de entrada
            batch_size: Tamaño de batch
            
        Returns:
            Resultados del test
        """
        try:
            # Crear input dummy
            dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                output = self.model(dummy_input)
            
            return {
                'success': True,
                'input_shape': dummy_input.shape,
                'output_shape': output.shape,
                'output_dtype': str(output.dtype)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_gradient_flow(
        self,
        input_shape: tuple,
        loss_fn: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Test de flujo de gradientes.
        
        Args:
            input_shape: Forma de entrada
            loss_fn: Función de pérdida
            
        Returns:
            Resultados del test
        """
        try:
            if loss_fn is None:
                loss_fn = nn.MSELoss()
            
            self.model.train()
            dummy_input = torch.randn(1, *input_shape).to(self.device)
            dummy_target = torch.randn_like(self.model(dummy_input))
            
            # Forward y backward
            output = self.model(dummy_input)
            loss = loss_fn(output, dummy_target)
            loss.backward()
            
            # Verificar gradientes
            has_gradients = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in self.model.parameters()
            )
            
            # Verificar NaN/Inf
            has_nan = any(
                torch.isnan(p.grad).any() if p.grad is not None else False
                for p in self.model.parameters()
            )
            
            has_inf = any(
                torch.isinf(p.grad).any() if p.grad is not None else False
                for p in self.model.parameters()
            )
            
            return {
                'success': True,
                'has_gradients': has_gradients,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'loss': loss.item()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_inference_speed(
        self,
        input_shape: tuple,
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, Any]:
        """
        Test de velocidad de inferencia.
        
        Args:
            input_shape: Forma de entrada
            num_iterations: Número de iteraciones
            warmup: Iteraciones de warmup
            
        Returns:
            Resultados del test
        """
        try:
            import time
            
            dummy_input = torch.randn(1, *input_shape).to(self.device)
            self.model.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(warmup):
                    _ = self.model(dummy_input)
            
            # Synchronize GPU
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(num_iterations):
                    start = time.time()
                    _ = self.model(dummy_input)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    times.append(time.time() - start)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            return {
                'success': True,
                'avg_time_ms': avg_time * 1000,
                'std_time_ms': std_time * 1000,
                'fps': fps,
                'min_time_ms': np.min(times) * 1000,
                'max_time_ms': np.max(times) * 1000
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_memory_usage(
        self,
        input_shape: tuple,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        Test de uso de memoria.
        
        Args:
            input_shape: Forma de entrada
            batch_size: Tamaño de batch
            
        Returns:
            Resultados del test
        """
        try:
            if self.device.type != 'cuda':
                return {
                    'success': False,
                    'error': 'Memory test only available for CUDA'
                }
            
            torch.cuda.reset_peak_memory_stats(self.device)
            
            dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
            
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
            reserved = torch.cuda.memory_reserved(self.device) / 1024**2  # MB
            peak = torch.cuda.max_memory_allocated(self.device) / 1024**2  # MB
            
            return {
                'success': True,
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'peak_mb': peak
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_all_tests(
        self,
        input_shape: tuple,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Ejecutar todos los tests.
        
        Args:
            input_shape: Forma de entrada
            **kwargs: Argumentos adicionales
            
        Returns:
            Resultados de todos los tests
        """
        results = {
            'forward_pass': self.test_forward_pass(input_shape, **kwargs),
            'gradient_flow': self.test_gradient_flow(input_shape, **kwargs),
            'inference_speed': self.test_inference_speed(input_shape, **kwargs),
            'memory_usage': self.test_memory_usage(input_shape, **kwargs)
        }
        
        results['all_passed'] = all(
            r.get('success', False) for r in results.values()
        )
        
        return results








