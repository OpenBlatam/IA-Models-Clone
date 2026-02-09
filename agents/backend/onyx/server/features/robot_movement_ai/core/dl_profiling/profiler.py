"""
Model Profiler - Modular Profiling
==================================

Profiling modular para modelos y código.
"""

import logging
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class CodeProfiler:
    """Profiler de código."""
    
    def __init__(self):
        """Inicializar profiler."""
        self.timings: Dict[str, List[float]] = {}
        self.counts: Dict[str, int] = {}
    
    @contextmanager
    def profile(self, name: str):
        """
        Context manager para profiling.
        
        Args:
            name: Nombre de la operación
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            if name not in self.timings:
                self.timings[name] = []
                self.counts[name] = 0
            self.timings[name].append(duration)
            self.counts[name] += 1
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Obtener estadísticas de una operación.
        
        Args:
            name: Nombre de la operación
            
        Returns:
            Estadísticas
        """
        if name not in self.timings:
            return {}
        
        times = self.timings[name]
        return {
            'count': self.counts[name],
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'std': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Obtener estadísticas de todas las operaciones."""
        return {name: self.get_stats(name) for name in self.timings.keys()}
    
    def reset(self):
        """Resetear profiler."""
        self.timings.clear()
        self.counts.clear()


class ModelProfiler:
    """Profiler de modelos."""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Inicializar profiler de modelo.
        
        Args:
            model: Modelo a perfilar
            device: Dispositivo
        """
        self.model = model
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
    
    def profile_forward(
        self,
        input_shape: tuple,
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, Any]:
        """
        Perfilar forward pass.
        
        Args:
            input_shape: Forma de entrada
            num_iterations: Número de iteraciones
            warmup: Iteraciones de warmup
            
        Returns:
            Resultados del profiling
        """
        try:
            import torch.profiler as profiler
            
            dummy_input = torch.randn(1, *input_shape).to(self.device)
            self.model.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(warmup):
                    _ = self.model(dummy_input)
            
            # Profiling
            with torch.no_grad():
                with profiler.profile(
                    activities=[
                        profiler.ProfilerActivity.CPU,
                        profiler.ProfilerActivity.CUDA
                    ] if self.device.type == 'cuda' else [profiler.ProfilerActivity.CPU],
                    record_shapes=True,
                    profile_memory=True
                ) as prof:
                    for _ in range(num_iterations):
                        _ = self.model(dummy_input)
            
            # Obtener resultados
            events = prof.key_averages()
            
            return {
                'success': True,
                'events': [
                    {
                        'name': event.key,
                        'cpu_time': event.cpu_time_total_us / 1000,  # ms
                        'cuda_time': event.cuda_time_total_us / 1000 if self.device.type == 'cuda' else 0,  # ms
                        'count': event.count,
                        'cpu_memory': event.cpu_memory_usage / 1024**2,  # MB
                        'cuda_memory': event.cuda_memory_usage / 1024**2 if self.device.type == 'cuda' else 0  # MB
                    }
                    for event in events
                ]
            }
        except Exception as e:
            logger.error(f"Error in profiling: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def profile_memory(
        self,
        input_shape: tuple,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        Perfilar uso de memoria.
        
        Args:
            input_shape: Forma de entrada
            batch_size: Tamaño de batch
            
        Returns:
            Resultados del profiling
        """
        try:
            if self.device.type != 'cuda':
                return {
                    'success': False,
                    'error': 'Memory profiling only available for CUDA'
                }
            
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
            
            dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                output = self.model(dummy_input)
            
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
            reserved = torch.cuda.memory_reserved(self.device) / 1024**2  # MB
            peak = torch.cuda.max_memory_allocated(self.device) / 1024**2  # MB
            
            # Información del modelo
            num_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            param_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2
            
            return {
                'success': True,
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'peak_mb': peak,
                'num_parameters': num_params,
                'trainable_parameters': trainable_params,
                'parameter_size_mb': param_size_mb,
                'input_size_mb': dummy_input.numel() * dummy_input.element_size() / 1024**2,
                'output_size_mb': output.numel() * output.element_size() / 1024**2
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }








