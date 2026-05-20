"""
Profiling and Optimization
===========================

Utilidades para profiling y optimización de modelos.
"""

import logging
from typing import Dict, Any, Optional, List
import time

try:
    import torch
    import torch.nn as nn
    from torch.profiler import profile, record_function, ProfilerActivity
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    profile = None
    record_function = None
    ProfilerActivity = None

logger = logging.getLogger(__name__)


class ModelProfiler:
    """
    Profiler de modelos.
    
    Analiza rendimiento y cuellos de botella.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Inicializar profiler.
        
        Args:
            device: Dispositivo
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available")
            return
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.profile_data: List[Dict[str, Any]] = []
    
    def profile_model(
        self,
        model: nn.Module,
        input_shape: tuple,
        num_iterations: int = 10,
        warmup: int = 3
    ) -> Dict[str, Any]:
        """
        Profilear modelo.
        
        Args:
            model: Modelo
            input_shape: Forma de entrada
            num_iterations: Número de iteraciones
            warmup: Iteraciones de warmup
            
        Returns:
            Estadísticas de profiling
        """
        if not TORCH_AVAILABLE:
            return {}
        
        model.eval()
        model.to(self.device)
        
        # Crear input dummy
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
        
        # Sincronizar GPU
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Profiling
        times = []
        memory_used = []
        
        with torch.no_grad():
            for i in range(num_iterations):
                if self.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    
                    start.record()
                    _ = model(dummy_input)
                    end.record()
                    torch.cuda.synchronize()
                    
                    elapsed = start.elapsed_time(end)  # ms
                    memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                else:
                    start = time.time()
                    _ = model(dummy_input)
                    elapsed = (time.time() - start) * 1000  # ms
                    memory = 0
                
                times.append(elapsed)
                memory_used.append(memory)
        
        stats = {
            "mean_time_ms": sum(times) / len(times),
            "std_time_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "mean_memory_mb": sum(memory_used) / len(memory_used) if memory_used else 0,
            "max_memory_mb": max(memory_used) if memory_used else 0,
            "device": self.device
        }
        
        self.profile_data.append(stats)
        return stats
    
    def profile_with_torch_profiler(
        self,
        model: nn.Module,
        input_shape: tuple,
        activities: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Profilear con torch.profiler.
        
        Args:
            model: Modelo
            input_shape: Forma de entrada
            activities: Actividades a profilear
            
        Returns:
            Estadísticas
        """
        if not TORCH_AVAILABLE or profile is None:
            return {}
        
        model.eval()
        model.to(self.device)
        
        dummy_input = torch.randn(input_shape).to(self.device)
        
        if activities is None:
            if self.device == "cuda":
                activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            else:
                activities = [ProfilerActivity.CPU]
        
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True
        ) as prof:
            with record_function("model_inference"):
                _ = model(dummy_input)
        
        # Exportar resultados
        return {
            "profiler_output": prof.key_averages().table(sort_by="cuda_time_total" if self.device == "cuda" else "cpu_time_total"),
            "activities": [str(a) for a in activities]
        }
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        input_shape: tuple
    ) -> Dict[str, Dict[str, Any]]:
        """
        Comparar múltiples modelos.
        
        Args:
            models: Diccionario de modelos
            input_shape: Forma de entrada
            
        Returns:
            Comparación de estadísticas
        """
        results = {}
        
        for name, model in models.items():
            stats = self.profile_model(model, input_shape)
            results[name] = stats
        
        return results


class ModelOptimizer:
    """
    Optimizador de modelos.
    
    Aplica optimizaciones como fusion, quantization, etc.
    """
    
    @staticmethod
    def fuse_conv_bn(model: nn.Module) -> nn.Module:
        """
        Fusionar Conv + BN.
        
        Args:
            model: Modelo
            
        Returns:
            Modelo optimizado
        """
        if not TORCH_AVAILABLE:
            return model
        
        try:
            from torch.quantization import fuse_modules
            # Ejemplo para módulos específicos
            # En producción, identificar y fusionar automáticamente
            return model
        except Exception as e:
            logger.warning(f"Could not fuse modules: {e}")
            return model
    
    @staticmethod
    def optimize_for_inference(model: nn.Module) -> nn.Module:
        """
        Optimizar modelo para inferencia.
        
        Args:
            model: Modelo
            
        Returns:
            Modelo optimizado
        """
        if not TORCH_AVAILABLE:
            return model
        
        model.eval()
        
        # JIT compile si es posible
        try:
            # Intentar trazar modelo
            dummy_input = torch.randn(1, 3, 224, 224)
            traced = torch.jit.trace(model, dummy_input)
            return traced
        except Exception as e:
            logger.warning(f"Could not JIT compile: {e}")
            return model

