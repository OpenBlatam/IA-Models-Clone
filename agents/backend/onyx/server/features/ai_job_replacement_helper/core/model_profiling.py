"""
Model Profiling Service - Profiling de modelos
==============================================

Sistema para profiling y análisis de rendimiento de modelos.
Sigue mejores prácticas de PyTorch profiling.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.profiler import profile, record_function, ProfilerActivity
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


@dataclass
class ProfileResult:
    """Resultado de profiling"""
    total_time: float
    forward_time: float
    backward_time: Optional[float] = None
    memory_allocated: float = 0.0  # MB
    memory_reserved: float = 0.0  # MB
    flops: Optional[int] = None
    num_parameters: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


class ModelProfilingService:
    """Servicio de profiling de modelos"""
    
    def __init__(self):
        """Inicializar servicio"""
        logger.info(f"ModelProfilingService initialized (PyTorch: {TORCH_AVAILABLE})")
    
    def profile_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
        num_runs: int = 10,
        warmup_runs: int = 3,
        use_profiler: bool = True
    ) -> ProfileResult:
        """
        Hacer profiling de un modelo.
        
        Args:
            model: Modelo a perfilar
            input_shape: Forma del input (sin batch dimension)
            device: Dispositivo (None = auto)
            num_runs: Número de runs para promediar
            warmup_runs: Runs de warmup
            use_profiler: Usar torch.profiler (más detallado pero más lento)
        
        Returns:
            ProfileResult con métricas
        """
        if not TORCH_AVAILABLE:
            return ProfileResult(total_time=0.0, forward_time=0.0)
        
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)
        
        # Sync if CUDA
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Profile forward pass
        forward_times = []
        memory_allocated = 0.0
        memory_reserved = 0.0
        
        if use_profiler and device.type == "cuda":
            # Use torch.profiler for detailed profiling
            try:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                ) as prof:
                    with record_function("model_forward"):
                        with torch.no_grad():
                            _ = model(dummy_input)
                
                # Get timing
                events = prof.key_averages()
                forward_time = sum(event.cuda_time_total for event in events) / 1000.0  # ms to s
                forward_times.append(forward_time)
                
                # Get memory
                for event in events:
                    if hasattr(event, "cuda_memory_usage"):
                        memory_allocated = max(memory_allocated, event.cuda_memory_usage / 1024**2)  # MB
                
            except Exception as e:
                logger.warning(f"Error using profiler: {e}, falling back to simple timing")
                use_profiler = False
        
        if not use_profiler:
            # Simple timing
            for _ in range(num_runs):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                
                start_time = time.time()
                with torch.no_grad():
                    _ = model(dummy_input)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    memory_allocated = max(
                        memory_allocated,
                        torch.cuda.max_memory_allocated() / 1024**2
                    )
                    memory_reserved = max(
                        memory_reserved,
                        torch.cuda.max_memory_reserved() / 1024**2
                    )
                
                forward_time = time.time() - start_time
                forward_times.append(forward_time)
        
        avg_forward_time = sum(forward_times) / len(forward_times)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        # Estimate FLOPs (simplified)
        flops = self._estimate_flops(model, input_shape)
        
        return ProfileResult(
            total_time=avg_forward_time,
            forward_time=avg_forward_time,
            memory_allocated=memory_allocated,
            memory_reserved=memory_reserved,
            flops=flops,
            num_parameters=num_params,
            details={
                "device": str(device),
                "input_shape": input_shape,
                "num_runs": num_runs,
            }
        )
    
    def _estimate_flops(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...]
    ) -> Optional[int]:
        """
        Estimar FLOPs del modelo (simplificado).
        
        Args:
            model: Modelo
            input_shape: Forma del input
        
        Returns:
            FLOPs estimados
        """
        try:
            # Try to use fvcore if available
            try:
                from fvcore.nn import FlopCountMode, flop_count
                dummy_input = torch.randn(1, *input_shape)
                flops_dict, _ = flop_count(model, (dummy_input,))
                total_flops = sum(flops_dict.values())
                return int(total_flops)
            except ImportError:
                # Fallback: simple estimation
                # This is a very rough estimate
                num_params = sum(p.numel() for p in model.parameters())
                batch_size = 1
                # Rough estimate: 2 * params * batch_size
                return int(2 * num_params * batch_size)
        except Exception as e:
            logger.warning(f"Error estimating FLOPs: {e}")
            return None
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        input_shape: Tuple[int, ...],
        device: Optional[torch.device] = None
    ) -> Dict[str, ProfileResult]:
        """
        Comparar múltiples modelos.
        
        Args:
            models: Diccionario de modelos {name: model}
            input_shape: Forma del input
            device: Dispositivo
        
        Returns:
            Diccionario de resultados {name: ProfileResult}
        """
        results = {}
        
        for name, model in models.items():
            logger.info(f"Profiling model: {name}")
            try:
                result = self.profile_model(model, input_shape, device)
                results[name] = result
            except Exception as e:
                logger.error(f"Error profiling {name}: {e}", exc_info=True)
                results[name] = ProfileResult(total_time=0.0, forward_time=0.0)
        
        return results
    
    def profile_training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        sample_input: torch.Tensor,
        sample_target: torch.Tensor,
        device: Optional[torch.device] = None,
        num_runs: int = 5
    ) -> ProfileResult:
        """
        Hacer profiling de un step de entrenamiento completo.
        
        Args:
            model: Modelo
            optimizer: Optimizador
            criterion: Función de pérdida
            sample_input: Input de ejemplo
            sample_target: Target de ejemplo
            device: Dispositivo
            num_runs: Número de runs
        
        Returns:
            ProfileResult
        """
        if not TORCH_AVAILABLE:
            return ProfileResult(total_time=0.0, forward_time=0.0)
        
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()
        
        sample_input = sample_input.to(device)
        sample_target = sample_target.to(device)
        
        forward_times = []
        backward_times = []
        total_times = []
        
        for _ in range(num_runs):
            optimizer.zero_grad()
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            # Forward
            start_forward = time.time()
            outputs = model(sample_input)
            loss = criterion(outputs, sample_target)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_time = time.time() - start_forward
            
            # Backward
            start_backward = time.time()
            loss.backward()
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            backward_time = time.time() - start_backward
            
            optimizer.step()
            
            total_time = time.time() - start_forward
            forward_times.append(forward_time)
            backward_times.append(backward_time)
            total_times.append(total_time)
        
        # Get memory
        memory_allocated = 0.0
        memory_reserved = 0.0
        if device.type == "cuda":
            memory_allocated = torch.cuda.max_memory_allocated() / 1024**2
            memory_reserved = torch.cuda.max_memory_reserved() / 1024**2
        
        return ProfileResult(
            total_time=sum(total_times) / len(total_times),
            forward_time=sum(forward_times) / len(forward_times),
            backward_time=sum(backward_times) / len(backward_times),
            memory_allocated=memory_allocated,
            memory_reserved=memory_reserved,
        )




