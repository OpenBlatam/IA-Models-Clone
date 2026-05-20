"""
Performance Generator - Generador de utilidades de performance y profiling
===========================================================================

Genera módulos para profiling, optimización y análisis de performance.
Incluye optimizaciones de inferencia rápida.
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PerformanceGenerator:
    """Generador de utilidades de performance"""
    
    def __init__(self):
        """Inicializa el generador de performance"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de performance.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar módulos de performance
        self._generate_profiler(utils_dir, keywords, project_info)
        self._generate_optimizer(utils_dir, keywords, project_info)
        # Generar optimizaciones de inferencia
        from .inference_optimizer import InferenceOptimizer
        inference_optimizer = InferenceOptimizer()
        inference_optimizer.generate(utils_dir, keywords, project_info)
        # Generar optimizaciones de batch
        from .batch_optimizer import BatchOptimizer
        batch_optimizer = BatchOptimizer()
        batch_optimizer.generate(utils_dir, keywords, project_info)
        # Generar optimizaciones de memoria
        from .memory_optimizer import MemoryOptimizer
        memory_optimizer = MemoryOptimizer()
        memory_optimizer.generate(utils_dir, keywords, project_info)
        # Generar optimizaciones de velocidad ultra agresivas
        from .speed_optimizer import SpeedOptimizer
        speed_optimizer = SpeedOptimizer()
        speed_optimizer.generate(utils_dir, keywords, project_info)
        self._generate_performance_init(utils_dir, keywords)
    
    def _generate_performance_init(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """Genera __init__.py del módulo de performance"""
        
        init_content = '''"""
Performance Utilities Module
=============================

Utilidades para profiling y optimización de performance.
"""

from .profiler import ModelProfiler, profile_model, profile_training
from .optimizer import optimize_model, get_optimization_suggestions
from .inference_optimizer import (
    FastInferenceModel,
    optimize_for_inference,
)
from .quantization_utils import quantize_model, get_model_size_mb
from .caching_utils import InferenceCache, cached_tokenize
from .dynamic_batching import DynamicBatcher
from .batch_utils import (
    collate_batch,
    process_batch_fast,
    split_large_batch,
)
from .memory_optimizer import (
    enable_gradient_checkpointing,
    optimize_memory_for_training,
    get_memory_usage,
    clear_memory_cache,
)
from .speed_optimizer import (
    apply_maximum_speed_optimizations,
    optimize_transformer_for_speed,
    optimize_diffusion_for_speed,
    enable_fast_inference_mode,
)

__all__ = [
    "ModelProfiler",
    "profile_model",
    "profile_training",
    "optimize_model",
    "get_optimization_suggestions",
    "FastInferenceModel",
    "optimize_for_inference",
    "quantize_model",
    "get_model_size_mb",
    "InferenceCache",
    "cached_tokenize",
    "DynamicBatcher",
    "collate_batch",
    "process_batch_fast",
    "split_large_batch",
    "enable_gradient_checkpointing",
    "optimize_memory_for_training",
    "get_memory_usage",
    "clear_memory_cache",
    "apply_maximum_speed_optimizations",
    "optimize_transformer_for_speed",
    "optimize_diffusion_for_speed",
    "enable_fast_inference_mode",
]
'''
        
        perf_dir = utils_dir / "performance"
        perf_dir.mkdir(parents=True, exist_ok=True)
        (perf_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    
    def _generate_profiler(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de profiling"""
        
        profiler_content = '''"""
Model Profiler - Profiling de modelos y entrenamiento
======================================================

Utilidades para analizar performance de modelos y identificar bottlenecks.
"""

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Dict, Any, Optional, List
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ModelProfiler:
    """
    Profiler para modelos de PyTorch.
    
    Permite analizar:
    - Tiempo de ejecución
    - Uso de memoria
    - Operaciones más costosas
    - Bottlenecks en el modelo
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        use_cuda_profiling: bool = True,
    ):
        """
        Inicializa el profiler.
        
        Args:
            model: Modelo a perfilar
            device: Dispositivo a usar
            use_cuda_profiling: Si usar profiling de CUDA
        """
        self.model = model
        self.device = device
        self.use_cuda_profiling = use_cuda_profiling and device == "cuda"
        
        # Actividades a perfilar
        self.activities = [ProfilerActivity.CPU]
        if self.use_cuda_profiling:
            self.activities.append(ProfilerActivity.CUDA)
    
    @contextmanager
    def profile_forward(self, *args, **kwargs):
        """
        Context manager para perfilar forward pass.
        
        Args:
            *args: Argumentos para forward
            **kwargs: Keyword arguments para forward
        
        Yields:
            Resultado del forward pass
        """
        with profile(
            activities=self.activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with record_function("model_forward"):
                output = self.model(*args, **kwargs)
            
            yield output, prof
    
    def profile_inference(
        self,
        input_data,
        num_runs: int = 10,
        warmup_runs: int = 3,
    ) -> Dict[str, Any]:
        """
        Perfila inferencia del modelo.
        
        Args:
            input_data: Datos de entrada
            num_runs: Número de ejecuciones para promediar
            warmup_runs: Número de ejecuciones de warmup
        
        Returns:
            Diccionario con métricas de performance
        """
        self.model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(input_data)
        
        # Sincronizar GPU si es necesario
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Medir tiempo
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                start_time = time.time()
                _ = self.model(input_data)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Medir memoria si es CUDA
        memory_stats = {}
        if self.device == "cuda":
            memory_stats = {
                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            }
        
        return {
            "avg_inference_time_ms": avg_time * 1000,
            "min_inference_time_ms": min_time * 1000,
            "max_inference_time_ms": max_time * 1000,
            "throughput_samples_per_sec": 1.0 / avg_time if avg_time > 0 else 0,
            "memory_stats": memory_stats,
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen del modelo.
        
        Returns:
            Diccionario con información del modelo
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Calcular tamaño aproximado en MB
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024**2
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "model_size_mb": model_size_mb,
            "device": self.device,
        }


def profile_model(
    model: nn.Module,
    input_data,
    device: str = "cuda",
    num_runs: int = 10,
) -> Dict[str, Any]:
    """
    Función helper para perfilar un modelo.
    
    Args:
        model: Modelo a perfilar
        input_data: Datos de entrada
        device: Dispositivo a usar
        num_runs: Número de ejecuciones
    
    Returns:
        Diccionario con métricas de performance
    """
    profiler = ModelProfiler(model, device=device)
    return profiler.profile_inference(input_data, num_runs=num_runs)


def profile_training(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    num_batches: int = 5,
) -> Dict[str, Any]:
    """
    Perfila un loop de entrenamiento.
    
    Args:
        model: Modelo a entrenar
        dataloader: DataLoader con datos
        device: Dispositivo a usar
        num_batches: Número de batches a perfilar
    
    Returns:
        Diccionario con métricas de performance
    """
    model.train()
    profiler = ModelProfiler(model, device=device)
    
    batch_times = []
    data_loading_times = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        # Medir tiempo de carga de datos
        data_start = time.time()
        if isinstance(batch, dict):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        else:
            batch = batch.to(device)
        
        if device == "cuda":
            torch.cuda.synchronize()
        data_loading_times.append(time.time() - data_start)
        
        # Medir tiempo de forward/backward
        batch_start = time.time()
        with profiler.profile_forward(batch) as (output, prof):
            loss = output if isinstance(output, torch.Tensor) else output.loss
            loss.backward()
        
        if device == "cuda":
            torch.cuda.synchronize()
        batch_times.append(time.time() - batch_start)
    
    return {
        "avg_batch_time_ms": (sum(batch_times) / len(batch_times)) * 1000,
        "avg_data_loading_time_ms": (sum(data_loading_times) / len(data_loading_times)) * 1000,
        "data_loading_percentage": (
            sum(data_loading_times) / (sum(batch_times) + sum(data_loading_times))
        ) * 100,
    }
'''
        
        perf_dir = utils_dir / "performance"
        (perf_dir / "profiler.py").write_text(profiler_content, encoding="utf-8")
    
    def _generate_optimizer(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de optimización"""
        
        optimizer_content = '''"""
Model Optimizer - Optimización de modelos
==========================================

Utilidades para optimizar modelos y sugerir mejoras de performance.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def optimize_model(
    model: nn.Module,
    device: str = "cuda",
    enable_optimizations: bool = True,
    compile_mode: str = "reduce-overhead",
    use_torch_compile: bool = True,
) -> nn.Module:
    """
    Aplica optimizaciones al modelo para máxima velocidad.
    
    Args:
        model: Modelo a optimizar
        device: Dispositivo a usar
        enable_optimizations: Si habilitar optimizaciones
        compile_mode: Modo de compilación (default, reduce-overhead, max-autotune)
        use_torch_compile: Si usar torch.compile
    
    Returns:
        Modelo optimizado
    """
    if not enable_optimizations:
        return model
    
    # Optimizaciones específicas de CUDA
    if device == "cuda":
        # Habilitar cuDNN benchmarking para tamaños de input constantes
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # Más rápido
        logger.info("cuDNN benchmarking habilitado")
        
        # Habilitar optimizaciones de memoria
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 habilitado para mayor velocidad")
    
    # Compilar modelo con torch.compile si está disponible (PyTorch 2.0+)
    if use_torch_compile:
        try:
            if hasattr(torch, "compile"):
                # Usar modo más agresivo para máxima velocidad
                if compile_mode == "max-autotune":
                    model = torch.compile(
                        model,
                        mode="max-autotune",
                        fullgraph=True,  # Gráfico completo para mejor optimización
                    )
                else:
                    model = torch.compile(
                        model,
                        mode=compile_mode,
                        fullgraph=False,  # Permite gráficos dinámicos
                    )
                logger.info(f"Modelo compilado con torch.compile (mode={compile_mode})")
        except Exception as e:
            logger.warning(f"No se pudo compilar modelo: {e}")
    
    # Optimizaciones adicionales de JIT si compile falla
    if not use_torch_compile or not hasattr(torch, "compile"):
        try:
            # Intentar JIT script para modelos compatibles
            if hasattr(model, "forward"):
                model = torch.jit.optimize_for_inference(torch.jit.script(model))
                logger.info("Modelo optimizado con JIT script")
        except:
            try:
                # Fallback a JIT trace
                dummy_input = torch.randn(1, 3, 224, 224).to(device) if device == "cuda" else torch.randn(1, 3, 224, 224)
                model = torch.jit.trace(model, dummy_input)
                model = torch.jit.optimize_for_inference(model)
                logger.info("Modelo optimizado con JIT trace")
            except:
                pass  # JIT no es compatible con todos los modelos
    
    # Optimizar para inferencia
    model.eval()
    if device == "cuda":
        # Fusionar operaciones cuando sea posible
        try:
            torch.jit.optimize_for_inference(model)
            logger.info("Modelo optimizado para inferencia")
        except:
            pass  # No todos los modelos soportan JIT
    
    return model


def get_optimization_suggestions(
    model: nn.Module,
    profiler_results: Dict[str, Any],
) -> List[str]:
    """
    Genera sugerencias de optimización basadas en profiling.
    
    Args:
        model: Modelo analizado
        profiler_results: Resultados del profiling
    
    Returns:
        Lista de sugerencias
    """
    suggestions = []
    
    # Analizar tiempo de inferencia
    avg_time = profiler_results.get("avg_inference_time_ms", 0)
    if avg_time > 100:
        suggestions.append(
            f"Inferencia lenta ({avg_time:.2f}ms). "
            "Considera usar torch.compile o reducir tamaño del modelo."
        )
    
    # Analizar memoria
    memory_stats = profiler_results.get("memory_stats", {})
    allocated_mb = memory_stats.get("allocated_mb", 0)
    if allocated_mb > 8000:  # > 8GB
        suggestions.append(
            f"Uso alto de memoria ({allocated_mb:.2f}MB). "
            "Considera usar gradient checkpointing o mixed precision."
        )
    
    # Analizar data loading
    data_loading_pct = profiler_results.get("data_loading_percentage", 0)
    if data_loading_pct > 30:
        suggestions.append(
            f"Data loading toma {data_loading_pct:.1f}% del tiempo. "
            "Considera aumentar num_workers o usar prefetching."
        )
    
    # Analizar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    if total_params > 100_000_000:  # > 100M
        suggestions.append(
            f"Modelo grande ({total_params:,} parámetros). "
            "Considera usar LoRA o pruning para reducir tamaño."
        )
    
    if not suggestions:
        suggestions.append("Modelo está bien optimizado. No se encontraron problemas.")
    
    return suggestions
'''
        
        perf_dir = utils_dir / "performance"
        (perf_dir / "optimizer.py").write_text(optimizer_content, encoding="utf-8")

