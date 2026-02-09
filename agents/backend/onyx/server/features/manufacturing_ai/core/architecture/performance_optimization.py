"""
Performance Optimization
========================

Optimizaciones avanzadas de rendimiento para inferencia y entrenamiento.
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from functools import lru_cache
import time

try:
    import torch
    import torch.nn as nn
    import torch.jit as jit
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    jit = None

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Cache de modelos compilados.
    
    Almacena modelos JIT compilados para inferencia rápida.
    """
    
    def __init__(self, max_size: int = 10):
        """
        Inicializar cache.
        
        Args:
            max_size: Tamaño máximo del cache
        """
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.access_times: Dict[str, float] = {}
    
    def get(self, model_id: str) -> Optional[Any]:
        """
        Obtener modelo del cache.
        
        Args:
            model_id: ID del modelo
            
        Returns:
            Modelo compilado o None
        """
        if model_id in self.cache:
            self.access_times[model_id] = time.time()
            return self.cache[model_id]
        return None
    
    def put(self, model_id: str, compiled_model: Any):
        """
        Agregar modelo al cache.
        
        Args:
            model_id: ID del modelo
            compiled_model: Modelo compilado
        """
        if len(self.cache) >= self.max_size:
            # Eliminar menos usado
            oldest = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[oldest]
            del self.access_times[oldest]
        
        self.cache[model_id] = compiled_model
        self.access_times[model_id] = time.time()
    
    def clear(self):
        """Limpiar cache."""
        self.cache.clear()
        self.access_times.clear()


class FastInference:
    """
    Optimizador de inferencia rápida.
    
    Compila modelos y optimiza para inferencia.
    """
    
    def __init__(self, use_jit: bool = True, use_torch_compile: bool = False):
        """
        Inicializar optimizador.
        
        Args:
            use_jit: Usar JIT compilation
            use_torch_compile: Usar torch.compile (PyTorch 2.0+)
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available")
            self.use_jit = False
            self.use_torch_compile = False
            return
        
        self.use_jit = use_jit and hasattr(torch.jit, 'script')
        self.use_torch_compile = use_torch_compile and hasattr(torch, 'compile')
        self.cache = ModelCache()
    
    def optimize_model(
        self,
        model: nn.Module,
        model_id: Optional[str] = None,
        example_input: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """
        Optimizar modelo para inferencia.
        
        Args:
            model: Modelo a optimizar
            model_id: ID del modelo (para cache)
            example_input: Input de ejemplo
            
        Returns:
            Modelo optimizado
        """
        if not TORCH_AVAILABLE:
            return model
        
        # Verificar cache
        if model_id:
            cached = self.cache.get(model_id)
            if cached is not None:
                logger.info(f"Using cached model: {model_id}")
                return cached
        
        model.eval()
        
        # torch.compile (PyTorch 2.0+)
        if self.use_torch_compile:
            try:
                optimized = torch.compile(model, mode="reduce-overhead")
                logger.info("Model optimized with torch.compile")
                if model_id:
                    self.cache.put(model_id, optimized)
                return optimized
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}, falling back to JIT")
        
        # JIT compilation
        if self.use_jit and example_input is not None:
            try:
                with torch.no_grad():
                    traced = jit.trace(model, example_input)
                    traced.eval()
                    logger.info("Model optimized with JIT trace")
                    if model_id:
                        self.cache.put(model_id, traced)
                    return traced
            except Exception as e:
                logger.warning(f"JIT trace failed: {e}")
        
        return model
    
    def batch_inference(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Inferencia por batches optimizada.
        
        Args:
            model: Modelo
            inputs: Inputs [N, ...]
            batch_size: Tamaño de batch
            
        Returns:
            Outputs [N, ...]
        """
        if not TORCH_AVAILABLE:
            return inputs
        
        model.eval()
        outputs = []
        
        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                output = model(batch)
                outputs.append(output)
        
        return torch.cat(outputs, dim=0)


class MemoryOptimizer:
    """
    Optimizador de memoria.
    
    Reduce uso de memoria durante entrenamiento e inferencia.
    """
    
    @staticmethod
    def clear_cache():
        """Limpiar cache de CUDA."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def optimize_for_inference(model: nn.Module):
        """
        Optimizar modelo para inferencia.
        
        Args:
            model: Modelo
        """
        if not TORCH_AVAILABLE:
            return
        
        model.eval()
        
        # Desactivar gradient computation
        for param in model.parameters():
            param.requires_grad = False
    
    @staticmethod
    def set_deterministic(seed: int = 42):
        """
        Configurar determinismo.
        
        Args:
            seed: Semilla
        """
        if not TORCH_AVAILABLE:
            return
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class BatchProcessor:
    """
    Procesador de batches optimizado.
    
    Procesa batches de manera eficiente.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Inicializar procesador.
        
        Args:
            device: Dispositivo
        """
        if not TORCH_AVAILABLE:
            self.device = None
            return
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def process_batch(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        preprocess: Optional[Callable] = None,
        postprocess: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Procesar batch.
        
        Args:
            model: Modelo
            inputs: Inputs
            preprocess: Función de preprocesamiento
            postprocess: Función de postprocesamiento
            
        Returns:
            Outputs procesados
        """
        if not TORCH_AVAILABLE:
            return inputs
        
        # Preprocesamiento
        if preprocess:
            inputs = preprocess(inputs)
        
        # Mover a dispositivo
        inputs = inputs.to(self.device)
        model = model.to(self.device)
        
        # Inferencia
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        
        # Postprocesamiento
        if postprocess:
            outputs = postprocess(outputs)
        
        return outputs
    
    def process_batches(
        self,
        model: nn.Module,
        data_loader: Any,
        max_batches: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Procesar múltiples batches.
        
        Args:
            model: Modelo
            data_loader: DataLoader
            max_batches: Número máximo de batches
            
        Returns:
            Lista de outputs
        """
        if not TORCH_AVAILABLE:
            return []
        
        model.eval()
        outputs = []
        
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if max_batches and i >= max_batches:
                    break
                
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                
                inputs = inputs.to(self.device)
                output = model(inputs)
                outputs.append(output.cpu())
        
        return outputs


class AsyncInference:
    """
    Inferencia asíncrona.
    
    Procesa inferencias en paralelo.
    """
    
    def __init__(self, num_workers: int = 2):
        """
        Inicializar inferencia asíncrona.
        
        Args:
            num_workers: Número de workers
        """
        self.num_workers = num_workers
        self.queue = []
    
    def enqueue(self, model: nn.Module, inputs: torch.Tensor):
        """
        Encolar inferencia.
        
        Args:
            model: Modelo
            inputs: Inputs
        """
        self.queue.append((model, inputs))
    
    def process_queue(self) -> List[torch.Tensor]:
        """
        Procesar cola.
        
        Returns:
            Lista de outputs
        """
        outputs = []
        
        for model, inputs in self.queue:
            model.eval()
            with torch.no_grad():
                output = model(inputs)
                outputs.append(output)
        
        self.queue.clear()
        return outputs


# Instancias globales
_fast_inference = None
_memory_optimizer = None
_batch_processor = None


def get_fast_inference() -> FastInference:
    """Obtener instancia global de FastInference."""
    global _fast_inference
    if _fast_inference is None:
        _fast_inference = FastInference()
    return _fast_inference


def get_memory_optimizer() -> MemoryOptimizer:
    """Obtener instancia global de MemoryOptimizer."""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer


def get_batch_processor(device: Optional[str] = None) -> BatchProcessor:
    """Obtener instancia global de BatchProcessor."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor(device)
    return _batch_processor

