"""
Speed Optimizations
===================

Optimizaciones específicas para máxima velocidad.
"""

import logging
from typing import Dict, Any, Optional, List
import threading
from queue import Queue

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


class ModelWarmup:
    """
    Warmup de modelos.
    
    Pre-calienta modelos para inferencia más rápida.
    """
    
    @staticmethod
    def warmup(
        model: nn.Module,
        input_shape: tuple,
        num_iterations: int = 10,
        device: Optional[str] = None
    ):
        """
        Pre-calentar modelo.
        
        Args:
            model: Modelo
            input_shape: Forma de entrada
            num_iterations: Número de iteraciones
            device: Dispositivo
        """
        if not TORCH_AVAILABLE:
            return
        
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        dummy_input = torch.randn(input_shape).to(device)
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        logger.info(f"Model warmed up with {num_iterations} iterations")


class ParallelInference:
    """
    Inferencia paralela.
    
    Procesa múltiples modelos en paralelo.
    """
    
    def __init__(self, num_threads: int = 4):
        """
        Inicializar inferencia paralela.
        
        Args:
            num_threads: Número de threads
        """
        self.num_threads = num_threads
        self.results = Queue()
    
    def process_parallel(
        self,
        models: List[nn.Module],
        inputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Procesar modelos en paralelo.
        
        Args:
            models: Lista de modelos
            inputs: Lista de inputs
            
        Returns:
            Lista de outputs
        """
        if not TORCH_AVAILABLE:
            return []
        
        def worker(model, input_tensor, idx):
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                self.results.put((idx, output))
        
        threads = []
        for i, (model, inp) in enumerate(zip(models, inputs)):
            thread = threading.Thread(target=worker, args=(model, inp, i))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
        
        # Recopilar resultados
        outputs = [None] * len(models)
        while not self.results.empty():
            idx, output = self.results.get()
            outputs[idx] = output
        
        return outputs


class TensorOptimizer:
    """
    Optimizador de tensores.
    
    Optimiza operaciones con tensores.
    """
    
    @staticmethod
    def pin_memory(tensor: torch.Tensor) -> torch.Tensor:
        """
        Fijar memoria para transferencia más rápida.
        
        Args:
            tensor: Tensor
            
        Returns:
            Tensor con memoria fijada
        """
        if not TORCH_AVAILABLE:
            return tensor
        
        if tensor.is_cuda:
            return tensor
        return tensor.pin_memory()
    
    @staticmethod
    def non_blocking_transfer(tensor: torch.Tensor, device: str) -> torch.Tensor:
        """
        Transferencia no bloqueante.
        
        Args:
            tensor: Tensor
            device: Dispositivo
            
        Returns:
            Tensor transferido
        """
        if not TORCH_AVAILABLE:
            return tensor
        
        return tensor.to(device, non_blocking=True)
    
    @staticmethod
    def contiguous(tensor: torch.Tensor) -> torch.Tensor:
        """
        Asegurar que tensor sea contiguo.
        
        Args:
            tensor: Tensor
            
        Returns:
            Tensor contiguo
        """
        if not TORCH_AVAILABLE:
            return tensor
        
        return tensor.contiguous()


class DataLoaderOptimizer:
    """
    Optimizador de DataLoader.
    
    Configura DataLoader para máximo rendimiento.
    """
    
    @staticmethod
    def get_optimized_config(
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True
    ) -> Dict[str, Any]:
        """
        Obtener configuración optimizada.
        
        Args:
            num_workers: Número de workers
            pin_memory: Fijar memoria
            prefetch_factor: Factor de prefetch
            persistent_workers: Workers persistentes
            
        Returns:
            Configuración
        """
        return {
            "num_workers": num_workers,
            "pin_memory": pin_memory and TORCH_AVAILABLE and torch.cuda.is_available(),
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
            "drop_last": False
        }


class InferencePipeline:
    """
    Pipeline de inferencia optimizado.
    
    Pipeline completo para inferencia rápida.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        use_fp16: bool = False,
        batch_size: int = 32
    ):
        """
        Inicializar pipeline.
        
        Args:
            model: Modelo
            device: Dispositivo
            use_fp16: Usar FP16
            batch_size: Tamaño de batch
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.batch_size = batch_size
        
        if self.use_fp16:
            self.model = self.model.half()
        
        # Warmup
        ModelWarmup.warmup(self.model, (1, 3, 224, 224), device=self.device)
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Predecir.
        
        Args:
            inputs: Inputs
            
        Returns:
            Outputs
        """
        inputs = inputs.to(self.device)
        
        if self.use_fp16:
            inputs = inputs.half()
        
        with torch.no_grad():
            outputs = self.model(inputs)
        
        if self.use_fp16:
            outputs = outputs.float()
        
        return outputs
    
    def predict_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Predecir por batches.
        
        Args:
            inputs: Inputs [N, ...]
            
        Returns:
            Outputs [N, ...]
        """
        outputs = []
        
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            output = self.predict(batch)
            outputs.append(output)
        
        return torch.cat(outputs, dim=0)


# Instancias globales
_tensor_optimizer = None
_dataloader_optimizer = None


def get_tensor_optimizer() -> TensorOptimizer:
    """Obtener instancia global."""
    global _tensor_optimizer
    if _tensor_optimizer is None:
        _tensor_optimizer = TensorOptimizer()
    return _tensor_optimizer


def get_dataloader_optimizer() -> DataLoaderOptimizer:
    """Obtener instancia global."""
    global _dataloader_optimizer
    if _dataloader_optimizer is None:
        _dataloader_optimizer = DataLoaderOptimizer()
    return _dataloader_optimizer

