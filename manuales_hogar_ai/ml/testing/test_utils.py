"""
Test Utils
==========

Utilidades para testing.
"""

import logging
import torch
import numpy as np
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)


class TestUtils:
    """Utilidades para testing."""
    
    @staticmethod
    def create_dummy_tensor(
        shape: tuple,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Crear tensor dummy para testing.
        
        Args:
            shape: Forma del tensor
            dtype: Tipo de datos
            device: Dispositivo
        
        Returns:
            Tensor dummy
        """
        return torch.randn(shape, dtype=dtype, device=device)
    
    @staticmethod
    def create_dummy_batch(
        batch_size: int = 8,
        seq_length: int = 512,
        vocab_size: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """
        Crear batch dummy.
        
        Args:
            batch_size: Tamaño de batch
            seq_length: Longitud de secuencia
            vocab_size: Tamaño de vocabulario
        
        Returns:
            Batch dummy
        """
        return {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length)),
            "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.long)
        }
    
    @staticmethod
    def assert_tensor_equal(
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-8
    ) -> bool:
        """
        Verificar si tensores son iguales.
        
        Args:
            tensor1: Primer tensor
            tensor2: Segundo tensor
            rtol: Tolerancia relativa
            atol: Tolerancia absoluta
        
        Returns:
            True si son iguales
        """
        return torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
    
    @staticmethod
    def measure_inference_time(
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        num_runs: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """
        Medir tiempo de inferencia.
        
        Args:
            model: Modelo
            input_tensor: Tensor de entrada
            num_runs: Número de ejecuciones
            warmup: Ejecuciones de warmup
        
        Returns:
            Estadísticas de tiempo
        """
        model.eval()
        times = []
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(input_tensor)
        
        # Medir
        import time
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(input_tensor)
                times.append(time.time() - start)
        
        return {
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
            "p50": np.percentile(times, 50),
            "p95": np.percentile(times, 95),
            "p99": np.percentile(times, 99)
        }




