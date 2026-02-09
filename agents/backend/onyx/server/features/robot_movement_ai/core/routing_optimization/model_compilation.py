"""
Model Compilation
=================

Compilación de modelos para máxima velocidad.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelCompiler:
    """
    Compilador de modelos para optimización.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar compilador.
        
        Args:
            model: Modelo a compilar
        """
        self.model = model
        self.compiled_model = None
    
    def compile_torchscript(self, 
                          example_input: Optional[torch.Tensor] = None,
                          optimize: bool = True) -> nn.Module:
        """
        Compilar con TorchScript.
        
        Args:
            example_input: Input de ejemplo
            optimize: Aplicar optimizaciones
            
        Returns:
            Modelo compilado
        """
        self.model.eval()
        
        try:
            if example_input is None:
                # Crear input de ejemplo
                if hasattr(self.model, 'config'):
                    input_dim = self.model.config.input_dim
                else:
                    input_dim = 20
                example_input = torch.randn(1, input_dim)
            
            # Tracing
            traced_model = torch.jit.trace(self.model, example_input)
            
            if optimize:
                traced_model = torch.jit.optimize_for_inference(traced_model)
            
            self.compiled_model = traced_model
            logger.info("Modelo compilado con TorchScript")
            return traced_model
        except Exception as e:
            logger.warning(f"Error en TorchScript: {e}")
            return self.model
    
    def compile_torch_compile(self,
                            mode: str = "reduce-overhead",
                            fullgraph: bool = False) -> nn.Module:
        """
        Compilar con torch.compile (PyTorch 2.0+).
        
        Args:
            mode: Modo de compilación ("default", "reduce-overhead", "max-autotune")
            fullgraph: Compilar grafo completo
            
        Returns:
            Modelo compilado
        """
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile no disponible (requiere PyTorch 2.0+)")
            return self.model
        
        try:
            self.compiled_model = torch.compile(
                self.model,
                mode=mode,
                fullgraph=fullgraph
            )
            logger.info(f"Modelo compilado con torch.compile (mode: {mode})")
            return self.compiled_model
        except Exception as e:
            logger.warning(f"Error en torch.compile: {e}")
            return self.model
    
    def compile_both(self, example_input: Optional[torch.Tensor] = None) -> nn.Module:
        """
        Compilar con ambos métodos.
        
        Args:
            example_input: Input de ejemplo
            
        Returns:
            Modelo compilado
        """
        # Primero torch.compile
        if hasattr(torch, 'compile'):
            self.model = self.compile_torch_compile()
        
        # Luego TorchScript
        self.model = self.compile_torchscript(example_input)
        
        return self.model


def compile_with_torchscript(model: nn.Module,
                            example_input: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Compilar modelo con TorchScript.
    
    Args:
        model: Modelo
        example_input: Input de ejemplo
        
    Returns:
        Modelo compilado
    """
    compiler = ModelCompiler(model)
    return compiler.compile_torchscript(example_input)


def compile_with_torch_compile(model: nn.Module,
                              mode: str = "reduce-overhead") -> nn.Module:
    """
    Compilar modelo con torch.compile.
    
    Args:
        model: Modelo
        mode: Modo de compilación
        
    Returns:
        Modelo compilado
    """
    compiler = ModelCompiler(model)
    return compiler.compile_torch_compile(mode=mode)


def benchmark_model(model: nn.Module,
                   input_shape: tuple = (1, 20),
                   device: str = "cuda",
                   num_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark de modelo.
    
    Args:
        model: Modelo
        input_shape: Forma de entrada
        device: Dispositivo
        num_runs: Número de runs
        
    Returns:
        Métricas de rendimiento
    """
    model.eval()
    model.to(device)
    
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Sincronizar
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    import time
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_runs * 1000  # ms
    
    return {
        "avg_inference_time_ms": avg_time,
        "throughput_samples_per_sec": 1000 / avg_time if avg_time > 0 else 0,
        "total_time_sec": elapsed_time
    }

