"""
Inference Optimization
======================

Optimizaciones para inferencia rápida.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class InferenceOptimizer:
    """
    Optimizador para inferencia rápida.
    """
    
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        """
        Inicializar optimizador.
        
        Args:
            model: Modelo a optimizar
            device: Dispositivo
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def optimize(self, 
                 use_jit: bool = True,
                 use_torch_compile: bool = False,
                 use_fusion: bool = True,
                 use_cudnn: bool = True) -> nn.Module:
        """
        Aplicar optimizaciones.
        
        Args:
            use_jit: Usar TorchScript JIT
            use_torch_compile: Usar torch.compile (PyTorch 2.0+)
            use_fusion: Usar kernel fusion
            use_cudnn: Optimizar cuDNN
            
        Returns:
            Modelo optimizado
        """
        # Optimizaciones de cuDNN
        if use_cudnn and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logger.info("cuDNN optimizado")
        
        # Kernel fusion
        if use_fusion and torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("TF32 habilitado para kernel fusion")
        
        # TorchScript JIT
        if use_jit:
            try:
                self.model = torch.jit.script(self.model)
                logger.info("Modelo compilado con TorchScript JIT")
            except Exception as e:
                logger.warning(f"Error en JIT compilation: {e}")
        
        # torch.compile (PyTorch 2.0+)
        if use_torch_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Modelo compilado con torch.compile")
            except Exception as e:
                logger.warning(f"Error en torch.compile: {e}")
        
        return self.model
    
    def optimize_for_batch_inference(self, batch_size: int = 32):
        """
        Optimizar para inferencia por batches.
        
        Args:
            batch_size: Tamaño de batch esperado
        """
        # Pre-compilar con tamaño de batch
        dummy_input = torch.randn(batch_size, self.model.config.input_dim).to(self.device)
        
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        logger.info(f"Modelo optimizado para batch size: {batch_size}")


def compile_model(model: nn.Module, 
                 method: str = "torchscript",
                 optimize: bool = True) -> nn.Module:
    """
    Compilar modelo para inferencia rápida.
    
    Args:
        model: Modelo a compilar
        method: Método ("torchscript", "torch_compile", "both")
        optimize: Aplicar optimizaciones adicionales
        
    Returns:
        Modelo compilado
    """
    model.eval()
    
    if method == "torchscript":
        try:
            model = torch.jit.script(model)
            logger.info("Modelo compilado con TorchScript")
        except Exception as e:
            logger.warning(f"Error en TorchScript: {e}")
    
    elif method == "torch_compile" and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Modelo compilado con torch.compile")
        except Exception as e:
            logger.warning(f"Error en torch.compile: {e}")
    
    elif method == "both":
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Modelo compilado con torch.compile")
            except Exception as e:
                logger.warning(f"Error en torch.compile: {e}")
        
        try:
            model = torch.jit.script(model)
            logger.info("Modelo compilado con TorchScript")
        except Exception as e:
            logger.warning(f"Error en TorchScript: {e}")
    
    if optimize:
        optimizer = InferenceOptimizer(model)
        model = optimizer.optimize(use_jit=False, use_torch_compile=False)
    
    return model


def optimize_for_inference(model: nn.Module,
                          input_shape: tuple = (1, 20),
                          device: str = "cuda") -> nn.Module:
    """
    Optimizar modelo para inferencia.
    
    Args:
        model: Modelo
        input_shape: Forma de entrada esperada
        device: Dispositivo
        
    Returns:
        Modelo optimizado
    """
    model.eval()
    model.to(device)
    
    # Warmup
    dummy_input = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Compilar
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except:
            pass
    
    logger.info("Modelo optimizado para inferencia")
    return model

