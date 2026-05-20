"""
Aggressive JIT Compilation
==========================

Compilación JIT más agresiva para máximo rendimiento.
"""

import logging
import torch
from typing import Any, Callable, Optional
from functools import wraps

logger = logging.getLogger(__name__)


class AggressiveJIT:
    """Compilación JIT agresiva."""
    
    @staticmethod
    def compile_function(func: Callable, mode: str = "reduce-overhead") -> Callable:
        """
        Compilar función con torch.jit.script.
        
        Args:
            func: Función a compilar
            mode: Modo de compilación
        
        Returns:
            Función compilada
        """
        try:
            if hasattr(torch.jit, 'script'):
                compiled = torch.jit.script(func)
                logger.info(f"Función compilada con JIT: {func.__name__}")
                return compiled
            return func
        except Exception as e:
            logger.warning(f"No se pudo compilar {func.__name__}: {str(e)}")
            return func
    
    @staticmethod
    def compile_module(module: torch.nn.Module) -> torch.nn.Module:
        """
        Compilar módulo completo.
        
        Args:
            module: Módulo a compilar
        
        Returns:
            Módulo compilado
        """
        try:
            if hasattr(torch.jit, 'script'):
                compiled = torch.jit.script(module)
                logger.info("Módulo compilado con JIT")
                return compiled
            return module
        except Exception as e:
            logger.warning(f"No se pudo compilar módulo: {str(e)}")
            return module
    
    @staticmethod
    def optimize_with_torch_compile(
        func: Callable,
        mode: str = "reduce-overhead",
        fullgraph: bool = True
    ) -> Callable:
        """
        Optimizar con torch.compile (PyTorch 2.0+).
        
        Args:
            func: Función a optimizar
            mode: Modo de compilación
            fullgraph: Compilar grafo completo
        
        Returns:
            Función optimizada
        """
        try:
            if hasattr(torch, 'compile'):
                compiled = torch.compile(
                    func,
                    mode=mode,
                    fullgraph=fullgraph
                )
                logger.info(f"Función compilada con torch.compile: {func.__name__}")
                return compiled
            return func
        except Exception as e:
            logger.warning(f"No se pudo compilar con torch.compile: {str(e)}")
            return func


def jit_compile(mode: str = "reduce-overhead"):
    """
    Decorador para compilar funciones con JIT.
    
    Args:
        mode: Modo de compilación
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            compiled_func = AggressiveJIT.optimize_with_torch_compile(func, mode=mode)
            return compiled_func(*args, **kwargs)
        return wrapper
    return decorator




