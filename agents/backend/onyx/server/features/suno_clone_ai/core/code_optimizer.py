"""
Code Optimizer
Optimizaciones de código en tiempo de ejecución
"""

import logging
import sys
from functools import lru_cache, wraps
from typing import Any, Callable
import inspect

logger = logging.getLogger(__name__)


class CodeOptimizer:
    """Optimizador de código en runtime"""
    
    def __init__(self):
        self._optimized_functions = set()
        self._jit_enabled = False
    
    def enable_jit(self):
        """Habilita JIT compilation con numba"""
        try:
            from numba import jit
            self._jit = jit
            self._jit_enabled = True
            logger.info("JIT compilation enabled")
        except ImportError:
            logger.warning("numba not available, JIT disabled")
            self._jit_enabled = False
    
    def optimize_function(self, func: Callable, use_jit: bool = False) -> Callable:
        """
        Optimiza una función
        
        Args:
            func: Función a optimizar
            use_jit: Usar JIT compilation si está disponible
        """
        if func in self._optimized_functions:
            return func
        
        # Aplicar cache si es apropiado
        if not hasattr(func, '__wrapped__'):
            # Verificar si la función es cacheable
            sig = inspect.signature(func)
            if len(sig.parameters) <= 5:  # Cache solo si pocos parámetros
                func = lru_cache(maxsize=128)(func)
        
        # Aplicar JIT si está habilitado y es numérico
        if use_jit and self._jit_enabled:
            try:
                # Solo para funciones numéricas
                if self._is_numeric_function(func):
                    func = self._jit(nopython=True, cache=True)(func)
            except Exception as e:
                logger.warning(f"Could not apply JIT to {func.__name__}: {e}")
        
        self._optimized_functions.add(func)
        return func
    
    def _is_numeric_function(self, func: Callable) -> bool:
        """Verifica si una función es principalmente numérica"""
        # Heurística simple: verifica imports de numpy/scipy
        source = inspect.getsource(func)
        numeric_keywords = ['numpy', 'np.', 'scipy', 'math.', 'np.array']
        return any(keyword in source for keyword in numeric_keywords)
    
    def optimize_imports(self):
        """Optimiza imports comunes"""
        # Pre-importar y cachear módulos críticos
        critical_modules = [
            'json', 'hashlib', 'uuid', 'datetime',
            'asyncio', 'functools', 'collections'
        ]
        
        for module_name in critical_modules:
            try:
                module = __import__(module_name)
                sys.modules[f'_opt_{module_name}'] = module
            except ImportError:
                pass


# Instancia global
_code_optimizer: 'CodeOptimizer' = None


def get_code_optimizer() -> CodeOptimizer:
    """Obtiene el optimizador de código"""
    global _code_optimizer
    if _code_optimizer is None:
        _code_optimizer = CodeOptimizer()
    return _code_optimizer


def optimized(func: Callable = None, use_jit: bool = False):
    """Decorator para optimizar funciones"""
    def decorator(f: Callable) -> Callable:
        optimizer = get_code_optimizer()
        return optimizer.optimize_function(f, use_jit=use_jit)
    
    if func is None:
        return decorator
    else:
        return decorator(func)















