"""
Performance Optimizer
Optimizaciones agresivas de rendimiento
"""

import logging
import sys
from functools import lru_cache
from typing import Any, Dict

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Optimizador de rendimiento global"""
    
    def __init__(self):
        self._optimized = False
    
    def optimize_all(self):
        """Aplica todas las optimizaciones"""
        if self._optimized:
            return
        
        logger.info("Applying performance optimizations...")
        
        # 1. Optimizar sys.path
        self._optimize_sys_path()
        
        # 2. Optimizar imports
        self._optimize_imports()
        
        # 3. Configurar Python optimizations
        self._configure_python_opts()
        
        # 4. Pre-compilar regex
        self._precompile_regex()
        
        self._optimized = True
        logger.info("Performance optimizations applied")
    
    def _optimize_sys_path(self):
        """Optimiza sys.path para imports más rápidos"""
        # Mantener sys.path limpio y ordenado
        if hasattr(sys, 'path'):
            # Remover paths duplicados
            seen = set()
            sys.path = [p for p in sys.path if p not in seen and not seen.add(p)]
    
    def _optimize_imports(self):
        """Optimiza imports comunes"""
        # Pre-importar módulos críticos
        try:
            import json
            import asyncio
            import hashlib
            # Cachear imports
            sys.modules['_fast_json'] = json
            sys.modules['_fast_asyncio'] = asyncio
            sys.modules['_fast_hashlib'] = hashlib
        except Exception as e:
            logger.warning(f"Could not optimize imports: {e}")
    
    def _configure_python_opts(self):
        """Configura opciones de Python para mejor rendimiento"""
        # Habilitar optimizaciones de bytecode
        sys.dont_write_bytecode = False  # Escribir .pyc files
    
    def _precompile_regex(self):
        """Pre-compila expresiones regulares comunes"""
        try:
            import re
            # Compilar regex comunes
            self._email_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
            self._uuid_regex = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
        except Exception as e:
            logger.warning(f"Could not precompile regex: {e}")


# Instancia global
_perf_optimizer = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Obtiene el optimizador de rendimiento"""
    global _perf_optimizer
    if _perf_optimizer is None:
        _perf_optimizer = PerformanceOptimizer()
    return _perf_optimizer















