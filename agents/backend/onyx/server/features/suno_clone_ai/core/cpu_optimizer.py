"""
CPU Optimizer
Optimizaciones específicas de CPU
"""

import logging
import os
import sys
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CPUOptimizer:
    """Optimizador de CPU"""
    
    def __init__(self):
        self._optimized = False
    
    def optimize(self):
        """Aplica optimizaciones de CPU"""
        if self._optimized:
            return
        
        logger.info("Applying CPU optimizations...")
        
        # 1. Configurar número de threads para numpy/scipy
        self._configure_numpy_threads()
        
        # 2. Configurar affinity (si es posible)
        self._configure_cpu_affinity()
        
        # 3. Habilitar optimizaciones de Python
        self._enable_python_optimizations()
        
        self._optimized = True
        logger.info("CPU optimizations applied")
    
    def _configure_numpy_threads(self):
        """Configura threads de numpy"""
        try:
            import numpy as np
            # Limitar threads de numpy para evitar sobrecarga
            num_threads = min(4, os.cpu_count() or 4)
            os.environ['OMP_NUM_THREADS'] = str(num_threads)
            os.environ['MKL_NUM_THREADS'] = str(num_threads)
            os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
            logger.debug(f"Numpy threads configured: {num_threads}")
        except ImportError:
            pass
    
    def _configure_cpu_affinity(self):
        """Configura CPU affinity"""
        try:
            import psutil
            process = psutil.Process()
            # No cambiar affinity por defecto, pero registrar
            logger.debug(f"CPU affinity: {process.cpu_affinity()}")
        except (ImportError, AttributeError):
            pass
    
    def _enable_python_optimizations(self):
        """Habilita optimizaciones de Python"""
        # Habilitar optimizaciones de bytecode
        sys.dont_write_bytecode = False
        
        # Configurar hash randomization (para seguridad, pero puede afectar performance)
        # No cambiar por defecto
    
    def get_cpu_info(self) -> Dict[str, Any]:
        """Obtiene información de CPU"""
        try:
            import psutil
            return {
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }
        except ImportError:
            return {"error": "psutil not available"}


# Instancia global
_cpu_optimizer: Optional[CPUOptimizer] = None


def get_cpu_optimizer() -> CPUOptimizer:
    """Obtiene el optimizador de CPU"""
    global _cpu_optimizer
    if _cpu_optimizer is None:
        _cpu_optimizer = CPUOptimizer()
    return _cpu_optimizer

