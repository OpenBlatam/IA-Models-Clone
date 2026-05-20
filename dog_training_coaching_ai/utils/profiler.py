"""
Profiling Utilities
===================
Utilidades para profiling de código.
"""

import cProfile
import pstats
import io
from typing import Callable, Optional, Dict, Any
from contextlib import contextmanager
import time

from .logger import get_logger

logger = get_logger(__name__)


class Profiler:
    """Clase para profiling de código."""
    
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.stats: Optional[pstats.Stats] = None
    
    @contextmanager
    def profile(self):
        """Context manager para profiling."""
        self.profiler.enable()
        try:
            yield
        finally:
            self.profiler.disable()
            stream = io.StringIO()
            self.stats = pstats.Stats(self.profiler, stream=stream)
    
    def get_stats(self, sort_by: str = "cumulative", limit: int = 20) -> Dict[str, Any]:
        """
        Obtener estadísticas de profiling.
        
        Args:
            sort_by: Campo por el que ordenar
            limit: Número de líneas a retornar
            
        Returns:
            Estadísticas
        """
        if not self.stats:
            return {"error": "No profiling data available"}
        
        stream = io.StringIO()
        self.stats.sort_stats(sort_by)
        self.stats.print_stats(limit)
        stats_output = stream.getvalue()
        
        return {
            "stats": stats_output,
            "total_calls": self.stats.total_calls,
            "primitive_calls": self.stats.primitive_calls,
            "total_time": self.stats.total_tt
        }
    
    def get_top_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener top funciones por tiempo.
        
        Args:
            limit: Número de funciones
            
        Returns:
            Lista de funciones
        """
        if not self.stats:
            return []
        
        top_functions = []
        self.stats.sort_stats("cumulative")
        
        for func, (cc, nc, tt, ct, callers) in self.stats.stats.items():
            top_functions.append({
                "function": f"{func[0]}:{func[1]}({func[2]})",
                "cumulative_time": ct,
                "total_time": tt,
                "calls": cc,
                "primitive_calls": nc
            })
            
            if len(top_functions) >= limit:
                break
        
        return top_functions


def profile_function(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Profilear función.
    
    Args:
        func: Función a profilear
        *args: Argumentos posicionales
        **kwargs: Argumentos nombrados
        
    Returns:
        Estadísticas de profiling
    """
    profiler = Profiler()
    
    with profiler.profile():
        result = func(*args, **kwargs)
    
    return {
        "result": result,
        "stats": profiler.get_stats(),
        "top_functions": profiler.get_top_functions()
    }



