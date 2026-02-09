"""
Optimization Profiler
=====================

Profiler avanzado para optimizaciones.
"""

import time
import cProfile
import pstats
import io
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Resultado de profiling."""
    function_name: str
    total_time: float
    cumulative_time: float
    call_count: int
    per_call_time: float
    file_path: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class OptimizationProfile:
    """Perfil de optimización."""
    profile_id: str
    function_name: str
    total_time: float
    call_count: int
    top_functions: List[ProfileResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class OptimizationProfiler:
    """
    Profiler de optimizaciones.
    
    Profila funciones para identificar cuellos de botella.
    """
    
    def __init__(self):
        """Inicializar profiler."""
        self.profiles: List[OptimizationProfile] = []
        self.max_profiles = 100
    
    def profile_function(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> OptimizationProfile:
        """
        Profilar función.
        
        Args:
            func: Función a profilear
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Perfil de optimización
        """
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
            end_time = time.perf_counter()
        
        total_time = end_time - start_time
        
        # Analizar resultados
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20
        
        # Parsear resultados
        output = s.getvalue()
        top_functions = self._parse_profile_output(output)
        
        profile = OptimizationProfile(
            profile_id=f"profile_{len(self.profiles)}",
            function_name=func.__name__,
            total_time=total_time,
            call_count=1,
            top_functions=top_functions
        )
        
        self.profiles.append(profile)
        
        # Limitar tamaño
        if len(self.profiles) > self.max_profiles:
            self.profiles = self.profiles[-self.max_profiles:]
        
        logger.info(f"Profiled {func.__name__}: {total_time:.4f}s")
        
        return profile
    
    def _parse_profile_output(self, output: str) -> List[ProfileResult]:
        """Parsear salida de profiling."""
        results = []
        lines = output.split('\n')
        
        for line in lines[5:25]:  # Saltar header, tomar top 20
            if not line.strip() or 'ncalls' in line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                try:
                    call_count = int(parts[0]) if parts[0] != 'ncalls' else 1
                    total_time = float(parts[1])
                    cumulative_time = float(parts[2])
                    
                    # Extraer nombre de función
                    func_name = ' '.join(parts[4:]) if len(parts) > 4 else parts[3]
                    
                    results.append(ProfileResult(
                        function_name=func_name,
                        total_time=total_time,
                        cumulative_time=cumulative_time,
                        call_count=call_count,
                        per_call_time=total_time / call_count if call_count > 0 else 0.0
                    ))
                except (ValueError, IndexError):
                    continue
        
        return results
    
    def get_bottlenecks(self, threshold: float = 0.1) -> List[ProfileResult]:
        """
        Identificar cuellos de botella.
        
        Args:
            threshold: Umbral de tiempo (segundos)
            
        Returns:
            Lista de funciones que son cuellos de botella
        """
        all_functions = {}
        
        for profile in self.profiles:
            for func_result in profile.top_functions:
                if func_result.function_name not in all_functions:
                    all_functions[func_result.function_name] = ProfileResult(
                        function_name=func_result.function_name,
                        total_time=0.0,
                        cumulative_time=0.0,
                        call_count=0,
                        per_call_time=0.0
                    )
                
                all_functions[func_result.function_name].total_time += func_result.total_time
                all_functions[func_result.function_name].call_count += func_result.call_count
        
        # Calcular promedio por llamada
        for func_result in all_functions.values():
            if func_result.call_count > 0:
                func_result.per_call_time = func_result.total_time / func_result.call_count
        
        # Filtrar por threshold
        bottlenecks = [
            func_result for func_result in all_functions.values()
            if func_result.per_call_time > threshold or func_result.total_time > threshold
        ]
        
        # Ordenar por tiempo total
        bottlenecks.sort(key=lambda x: x.total_time, reverse=True)
        
        return bottlenecks
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Obtener resumen de profiling."""
        if not self.profiles:
            return {
                "total_profiles": 0,
                "total_time": 0.0,
                "average_time": 0.0
            }
        
        total_time = sum(p.total_time for p in self.profiles)
        avg_time = total_time / len(self.profiles)
        
        return {
            "total_profiles": len(self.profiles),
            "total_time": total_time,
            "average_time": avg_time,
            "bottlenecks": [
                {
                    "function_name": b.function_name,
                    "total_time": b.total_time,
                    "per_call_time": b.per_call_time,
                    "call_count": b.call_count
                }
                for b in self.get_bottlenecks()
            ]
        }


# Instancia global
_optimization_profiler: Optional[OptimizationProfiler] = None


def get_optimization_profiler() -> OptimizationProfiler:
    """Obtener instancia global del profiler de optimizaciones."""
    global _optimization_profiler
    if _optimization_profiler is None:
        _optimization_profiler = OptimizationProfiler()
    return _optimization_profiler






