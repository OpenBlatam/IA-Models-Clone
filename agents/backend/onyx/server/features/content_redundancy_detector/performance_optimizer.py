"""
Performance Optimizer for Ultra-High Performance Systems
Optimizador de Performance para sistemas de ultra-alto rendimiento
"""

import asyncio
import logging
import time
import json
import threading
import psutil
import gc
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter
from collections import defaultdict, deque
import numpy as np
import concurrent.futures
from functools import lru_cache, wraps
import weakref
import tracemalloc
import linecache
import sys

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Niveles de optimización"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    ULTRA = "ultra"
    EXTREME = "extreme"


class OptimizationType(Enum):
    """Tipos de optimización"""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    ALGORITHM = "algorithm"
    CONCURRENCY = "concurrency"
    I_O = "i_o"


class PerformanceMetric(Enum):
    """Métricas de performance"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"
    CONCURRENT_REQUESTS = "concurrent_requests"
    QUEUE_SIZE = "queue_size"


@dataclass
class PerformanceProfile:
    """Perfil de performance"""
    id: str
    name: str
    optimization_level: OptimizationLevel
    target_metrics: Dict[PerformanceMetric, float]
    current_metrics: Dict[PerformanceMetric, float]
    optimization_rules: List[Dict[str, Any]]
    created_at: float
    last_updated: float
    is_active: bool


@dataclass
class OptimizationRule:
    """Regla de optimización"""
    id: str
    name: str
    condition: str
    action: str
    parameters: Dict[str, Any]
    priority: int
    is_enabled: bool
    last_triggered: Optional[float]
    trigger_count: int


@dataclass
class PerformanceAlert:
    """Alerta de performance"""
    id: str
    metric: PerformanceMetric
    threshold: float
    current_value: float
    severity: str
    message: str
    timestamp: float
    is_resolved: bool


class MemoryOptimizer:
    """Optimizador de memoria"""
    
    def __init__(self):
        self.memory_pools: Dict[str, List[Any]] = defaultdict(list)
        self.object_cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self.memory_stats: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._gc_threshold = 0.8  # 80% memory usage
        self._pool_size_limit = 1000
    
    async def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimizar uso de memoria"""
        try:
            # Obtener estadísticas de memoria
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent / 100.0
            
            optimizations = []
            
            # Limpiar pools de memoria si están llenos
            if memory_usage > self._gc_threshold:
                await self._cleanup_memory_pools()
                optimizations.append("memory_pools_cleaned")
            
            # Ejecutar garbage collection si es necesario
            if memory_usage > 0.7:
                collected = gc.collect()
                optimizations.append(f"garbage_collection: {collected} objects")
            
            # Limpiar cache de objetos débiles
            if len(self.object_cache) > 10000:
                self.object_cache.clear()
                optimizations.append("weak_cache_cleared")
            
            # Optimizar pools de memoria
            await self._optimize_memory_pools()
            optimizations.append("memory_pools_optimized")
            
            return {
                "memory_usage_before": memory_usage,
                "memory_usage_after": psutil.virtual_memory().percent / 100.0,
                "optimizations_applied": optimizations,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing memory usage: {e}")
            return {"error": str(e)}
    
    async def _cleanup_memory_pools(self):
        """Limpiar pools de memoria"""
        async with self._lock:
            for pool_name, pool in self.memory_pools.items():
                if len(pool) > self._pool_size_limit:
                    # Mantener solo los elementos más recientes
                    self.memory_pools[pool_name] = pool[-self._pool_size_limit//2:]
    
    async def _optimize_memory_pools(self):
        """Optimizar pools de memoria"""
        async with self._lock:
            for pool_name, pool in self.memory_pools.items():
                # Remover elementos duplicados
                if isinstance(pool, list):
                    seen = set()
                    self.memory_pools[pool_name] = [
                        item for item in pool 
                        if not (item in seen or seen.add(item))
                    ]
    
    def get_memory_object(self, key: str, factory: Callable) -> Any:
        """Obtener objeto de memoria con cache"""
        if key in self.object_cache:
            return self.object_cache[key]
        
        obj = factory()
        self.object_cache[key] = obj
        return obj
    
    def add_to_memory_pool(self, pool_name: str, obj: Any):
        """Agregar objeto a pool de memoria"""
        async with self._lock:
            self.memory_pools[pool_name].append(obj)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de memoria"""
        memory_info = psutil.virtual_memory()
        return {
            "total_memory": memory_info.total,
            "available_memory": memory_info.available,
            "used_memory": memory_info.used,
            "memory_percent": memory_info.percent,
            "memory_pools_count": len(self.memory_pools),
            "total_pool_objects": sum(len(pool) for pool in self.memory_pools.values()),
            "weak_cache_size": len(self.object_cache)
        }


class CPUOptimizer:
    """Optimizador de CPU"""
    
    def __init__(self):
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=psutil.cpu_count() * 2)
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=psutil.cpu_count())
        self.cpu_stats: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._cpu_threshold = 0.8  # 80% CPU usage
    
    async def optimize_cpu_usage(self) -> Dict[str, Any]:
        """Optimizar uso de CPU"""
        try:
            # Obtener estadísticas de CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            optimizations = []
            
            # Ajustar tamaño de thread pool basado en carga de CPU
            if cpu_percent > self._cpu_threshold:
                # Reducir workers si CPU está sobrecargado
                new_workers = max(1, int(cpu_count * 0.5))
                self.thread_pool._max_workers = new_workers
                optimizations.append(f"thread_pool_reduced_to_{new_workers}")
            elif cpu_percent < 0.3:
                # Aumentar workers si CPU está subutilizado
                new_workers = min(cpu_count * 3, 32)
                self.thread_pool._max_workers = new_workers
                optimizations.append(f"thread_pool_increased_to_{new_workers}")
            
            # Optimizar procesos en background
            await self._optimize_background_processes()
            optimizations.append("background_processes_optimized")
            
            return {
                "cpu_usage_before": cpu_percent,
                "cpu_usage_after": psutil.cpu_percent(interval=0.1),
                "cpu_count": cpu_count,
                "thread_pool_workers": self.thread_pool._max_workers,
                "optimizations_applied": optimizations,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing CPU usage: {e}")
            return {"error": str(e)}
    
    async def _optimize_background_processes(self):
        """Optimizar procesos en background"""
        # Simular optimización de procesos
        await asyncio.sleep(0.01)
    
    def execute_cpu_intensive_task(self, func: Callable, *args, **kwargs):
        """Ejecutar tarea intensiva de CPU"""
        return self.thread_pool.submit(func, *args, **kwargs)
    
    def execute_parallel_task(self, func: Callable, data: List[Any]) -> List[Any]:
        """Ejecutar tarea en paralelo"""
        with self.process_pool as executor:
            futures = [executor.submit(func, item) for item in data]
            return [future.result() for future in futures]
    
    def get_cpu_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de CPU"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "thread_pool_workers": self.thread_pool._max_workers,
            "process_pool_workers": self.process_pool._max_workers
        }


class CacheOptimizer:
    """Optimizador de caché"""
    
    def __init__(self):
        self.cache_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "hits": 0, "misses": 0, "size": 0, "max_size": 1000
        })
        self.cache_policies: Dict[str, str] = {}  # LRU, LFU, TTL
        self._lock = threading.Lock()
    
    async def optimize_cache_performance(self) -> Dict[str, Any]:
        """Optimizar performance de caché"""
        try:
            optimizations = []
            
            # Analizar hit rates de caché
            for cache_name, stats in self.cache_stats.items():
                hit_rate = stats["hits"] / (stats["hits"] + stats["misses"]) if (stats["hits"] + stats["misses"]) > 0 else 0
                
                if hit_rate < 0.5:  # Hit rate bajo
                    # Aumentar tamaño de caché
                    stats["max_size"] = min(stats["max_size"] * 2, 10000)
                    optimizations.append(f"{cache_name}_size_increased")
                elif hit_rate > 0.9 and stats["size"] < stats["max_size"] * 0.5:
                    # Reducir tamaño de caché si está subutilizado
                    stats["max_size"] = max(stats["max_size"] // 2, 100)
                    optimizations.append(f"{cache_name}_size_decreased")
                
                # Optimizar política de caché
                if hit_rate < 0.3:
                    self.cache_policies[cache_name] = "LFU"  # Least Frequently Used
                    optimizations.append(f"{cache_name}_policy_changed_to_LFU")
                elif hit_rate > 0.8:
                    self.cache_policies[cache_name] = "LRU"  # Least Recently Used
                    optimizations.append(f"{cache_name}_policy_changed_to_LRU")
            
            return {
                "cache_stats": dict(self.cache_stats),
                "cache_policies": dict(self.cache_policies),
                "optimizations_applied": optimizations,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing cache performance: {e}")
            return {"error": str(e)}
    
    def record_cache_hit(self, cache_name: str):
        """Registrar hit de caché"""
        async with self._lock:
            self.cache_stats[cache_name]["hits"] += 1
    
    def record_cache_miss(self, cache_name: str):
        """Registrar miss de caché"""
        async with self._lock:
            self.cache_stats[cache_name]["misses"] += 1
    
    def update_cache_size(self, cache_name: str, size: int):
        """Actualizar tamaño de caché"""
        async with self._lock:
            self.cache_stats[cache_name]["size"] = size
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de caché"""
        return {
            "cache_stats": dict(self.cache_stats),
            "cache_policies": dict(self.cache_policies),
            "total_caches": len(self.cache_stats)
        }


class AlgorithmOptimizer:
    """Optimizador de algoritmos"""
    
    def __init__(self):
        self.algorithm_performance: Dict[str, Dict[str, Any]] = {}
        self.optimization_suggestions: Dict[str, List[str]] = {}
    
    async def optimize_algorithm_performance(self, algorithm_name: str, 
                                           input_size: int, execution_time: float) -> Dict[str, Any]:
        """Optimizar performance de algoritmo"""
        try:
            # Registrar performance del algoritmo
            if algorithm_name not in self.algorithm_performance:
                self.algorithm_performance[algorithm_name] = {
                    "execution_times": [],
                    "input_sizes": [],
                    "complexity": "unknown",
                    "optimization_level": OptimizationLevel.BASIC
                }
            
            perf_data = self.algorithm_performance[algorithm_name]
            perf_data["execution_times"].append(execution_time)
            perf_data["input_sizes"].append(input_size)
            
            # Mantener solo los últimos 100 registros
            if len(perf_data["execution_times"]) > 100:
                perf_data["execution_times"] = perf_data["execution_times"][-100:]
                perf_data["input_sizes"] = perf_data["input_sizes"][-100:]
            
            # Analizar complejidad
            complexity = self._analyze_complexity(perf_data)
            perf_data["complexity"] = complexity
            
            # Generar sugerencias de optimización
            suggestions = self._generate_optimization_suggestions(algorithm_name, complexity, execution_time)
            self.optimization_suggestions[algorithm_name] = suggestions
            
            return {
                "algorithm_name": algorithm_name,
                "complexity": complexity,
                "suggestions": suggestions,
                "performance_trend": self._calculate_performance_trend(perf_data),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing algorithm performance: {e}")
            return {"error": str(e)}
    
    def _analyze_complexity(self, perf_data: Dict[str, Any]) -> str:
        """Analizar complejidad del algoritmo"""
        if len(perf_data["execution_times"]) < 3:
            return "insufficient_data"
        
        times = np.array(perf_data["execution_times"])
        sizes = np.array(perf_data["input_sizes"])
        
        # Calcular correlación entre tamaño de entrada y tiempo de ejecución
        correlation = np.corrcoef(sizes, times)[0, 1]
        
        if correlation > 0.9:
            return "O(n)"  # Lineal
        elif correlation > 0.7:
            return "O(n log n)"  # Logarítmico
        elif correlation > 0.5:
            return "O(n²)"  # Cuadrático
        else:
            return "O(1)"  # Constante
    
    def _generate_optimization_suggestions(self, algorithm_name: str, complexity: str, 
                                         execution_time: float) -> List[str]:
        """Generar sugerencias de optimización"""
        suggestions = []
        
        if complexity == "O(n²)" and execution_time > 1.0:
            suggestions.extend([
                "Consider using a more efficient algorithm with O(n log n) complexity",
                "Implement caching for repeated calculations",
                "Use parallel processing for independent operations"
            ])
        elif complexity == "O(n)" and execution_time > 0.5:
            suggestions.extend([
                "Consider using vectorized operations",
                "Implement early termination conditions",
                "Use more efficient data structures"
            ])
        elif execution_time > 0.1:
            suggestions.extend([
                "Profile the algorithm to identify bottlenecks",
                "Consider using compiled extensions (Cython, Numba)",
                "Implement memoization for recursive calls"
            ])
        
        return suggestions
    
    def _calculate_performance_trend(self, perf_data: Dict[str, Any]) -> str:
        """Calcular tendencia de performance"""
        if len(perf_data["execution_times"]) < 5:
            return "insufficient_data"
        
        recent_times = perf_data["execution_times"][-5:]
        older_times = perf_data["execution_times"][-10:-5] if len(perf_data["execution_times"]) >= 10 else []
        
        if not older_times:
            return "insufficient_data"
        
        recent_avg = np.mean(recent_times)
        older_avg = np.mean(older_times)
        
        if recent_avg < older_avg * 0.9:
            return "improving"
        elif recent_avg > older_avg * 1.1:
            return "degrading"
        else:
            return "stable"


class ConcurrencyOptimizer:
    """Optimizador de concurrencia"""
    
    def __init__(self):
        self.concurrent_requests = 0
        self.max_concurrent_requests = 100
        self.request_queue = deque()
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self._lock = threading.Lock()
    
    async def optimize_concurrency(self) -> Dict[str, Any]:
        """Optimizar concurrencia"""
        try:
            optimizations = []
            
            # Ajustar límite de requests concurrentes basado en carga
            current_load = self.concurrent_requests / self.max_concurrent_requests
            
            if current_load > 0.9:
                # Reducir límite si está sobrecargado
                new_limit = max(10, int(self.max_concurrent_requests * 0.8))
                self.max_concurrent_requests = new_limit
                self.semaphore = asyncio.Semaphore(new_limit)
                optimizations.append(f"concurrent_limit_reduced_to_{new_limit}")
            elif current_load < 0.3:
                # Aumentar límite si está subutilizado
                new_limit = min(200, int(self.max_concurrent_requests * 1.2))
                self.max_concurrent_requests = new_limit
                self.semaphore = asyncio.Semaphore(new_limit)
                optimizations.append(f"concurrent_limit_increased_to_{new_limit}")
            
            # Optimizar cola de requests
            await self._optimize_request_queue()
            optimizations.append("request_queue_optimized")
            
            return {
                "current_concurrent_requests": self.concurrent_requests,
                "max_concurrent_requests": self.max_concurrent_requests,
                "queue_size": len(self.request_queue),
                "load_percentage": current_load * 100,
                "optimizations_applied": optimizations,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing concurrency: {e}")
            return {"error": str(e)}
    
    async def _optimize_request_queue(self):
        """Optimizar cola de requests"""
        # Limpiar requests antiguos de la cola
        current_time = time.time()
        while self.request_queue and current_time - self.request_queue[0] > 300:  # 5 minutos
            self.request_queue.popleft()
    
    async def acquire_semaphore(self):
        """Adquirir semáforo para request"""
        await self.semaphore.acquire()
        async with self._lock:
            self.concurrent_requests += 1
            self.request_queue.append(time.time())
    
    async def release_semaphore(self):
        """Liberar semáforo después de request"""
        self.semaphore.release()
        async with self._lock:
            self.concurrent_requests = max(0, self.concurrent_requests - 1)


class PerformanceOptimizer:
    """Optimizador principal de performance"""
    
    def __init__(self):
        self.memory_optimizer = MemoryOptimizer()
        self.cpu_optimizer = CPUOptimizer()
        self.cache_optimizer = CacheOptimizer()
        self.algorithm_optimizer = AlgorithmOptimizer()
        self.concurrency_optimizer = ConcurrencyOptimizer()
        self.optimization_profiles: Dict[str, PerformanceProfile] = {}
        self.optimization_rules: Dict[str, OptimizationRule] = {}
        self.performance_alerts: Dict[str, PerformanceAlert] = {}
        self.is_running = False
        self._optimization_task = None
        self._monitoring_task = None
        self._lock = threading.Lock()
    
    async def start(self):
        """Iniciar optimizador de performance"""
        try:
            self.is_running = True
            
            # Inicializar perfiles de optimización por defecto
            await self._initialize_default_profiles()
            
            # Iniciar tareas de optimización y monitoreo
            self._optimization_task = asyncio.create_task(self._optimization_loop())
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("Performance optimizer started")
            
        except Exception as e:
            logger.error(f"Error starting performance optimizer: {e}")
            raise
    
    async def stop(self):
        """Detener optimizador de performance"""
        try:
            self.is_running = False
            
            # Detener tareas
            if self._optimization_task:
                self._optimization_task.cancel()
                try:
                    await self._optimization_task
                except asyncio.CancelledError:
                    pass
            
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Performance optimizer stopped")
            
        except Exception as e:
            logger.error(f"Error stopping performance optimizer: {e}")
    
    async def _initialize_default_profiles(self):
        """Inicializar perfiles de optimización por defecto"""
        default_profiles = [
            {
                "name": "High Performance",
                "optimization_level": OptimizationLevel.ULTRA,
                "target_metrics": {
                    PerformanceMetric.RESPONSE_TIME: 0.1,
                    PerformanceMetric.THROUGHPUT: 1000,
                    PerformanceMetric.MEMORY_USAGE: 0.7,
                    PerformanceMetric.CPU_USAGE: 0.8
                }
            },
            {
                "name": "Balanced",
                "optimization_level": OptimizationLevel.ADVANCED,
                "target_metrics": {
                    PerformanceMetric.RESPONSE_TIME: 0.5,
                    PerformanceMetric.THROUGHPUT: 500,
                    PerformanceMetric.MEMORY_USAGE: 0.8,
                    PerformanceMetric.CPU_USAGE: 0.9
                }
            },
            {
                "name": "Resource Efficient",
                "optimization_level": OptimizationLevel.INTERMEDIATE,
                "target_metrics": {
                    PerformanceMetric.RESPONSE_TIME: 1.0,
                    PerformanceMetric.THROUGHPUT: 200,
                    PerformanceMetric.MEMORY_USAGE: 0.6,
                    PerformanceMetric.CPU_USAGE: 0.7
                }
            }
        ]
        
        for profile_data in default_profiles:
            await self.create_optimization_profile(
                name=profile_data["name"],
                optimization_level=profile_data["optimization_level"],
                target_metrics=profile_data["target_metrics"]
            )
    
    async def _optimization_loop(self):
        """Loop de optimización"""
        while self.is_running:
            try:
                # Ejecutar optimizaciones
                await self._run_optimizations()
                
                # Aplicar reglas de optimización
                await self._apply_optimization_rules()
                
                await asyncio.sleep(30)  # Optimizar cada 30 segundos
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(30)
    
    async def _monitoring_loop(self):
        """Loop de monitoreo"""
        while self.is_running:
            try:
                # Monitorear métricas de performance
                await self._monitor_performance_metrics()
                
                # Verificar alertas
                await self._check_performance_alerts()
                
                await asyncio.sleep(10)  # Monitorear cada 10 segundos
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _run_optimizations(self):
        """Ejecutar optimizaciones"""
        try:
            # Optimizar memoria
            memory_result = await self.memory_optimizer.optimize_memory_usage()
            
            # Optimizar CPU
            cpu_result = await self.cpu_optimizer.optimize_cpu_usage()
            
            # Optimizar caché
            cache_result = await self.cache_optimizer.optimize_cache_performance()
            
            # Optimizar concurrencia
            concurrency_result = await self.concurrency_optimizer.optimize_concurrency()
            
            logger.debug(f"Optimizations completed: memory={memory_result}, cpu={cpu_result}, cache={cache_result}, concurrency={concurrency_result}")
            
        except Exception as e:
            logger.error(f"Error running optimizations: {e}")
    
    async def _apply_optimization_rules(self):
        """Aplicar reglas de optimización"""
        for rule_id, rule in self.optimization_rules.items():
            if not rule.is_enabled:
                continue
            
            try:
                # Evaluar condición de la regla
                if await self._evaluate_rule_condition(rule):
                    # Ejecutar acción de la regla
                    await self._execute_rule_action(rule)
                    rule.last_triggered = time.time()
                    rule.trigger_count += 1
                    
            except Exception as e:
                logger.error(f"Error applying optimization rule {rule_id}: {e}")
    
    async def _evaluate_rule_condition(self, rule: OptimizationRule) -> bool:
        """Evaluar condición de regla"""
        # Implementar lógica de evaluación de condiciones
        # Por simplicidad, retornar False
        return False
    
    async def _execute_rule_action(self, rule: OptimizationRule):
        """Ejecutar acción de regla"""
        # Implementar lógica de ejecución de acciones
        pass
    
    async def _monitor_performance_metrics(self):
        """Monitorear métricas de performance"""
        try:
            # Obtener métricas actuales
            memory_stats = self.memory_optimizer.get_memory_stats()
            cpu_stats = self.cpu_optimizer.get_cpu_stats()
            cache_stats = self.cache_optimizer.get_cache_stats()
            
            # Actualizar perfiles de optimización
            for profile in self.optimization_profiles.values():
                profile.current_metrics = {
                    PerformanceMetric.MEMORY_USAGE: memory_stats["memory_percent"] / 100.0,
                    PerformanceMetric.CPU_USAGE: cpu_stats["cpu_percent"] / 100.0,
                    PerformanceMetric.CACHE_HIT_RATE: self._calculate_overall_cache_hit_rate(cache_stats),
                    PerformanceMetric.CONCURRENT_REQUESTS: self.concurrency_optimizer.concurrent_requests
                }
                profile.last_updated = time.time()
                
        except Exception as e:
            logger.error(f"Error monitoring performance metrics: {e}")
    
    def _calculate_overall_cache_hit_rate(self, cache_stats: Dict[str, Any]) -> float:
        """Calcular hit rate general de caché"""
        total_hits = sum(stats["hits"] for stats in cache_stats["cache_stats"].values())
        total_misses = sum(stats["misses"] for stats in cache_stats["cache_stats"].values())
        
        if total_hits + total_misses == 0:
            return 0.0
        
        return total_hits / (total_hits + total_misses)
    
    async def _check_performance_alerts(self):
        """Verificar alertas de performance"""
        for profile in self.optimization_profiles.values():
            for metric, target_value in profile.target_metrics.items():
                current_value = profile.current_metrics.get(metric, 0)
                
                # Verificar si se excede el umbral
                if current_value > target_value * 1.2:  # 20% por encima del objetivo
                    await self._create_performance_alert(
                        metric, target_value, current_value, "high"
                    )
    
    async def _create_performance_alert(self, metric: PerformanceMetric, threshold: float, 
                                      current_value: float, severity: str):
        """Crear alerta de performance"""
        alert_id = f"alert_{int(time.time())}_{metric.value}"
        
        alert = PerformanceAlert(
            id=alert_id,
            metric=metric,
            threshold=threshold,
            current_value=current_value,
            severity=severity,
            message=f"{metric.value} exceeded threshold: {current_value:.2f} > {threshold:.2f}",
            timestamp=time.time(),
            is_resolved=False
        )
        
        async with self._lock:
            self.performance_alerts[alert_id] = alert
    
    async def create_optimization_profile(self, name: str, optimization_level: OptimizationLevel,
                                        target_metrics: Dict[PerformanceMetric, float]) -> str:
        """Crear perfil de optimización"""
        profile_id = f"profile_{int(time.time())}_{name.lower().replace(' ', '_')}"
        
        profile = PerformanceProfile(
            id=profile_id,
            name=name,
            optimization_level=optimization_level,
            target_metrics=target_metrics,
            current_metrics={},
            optimization_rules=[],
            created_at=time.time(),
            last_updated=time.time(),
            is_active=True
        )
        
        async with self._lock:
            self.optimization_profiles[profile_id] = profile
        
        return profile_id
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        return {
            "is_running": self.is_running,
            "memory_stats": self.memory_optimizer.get_memory_stats(),
            "cpu_stats": self.cpu_optimizer.get_cpu_stats(),
            "cache_stats": self.cache_optimizer.get_cache_stats(),
            "concurrency_stats": {
                "current_requests": self.concurrency_optimizer.concurrent_requests,
                "max_requests": self.concurrency_optimizer.max_concurrent_requests,
                "queue_size": len(self.concurrency_optimizer.request_queue)
            },
            "optimization_profiles": len(self.optimization_profiles),
            "optimization_rules": len(self.optimization_rules),
            "active_alerts": len([a for a in self.performance_alerts.values() if not a.is_resolved])
        }


# Instancia global del optimizador de performance
performance_optimizer = PerformanceOptimizer()


# Router para endpoints de optimización de performance
performance_optimizer_router = APIRouter()


@performance_optimizer_router.post("/performance/optimize/memory")
async def optimize_memory_endpoint():
    """Optimizar memoria"""
    try:
        result = await performance_optimizer.memory_optimizer.optimize_memory_usage()
        return {
            "message": "Memory optimization completed",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error optimizing memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize memory: {str(e)}")


@performance_optimizer_router.post("/performance/optimize/cpu")
async def optimize_cpu_endpoint():
    """Optimizar CPU"""
    try:
        result = await performance_optimizer.cpu_optimizer.optimize_cpu_usage()
        return {
            "message": "CPU optimization completed",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error optimizing CPU: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize CPU: {str(e)}")


@performance_optimizer_router.post("/performance/optimize/cache")
async def optimize_cache_endpoint():
    """Optimizar caché"""
    try:
        result = await performance_optimizer.cache_optimizer.optimize_cache_performance()
        return {
            "message": "Cache optimization completed",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error optimizing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize cache: {str(e)}")


@performance_optimizer_router.post("/performance/optimize/concurrency")
async def optimize_concurrency_endpoint():
    """Optimizar concurrencia"""
    try:
        result = await performance_optimizer.concurrency_optimizer.optimize_concurrency()
        return {
            "message": "Concurrency optimization completed",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error optimizing concurrency: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize concurrency: {str(e)}")


@performance_optimizer_router.post("/performance/optimize/algorithm")
async def optimize_algorithm_endpoint(optimization_data: dict):
    """Optimizar algoritmo"""
    try:
        algorithm_name = optimization_data["algorithm_name"]
        input_size = optimization_data["input_size"]
        execution_time = optimization_data["execution_time"]
        
        result = await performance_optimizer.algorithm_optimizer.optimize_algorithm_performance(
            algorithm_name, input_size, execution_time
        )
        
        return {
            "message": "Algorithm optimization completed",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error optimizing algorithm: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize algorithm: {str(e)}")


@performance_optimizer_router.post("/performance/profiles")
async def create_optimization_profile_endpoint(profile_data: dict):
    """Crear perfil de optimización"""
    try:
        name = profile_data["name"]
        optimization_level = OptimizationLevel(profile_data["optimization_level"])
        target_metrics = {
            PerformanceMetric(metric): value 
            for metric, value in profile_data["target_metrics"].items()
        }
        
        profile_id = await performance_optimizer.create_optimization_profile(
            name, optimization_level, target_metrics
        )
        
        return {
            "message": "Optimization profile created successfully",
            "profile_id": profile_id,
            "name": name
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid optimization level or metric: {e}")
    except Exception as e:
        logger.error(f"Error creating optimization profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create optimization profile: {str(e)}")


@performance_optimizer_router.get("/performance/profiles")
async def get_optimization_profiles_endpoint():
    """Obtener perfiles de optimización"""
    try:
        profiles = performance_optimizer.optimization_profiles
        return {
            "profiles": [
                {
                    "id": profile.id,
                    "name": profile.name,
                    "optimization_level": profile.optimization_level.value,
                    "target_metrics": {metric.value: value for metric, value in profile.target_metrics.items()},
                    "current_metrics": {metric.value: value for metric, value in profile.current_metrics.items()},
                    "is_active": profile.is_active,
                    "created_at": profile.created_at,
                    "last_updated": profile.last_updated
                }
                for profile in profiles.values()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting optimization profiles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get optimization profiles: {str(e)}")


@performance_optimizer_router.get("/performance/alerts")
async def get_performance_alerts_endpoint():
    """Obtener alertas de performance"""
    try:
        alerts = performance_optimizer.performance_alerts
        return {
            "alerts": [
                {
                    "id": alert.id,
                    "metric": alert.metric.value,
                    "threshold": alert.threshold,
                    "current_value": alert.current_value,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                    "is_resolved": alert.is_resolved
                }
                for alert in alerts.values()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting performance alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance alerts: {str(e)}")


@performance_optimizer_router.get("/performance/stats")
async def get_performance_optimizer_stats_endpoint():
    """Obtener estadísticas del optimizador de performance"""
    try:
        stats = await performance_optimizer.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting performance optimizer stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance optimizer stats: {str(e)}")


# Funciones de utilidad para integración
async def start_performance_optimizer():
    """Iniciar optimizador de performance"""
    await performance_optimizer.start()


async def stop_performance_optimizer():
    """Detener optimizador de performance"""
    await performance_optimizer.stop()


async def optimize_algorithm_performance(algorithm_name: str, input_size: int, execution_time: float) -> Dict[str, Any]:
    """Optimizar performance de algoritmo"""
    return await performance_optimizer.algorithm_optimizer.optimize_algorithm_performance(
        algorithm_name, input_size, execution_time
    )


async def get_performance_optimizer_stats() -> Dict[str, Any]:
    """Obtener estadísticas del optimizador de performance"""
    return await performance_optimizer.get_system_stats()


# Decoradores de optimización
def optimize_memory(func):
    """Decorador para optimización de memoria"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Adquirir semáforo de concurrencia
        await performance_optimizer.concurrency_optimizer.acquire_semaphore()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            await performance_optimizer.concurrency_optimizer.release_semaphore()
    return wrapper


def optimize_cpu(func):
    """Decorador para optimización de CPU"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Registrar performance del algoritmo
            await performance_optimizer.algorithm_optimizer.optimize_algorithm_performance(
                func.__name__, len(args) + len(kwargs), execution_time
            )
            
            return result
        except Exception as e:
            logger.error(f"Error in optimized function {func.__name__}: {e}")
            raise
    return wrapper


def optimize_cache(cache_name: str):
    """Decorador para optimización de caché"""
    def decorator(func):
        @lru_cache(maxsize=1000)
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Registrar hit/miss de caché
            cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            try:
                result = func(*args, **kwargs)
                performance_optimizer.cache_optimizer.record_cache_hit(cache_name)
                return result
            except KeyError:
                performance_optimizer.cache_optimizer.record_cache_miss(cache_name)
                raise
        return wrapper
    return decorator


logger.info("Performance optimizer module loaded successfully")

