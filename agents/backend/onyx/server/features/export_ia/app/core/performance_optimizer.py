"""
Performance Optimizer - Sistema de optimización de rendimiento avanzado
"""

import asyncio
import logging
import time
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Niveles de optimización."""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    ENTERPRISE = "enterprise"


class CacheStrategy(Enum):
    """Estrategias de cache."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    active_connections: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CacheEntry:
    """Entrada de cache."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[timedelta] = None


class PerformanceOptimizer:
    """
    Sistema de optimización de rendimiento avanzado.
    """
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.STANDARD):
        """Inicializar optimizador de rendimiento."""
        self.optimization_level = optimization_level
        self.metrics_history: List[PerformanceMetrics] = []
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_strategy = CacheStrategy.ADAPTIVE
        self.max_cache_size = 1000
        self.max_cache_memory_mb = 500
        
        # Configuración de optimización
        self.enable_caching = True
        self.enable_compression = True
        self.enable_parallel_processing = True
        self.enable_memory_optimization = True
        self.enable_gc_optimization = True
        
        # Thread pools
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        
        # Monitoreo
        self.monitoring_active = False
        self.monitoring_interval = 5  # segundos
        self.metrics_collector_task = None
        
        # Configuración por nivel
        self._configure_by_level()
        
        logger.info(f"PerformanceOptimizer inicializado con nivel {optimization_level.value}")
    
    def _configure_by_level(self):
        """Configurar optimizaciones según el nivel."""
        if self.optimization_level == OptimizationLevel.BASIC:
            self.max_cache_size = 100
            self.max_cache_memory_mb = 50
            self.enable_compression = False
            self.enable_parallel_processing = False
        elif self.optimization_level == OptimizationLevel.STANDARD:
            self.max_cache_size = 500
            self.max_cache_memory_mb = 200
            self.enable_compression = True
            self.enable_parallel_processing = True
        elif self.optimization_level == OptimizationLevel.ADVANCED:
            self.max_cache_size = 1000
            self.max_cache_memory_mb = 500
            self.enable_compression = True
            self.enable_parallel_processing = True
            self.enable_memory_optimization = True
        elif self.optimization_level == OptimizationLevel.ENTERPRISE:
            self.max_cache_size = 5000
            self.max_cache_memory_mb = 2000
            self.enable_compression = True
            self.enable_parallel_processing = True
            self.enable_memory_optimization = True
            self.enable_gc_optimization = True
    
    async def initialize(self):
        """Inicializar el optimizador de rendimiento."""
        try:
            # Iniciar monitoreo de métricas
            await self.start_monitoring()
            
            # Configurar optimizaciones del sistema
            await self._configure_system_optimizations()
            
            logger.info("PerformanceOptimizer inicializado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar PerformanceOptimizer: {e}")
            raise
    
    async def shutdown(self):
        """Cerrar el optimizador de rendimiento."""
        try:
            # Detener monitoreo
            await self.stop_monitoring()
            
            # Limpiar cache
            await self.clear_cache()
            
            # Cerrar thread pools
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
            
            logger.info("PerformanceOptimizer cerrado")
            
        except Exception as e:
            logger.error(f"Error al cerrar PerformanceOptimizer: {e}")
    
    async def start_monitoring(self):
        """Iniciar monitoreo de métricas."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.metrics_collector_task = asyncio.create_task(self._metrics_collector())
        logger.info("Monitoreo de métricas iniciado")
    
    async def stop_monitoring(self):
        """Detener monitoreo de métricas."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.metrics_collector_task:
            self.metrics_collector_task.cancel()
            try:
                await self.metrics_collector_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Monitoreo de métricas detenido")
    
    async def _metrics_collector(self):
        """Recolector de métricas del sistema."""
        while self.monitoring_active:
            try:
                metrics = await self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Mantener solo las últimas 1000 métricas
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Optimizaciones automáticas
                await self._auto_optimize()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error en recolector de métricas: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_system_metrics(self) -> PerformanceMetrics:
        """Recolectar métricas del sistema."""
        try:
            # Métricas de CPU
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Métricas de memoria
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            memory_available = memory.available / (1024**3)  # GB
            
            # Métricas de disco
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Métricas de red
            network_io = psutil.net_io_counters()
            network_metrics = {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv
            }
            
            # Métricas de cache
            cache_hit_rate = await self._calculate_cache_hit_rate()
            
            return PerformanceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                memory_available=memory_available,
                disk_usage=disk_usage,
                network_io=network_metrics,
                cache_hit_rate=cache_hit_rate,
                active_connections=len(psutil.net_connections()),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error al recolectar métricas: {e}")
            return PerformanceMetrics()
    
    async def _auto_optimize(self):
        """Optimizaciones automáticas basadas en métricas."""
        if not self.metrics_history:
            return
        
        latest_metrics = self.metrics_history[-1]
        
        # Optimización de memoria
        if latest_metrics.memory_usage > 80:
            await self._optimize_memory()
        
        # Optimización de cache
        if latest_metrics.cache_hit_rate < 0.7:
            await self._optimize_cache()
        
        # Optimización de CPU
        if latest_metrics.cpu_usage > 90:
            await self._optimize_cpu()
    
    async def _optimize_memory(self):
        """Optimizar uso de memoria."""
        try:
            if self.enable_memory_optimization:
                # Limpiar cache si es necesario
                if len(self.cache) > self.max_cache_size * 0.8:
                    await self._cleanup_cache()
                
                # Forzar garbage collection
                if self.enable_gc_optimization:
                    gc.collect()
                
                logger.info("Optimización de memoria ejecutada")
                
        except Exception as e:
            logger.error(f"Error en optimización de memoria: {e}")
    
    async def _optimize_cache(self):
        """Optimizar estrategia de cache."""
        try:
            # Cambiar estrategia de cache si es necesario
            if self.cache_strategy == CacheStrategy.LRU:
                self.cache_strategy = CacheStrategy.LFU
            elif self.cache_strategy == CacheStrategy.LFU:
                self.cache_strategy = CacheStrategy.TTL
            else:
                self.cache_strategy = CacheStrategy.ADAPTIVE
            
            # Limpiar entradas expiradas
            await self._cleanup_expired_cache()
            
            logger.info(f"Estrategia de cache cambiada a {self.cache_strategy.value}")
            
        except Exception as e:
            logger.error(f"Error en optimización de cache: {e}")
    
    async def _optimize_cpu(self):
        """Optimizar uso de CPU."""
        try:
            # Reducir carga de procesamiento paralelo si es necesario
            if self.enable_parallel_processing:
                # Ajustar número de workers
                current_workers = self.thread_pool._max_workers
                if current_workers > 2:
                    # Crear nuevo pool con menos workers
                    self.thread_pool.shutdown(wait=False)
                    self.thread_pool = ThreadPoolExecutor(max_workers=max(2, current_workers - 1))
                
                logger.info("Optimización de CPU ejecutada")
                
        except Exception as e:
            logger.error(f"Error en optimización de CPU: {e}")
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Obtener valor del cache."""
        if not self.enable_caching or key not in self.cache:
            return None
        
        try:
            entry = self.cache[key]
            
            # Verificar TTL
            if entry.ttl and datetime.now() - entry.created_at > entry.ttl:
                del self.cache[key]
                return None
            
            # Actualizar estadísticas de acceso
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            
            return entry.value
            
        except Exception as e:
            logger.error(f"Error al obtener del cache: {e}")
            return None
    
    async def cache_set(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None,
        compress: bool = None
    ) -> bool:
        """Establecer valor en el cache."""
        if not self.enable_caching:
            return False
        
        try:
            # Determinar si comprimir
            should_compress = compress if compress is not None else self.enable_compression
            
            # Comprimir valor si es necesario
            if should_compress and isinstance(value, (str, bytes)):
                value = await self._compress_value(value)
            
            # Calcular tamaño
            size_bytes = await self._calculate_size(value)
            
            # Verificar límites de memoria
            if size_bytes > self.max_cache_memory_mb * 1024 * 1024:
                logger.warning(f"Valor demasiado grande para cache: {size_bytes} bytes")
                return False
            
            # Crear entrada de cache
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                size_bytes=size_bytes,
                ttl=ttl
            )
            
            # Limpiar cache si es necesario
            if len(self.cache) >= self.max_cache_size:
                await self._cleanup_cache()
            
            # Agregar al cache
            self.cache[key] = entry
            
            return True
            
        except Exception as e:
            logger.error(f"Error al establecer en cache: {e}")
            return False
    
    async def cache_delete(self, key: str) -> bool:
        """Eliminar valor del cache."""
        try:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
        except Exception as e:
            logger.error(f"Error al eliminar del cache: {e}")
            return False
    
    async def clear_cache(self):
        """Limpiar todo el cache."""
        try:
            self.cache.clear()
            logger.info("Cache limpiado")
        except Exception as e:
            logger.error(f"Error al limpiar cache: {e}")
    
    async def _cleanup_cache(self):
        """Limpiar cache según la estrategia."""
        try:
            if not self.cache:
                return
            
            # Limpiar entradas expiradas
            await self._cleanup_expired_cache()
            
            # Si aún necesitamos espacio, aplicar estrategia
            if len(self.cache) >= self.max_cache_size:
                if self.cache_strategy == CacheStrategy.LRU:
                    await self._cleanup_lru()
                elif self.cache_strategy == CacheStrategy.LFU:
                    await self._cleanup_lfu()
                elif self.cache_strategy == CacheStrategy.ADAPTIVE:
                    await self._cleanup_adaptive()
            
        except Exception as e:
            logger.error(f"Error en limpieza de cache: {e}")
    
    async def _cleanup_expired_cache(self):
        """Limpiar entradas expiradas del cache."""
        expired_keys = []
        now = datetime.now()
        
        for key, entry in self.cache.items():
            if entry.ttl and now - entry.created_at > entry.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
    
    async def _cleanup_lru(self):
        """Limpiar cache usando estrategia LRU."""
        # Ordenar por último acceso
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Eliminar el 20% más antiguo
        to_remove = len(sorted_entries) // 5
        for key, _ in sorted_entries[:to_remove]:
            del self.cache[key]
    
    async def _cleanup_lfu(self):
        """Limpiar cache usando estrategia LFU."""
        # Ordenar por frecuencia de acceso
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].access_count
        )
        
        # Eliminar el 20% menos usado
        to_remove = len(sorted_entries) // 5
        for key, _ in sorted_entries[:to_remove]:
            del self.cache[key]
    
    async def _cleanup_adaptive(self):
        """Limpiar cache usando estrategia adaptiva."""
        # Combinar LRU y LFU
        now = datetime.now()
        
        for key, entry in list(self.cache.items()):
            # Eliminar entradas muy antiguas y poco usadas
            age_hours = (now - entry.created_at).total_seconds() / 3600
            if age_hours > 24 and entry.access_count < 2:
                del self.cache[key]
    
    async def _calculate_cache_hit_rate(self) -> float:
        """Calcular tasa de aciertos del cache."""
        if not self.cache:
            return 0.0
        
        total_accesses = sum(entry.access_count for entry in self.cache.values())
        if total_accesses == 0:
            return 0.0
        
        # Simular tasa de aciertos basada en acceso reciente
        recent_accesses = sum(
            1 for entry in self.cache.values()
            if (datetime.now() - entry.last_accessed).total_seconds() < 3600
        )
        
        return recent_accesses / len(self.cache) if self.cache else 0.0
    
    async def _compress_value(self, value: Union[str, bytes]) -> bytes:
        """Comprimir valor para cache."""
        try:
            import gzip
            
            if isinstance(value, str):
                value = value.encode('utf-8')
            
            compressed = gzip.compress(value)
            return compressed
            
        except Exception as e:
            logger.error(f"Error al comprimir valor: {e}")
            return value if isinstance(value, bytes) else value.encode('utf-8')
    
    async def _decompress_value(self, value: bytes) -> Union[str, bytes]:
        """Descomprimir valor del cache."""
        try:
            import gzip
            
            decompressed = gzip.decompress(value)
            return decompressed.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error al descomprimir valor: {e}")
            return value
    
    async def _calculate_size(self, value: Any) -> int:
        """Calcular tamaño de un valor en bytes."""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (dict, list)):
                return len(json.dumps(value, default=str).encode('utf-8'))
            else:
                return len(str(value).encode('utf-8'))
        except Exception:
            return 1024  # Tamaño estimado por defecto
    
    async def _configure_system_optimizations(self):
        """Configurar optimizaciones del sistema."""
        try:
            # Configurar garbage collection
            if self.enable_gc_optimization:
                gc.set_threshold(700, 10, 10)
            
            # Configurar límites de memoria si es posible
            if hasattr(psutil, 'RLIMIT_AS'):
                import resource
                # Establecer límite de memoria (opcional)
                pass
            
            logger.info("Optimizaciones del sistema configuradas")
            
        except Exception as e:
            logger.error(f"Error al configurar optimizaciones del sistema: {e}")
    
    async def execute_parallel(
        self,
        tasks: List[Callable],
        max_workers: Optional[int] = None
    ) -> List[Any]:
        """Ejecutar tareas en paralelo."""
        if not self.enable_parallel_processing:
            # Ejecutar secuencialmente
            results = []
            for task in tasks:
                if asyncio.iscoroutinefunction(task):
                    result = await task()
                else:
                    result = task()
                results.append(result)
            return results
        
        try:
            # Ejecutar en paralelo
            if max_workers is None:
                max_workers = min(len(tasks), 4)
            
            # Usar asyncio para tareas asíncronas
            if all(asyncio.iscoroutinefunction(task) for task in tasks):
                results = await asyncio.gather(*[task() for task in tasks])
            else:
                # Usar thread pool para tareas síncronas
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    self.thread_pool,
                    lambda: [task() for task in tasks]
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error en ejecución paralela: {e}")
            # Fallback a ejecución secuencial
            return await self.execute_parallel(tasks, max_workers=1)
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de rendimiento."""
        try:
            if not self.metrics_history:
                return {"error": "No hay métricas disponibles"}
            
            latest_metrics = self.metrics_history[-1]
            
            # Calcular promedios
            if len(self.metrics_history) > 1:
                avg_cpu = sum(m.cpu_usage for m in self.metrics_history[-10:]) / min(10, len(self.metrics_history))
                avg_memory = sum(m.memory_usage for m in self.metrics_history[-10:]) / min(10, len(self.metrics_history))
                avg_cache_hit_rate = sum(m.cache_hit_rate for m in self.metrics_history[-10:]) / min(10, len(self.metrics_history))
            else:
                avg_cpu = latest_metrics.cpu_usage
                avg_memory = latest_metrics.memory_usage
                avg_cache_hit_rate = latest_metrics.cache_hit_rate
            
            return {
                "current": {
                    "cpu_usage": latest_metrics.cpu_usage,
                    "memory_usage": latest_metrics.memory_usage,
                    "memory_available_gb": latest_metrics.memory_available,
                    "disk_usage": latest_metrics.disk_usage,
                    "cache_hit_rate": latest_metrics.cache_hit_rate,
                    "active_connections": latest_metrics.active_connections,
                    "timestamp": latest_metrics.timestamp.isoformat()
                },
                "averages": {
                    "cpu_usage": avg_cpu,
                    "memory_usage": avg_memory,
                    "cache_hit_rate": avg_cache_hit_rate
                },
                "cache": {
                    "size": len(self.cache),
                    "max_size": self.max_cache_size,
                    "strategy": self.cache_strategy.value,
                    "memory_usage_mb": sum(entry.size_bytes for entry in self.cache.values()) / (1024 * 1024)
                },
                "optimization": {
                    "level": self.optimization_level.value,
                    "caching_enabled": self.enable_caching,
                    "compression_enabled": self.enable_compression,
                    "parallel_processing_enabled": self.enable_parallel_processing,
                    "memory_optimization_enabled": self.enable_memory_optimization,
                    "gc_optimization_enabled": self.enable_gc_optimization
                },
                "monitoring": {
                    "active": self.monitoring_active,
                    "interval_seconds": self.monitoring_interval,
                    "metrics_collected": len(self.metrics_history)
                }
            }
            
        except Exception as e:
            logger.error(f"Error al obtener métricas de rendimiento: {e}")
            return {"error": str(e)}
    
    async def optimize_system(self) -> Dict[str, Any]:
        """Optimizar sistema completo."""
        try:
            optimizations_applied = []
            
            # Optimización de memoria
            if self.enable_memory_optimization:
                await self._optimize_memory()
                optimizations_applied.append("memory_optimization")
            
            # Optimización de cache
            await self._optimize_cache()
            optimizations_applied.append("cache_optimization")
            
            # Optimización de CPU
            await self._optimize_cpu()
            optimizations_applied.append("cpu_optimization")
            
            # Garbage collection
            if self.enable_gc_optimization:
                gc.collect()
                optimizations_applied.append("garbage_collection")
            
            return {
                "optimizations_applied": optimizations_applied,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error en optimización del sistema: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del optimizador de rendimiento."""
        try:
            latest_metrics = self.metrics_history[-1] if self.metrics_history else None
            
            # Determinar estado de salud
            health_status = "healthy"
            issues = []
            
            if latest_metrics:
                if latest_metrics.cpu_usage > 90:
                    health_status = "warning"
                    issues.append("high_cpu_usage")
                
                if latest_metrics.memory_usage > 85:
                    health_status = "warning"
                    issues.append("high_memory_usage")
                
                if latest_metrics.cache_hit_rate < 0.5:
                    health_status = "warning"
                    issues.append("low_cache_hit_rate")
            
            return {
                "status": health_status,
                "issues": issues,
                "monitoring_active": self.monitoring_active,
                "cache_size": len(self.cache),
                "optimization_level": self.optimization_level.value,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




