"""
Optimized Export Engine - Motor optimizado con mejoras de rendimiento
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import weakref
import gc

from .models import ExportConfig, ExportFormat, DocumentType, QualityLevel
from .config import ConfigManager
from .task_manager import TaskManager
from .quality_manager import QualityManager
from ..exporters import ExporterFactory

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento del motor."""
    total_exports: int = 0
    successful_exports: int = 0
    failed_exports: int = 0
    average_processing_time: float = 0.0
    peak_memory_usage: float = 0.0
    cache_hit_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class OptimizedExportEngine:
    """
    Motor de exportación optimizado con mejoras de rendimiento.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Inicializar el motor optimizado."""
        self.config_manager = ConfigManager(config_path)
        self.task_manager = TaskManager(self.config_manager.system_config)
        self.quality_manager = QualityManager(self.config_manager)
        self.exporters = ExporterFactory()
        
        # Optimizaciones de rendimiento
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        self.cache = {}
        self.cache_ttl = 300  # 5 minutos
        self.performance_metrics = PerformanceMetrics()
        
        # Configuración de optimización
        self.enable_caching = True
        self.enable_parallel_processing = True
        self.enable_memory_optimization = True
        self.max_cache_size = 1000
        
        self._initialized = False
        self._start_time = time.time()
        
        logger.info("Optimized Export Engine inicializado")
    
    async def initialize(self):
        """Inicializar el motor con optimizaciones."""
        if not self._initialized:
            await self.task_manager.start()
            
            # Inicializar optimizaciones
            if self.enable_memory_optimization:
                await self._setup_memory_optimization()
            
            if self.enable_caching:
                await self._setup_caching()
            
            self._initialized = True
            logger.info("Optimized Export Engine completamente inicializado")
    
    async def shutdown(self):
        """Cerrar el motor y limpiar recursos."""
        if self._initialized:
            # Limpiar pools de threads
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
            
            # Limpiar cache
            if self.enable_caching:
                self.cache.clear()
            
            # Limpiar memoria
            if self.enable_memory_optimization:
                await self._cleanup_memory()
            
            await self.task_manager.stop()
            self._initialized = False
            logger.info("Optimized Export Engine cerrado")
    
    async def export_document(
        self,
        content: Dict[str, Any],
        config: ExportConfig,
        output_path: Optional[str] = None,
        optimize: bool = True
    ) -> str:
        """
        Exportar documento con optimizaciones.
        
        Args:
            content: Contenido del documento
            config: Configuración de exportación
            output_path: Ruta de salida opcional
            optimize: Habilitar optimizaciones
            
        Returns:
            ID de la tarea para seguimiento
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Validación optimizada
            await self._validate_input_optimized(content, config)
            
            # Verificar cache si está habilitado
            if self.enable_caching and optimize:
                cache_key = self._generate_cache_key(content, config)
                cached_result = await self._get_from_cache(cache_key)
                if cached_result:
                    logger.info(f"Resultado obtenido del cache: {cache_key}")
                    return cached_result
            
            # Procesamiento optimizado
            if self.enable_parallel_processing and optimize:
                task_id = await self._export_parallel(content, config, output_path)
            else:
                task_id = await self.task_manager.submit_task(content, config, output_path)
            
            # Actualizar métricas
            processing_time = time.time() - start_time
            await self._update_performance_metrics(processing_time, True)
            
            # Guardar en cache si está habilitado
            if self.enable_caching and optimize:
                await self._save_to_cache(cache_key, task_id)
            
            logger.info(f"Tarea de exportación optimizada creada: {task_id} (tiempo: {processing_time:.2f}s)")
            return task_id
            
        except Exception as e:
            processing_time = time.time() - start_time
            await self._update_performance_metrics(processing_time, False)
            logger.error(f"Error en exportación optimizada: {e}")
            raise
    
    async def _validate_input_optimized(self, content: Dict[str, Any], config: ExportConfig):
        """Validación optimizada de entrada."""
        if not content:
            raise ValueError("El contenido es requerido")
        
        if not config.format:
            raise ValueError("El formato de exportación es requerido")
        
        # Validación adicional de rendimiento
        if len(str(content)) > 10 * 1024 * 1024:  # 10MB
            raise ValueError("El contenido es demasiado grande (máximo 10MB)")
    
    async def _export_parallel(self, content: Dict[str, Any], config: ExportConfig, output_path: Optional[str] = None) -> str:
        """Exportación con procesamiento paralelo."""
        # Dividir el trabajo en tareas paralelas
        tasks = []
        
        # Tarea de validación
        if config.quality_level in [QualityLevel.PROFESSIONAL, QualityLevel.PREMIUM, QualityLevel.ENTERPRISE]:
            tasks.append(self._validate_content_async(content, config))
        
        # Tarea de mejora de calidad
        if config.quality_level in [QualityLevel.PREMIUM, QualityLevel.ENTERPRISE]:
            tasks.append(self._enhance_content_async(content, config))
        
        # Ejecutar tareas en paralelo
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Procesar resultados
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Error en tarea paralela: {result}")
        
        # Enviar tarea principal
        return await self.task_manager.submit_task(content, config, output_path)
    
    async def _validate_content_async(self, content: Dict[str, Any], config: ExportConfig) -> Dict[str, Any]:
        """Validación asíncrona de contenido."""
        return self.quality_manager.get_quality_metrics(content, config)
    
    async def _enhance_content_async(self, content: Dict[str, Any], config: ExportConfig) -> Dict[str, Any]:
        """Mejora asíncrona de contenido."""
        # Implementar mejora de contenido en paralelo
        return content
    
    def _generate_cache_key(self, content: Dict[str, Any], config: ExportConfig) -> str:
        """Generar clave de cache."""
        import hashlib
        content_str = str(sorted(content.items()))
        config_str = f"{config.format.value}_{config.document_type.value}_{config.quality_level.value}"
        combined = f"{content_str}_{config_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Obtener resultado del cache."""
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if datetime.now() - cached_data['timestamp'] < timedelta(seconds=self.cache_ttl):
                return cached_data['task_id']
            else:
                # Cache expirado
                del self.cache[cache_key]
        return None
    
    async def _save_to_cache(self, cache_key: str, task_id: str):
        """Guardar resultado en cache."""
        # Limpiar cache si está lleno
        if len(self.cache) >= self.max_cache_size:
            await self._cleanup_cache()
        
        self.cache[cache_key] = {
            'task_id': task_id,
            'timestamp': datetime.now()
        }
    
    async def _cleanup_cache(self):
        """Limpiar cache expirado."""
        now = datetime.now()
        expired_keys = [
            key for key, data in self.cache.items()
            if now - data['timestamp'] > timedelta(seconds=self.cache_ttl)
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        logger.info(f"Cache limpiado: {len(expired_keys)} entradas expiradas")
    
    async def _setup_memory_optimization(self):
        """Configurar optimización de memoria."""
        # Configurar garbage collection
        gc.set_threshold(700, 10, 10)
        
        # Configurar weak references para objetos grandes
        self._weak_refs = weakref.WeakValueDictionary()
        
        logger.info("Optimización de memoria configurada")
    
    async def _cleanup_memory(self):
        """Limpiar memoria."""
        # Forzar garbage collection
        collected = gc.collect()
        
        # Limpiar weak references
        if hasattr(self, '_weak_refs'):
            self._weak_refs.clear()
        
        logger.info(f"Memoria limpiada: {collected} objetos recolectados")
    
    async def _update_performance_metrics(self, processing_time: float, success: bool):
        """Actualizar métricas de rendimiento."""
        self.performance_metrics.total_exports += 1
        
        if success:
            self.performance_metrics.successful_exports += 1
        else:
            self.performance_metrics.failed_exports += 1
        
        # Actualizar tiempo promedio
        total_time = self.performance_metrics.average_processing_time * (self.performance_metrics.total_exports - 1)
        self.performance_metrics.average_processing_time = (total_time + processing_time) / self.performance_metrics.total_exports
        
        # Actualizar tasa de cache
        if self.enable_caching:
            cache_hits = sum(1 for data in self.cache.values() if datetime.now() - data['timestamp'] < timedelta(seconds=self.cache_ttl))
            self.performance_metrics.cache_hit_rate = cache_hits / len(self.cache) if self.cache else 0
        
        self.performance_metrics.last_updated = datetime.now()
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de rendimiento."""
        uptime = time.time() - self._start_time
        
        return {
            "uptime_seconds": uptime,
            "uptime_formatted": str(timedelta(seconds=int(uptime))),
            "total_exports": self.performance_metrics.total_exports,
            "successful_exports": self.performance_metrics.successful_exports,
            "failed_exports": self.performance_metrics.failed_exports,
            "success_rate": (self.performance_metrics.successful_exports / self.performance_metrics.total_exports * 100) if self.performance_metrics.total_exports > 0 else 0,
            "average_processing_time": self.performance_metrics.average_processing_time,
            "cache_hit_rate": self.performance_metrics.cache_hit_rate,
            "cache_size": len(self.cache),
            "memory_usage": self._get_memory_usage(),
            "last_updated": self.performance_metrics.last_updated.isoformat()
        }
    
    def _get_memory_usage(self) -> float:
        """Obtener uso de memoria."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Optimizar rendimiento del sistema."""
        optimizations = []
        
        # Limpiar cache
        if self.enable_caching:
            old_cache_size = len(self.cache)
            await self._cleanup_cache()
            new_cache_size = len(self.cache)
            optimizations.append(f"Cache limpiado: {old_cache_size} -> {new_cache_size} entradas")
        
        # Limpiar memoria
        if self.enable_memory_optimization:
            collected = gc.collect()
            optimizations.append(f"Memoria limpiada: {collected} objetos recolectados")
        
        # Optimizar pools de threads
        if hasattr(self.thread_pool, '_threads'):
            active_threads = len(self.thread_pool._threads)
            optimizations.append(f"Threads activos: {active_threads}")
        
        return {
            "optimizations_applied": optimizations,
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": await self.get_performance_metrics()
        }
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de tarea con optimizaciones."""
        return await self.task_manager.get_task_status(task_id)
    
    async def get_export_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas optimizadas."""
        stats = self.task_manager.get_statistics()
        performance = await self.get_performance_metrics()
        
        return {
            "task_statistics": {
                "total_tasks": stats.total_tasks,
                "active_tasks": stats.active_tasks,
                "completed_tasks": stats.completed_tasks,
                "failed_tasks": stats.failed_tasks,
                "average_processing_time": stats.average_processing_time
            },
            "performance_metrics": performance,
            "optimization_status": {
                "caching_enabled": self.enable_caching,
                "parallel_processing_enabled": self.enable_parallel_processing,
                "memory_optimization_enabled": self.enable_memory_optimization,
                "cache_size": len(self.cache),
                "max_cache_size": self.max_cache_size
            }
        }
    
    def list_supported_formats(self) -> List[Dict[str, Any]]:
        """Listar formatos soportados."""
        return self.exporters.list_supported_formats()
    
    def get_document_template(self, doc_type: DocumentType) -> Dict[str, Any]:
        """Obtener plantilla de documento."""
        return self.config_manager.get_template(doc_type)
    
    def validate_content(self, content: Dict[str, Any], config: ExportConfig) -> Dict[str, Any]:
        """Validar contenido y obtener métricas de calidad."""
        return self.quality_manager.get_quality_metrics(content, config)


# Instancia global del motor optimizado
_optimized_engine: Optional[OptimizedExportEngine] = None


def get_optimized_export_engine() -> OptimizedExportEngine:
    """Obtener la instancia global del motor optimizado."""
    global _optimized_engine
    if _optimized_engine is None:
        _optimized_engine = OptimizedExportEngine()
    return _optimized_engine




