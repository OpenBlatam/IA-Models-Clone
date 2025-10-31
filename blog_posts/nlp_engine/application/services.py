from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import time
from typing import Dict, List, Any, Optional, Union
import logging
from ..core.entities import AnalysisResult
from ..core.enums import ProcessingTier, CacheStrategy, Environment
from ..interfaces.cache import ICacheRepository, ICacheKeyGenerator
from ..interfaces.metrics import IMetricsCollector, IPerformanceMonitor, IHealthChecker
from ..interfaces.config import IConfigurationService
from .dto import (
from typing import Any, List, Dict, Optional
"""
🎯 APPLICATION SERVICES - Servicios de Aplicación
================================================

Servicios que coordinan la lógica de aplicación y proporcionan
funcionalidades transversales.
"""


    CacheRequest, HealthCheckRequest, HealthCheckResponse, 
    MetricsRequest, MetricsResponse, ConfigurationRequest, ConfigurationResponse
)


class AnalysisService:
    """Servicio principal de análisis que coordina la funcionalidad."""
    
    def __init__(
        self,
        config_service: IConfigurationService,
        metrics_collector: IMetricsCollector,
        cache_repository: ICacheRepository,
        performance_monitor: IPerformanceMonitor,
        health_checker: IHealthChecker
    ):
        
    """__init__ function."""
self._config_service = config_service
        self._metrics_collector = metrics_collector
        self._cache_repository = cache_repository
        self._performance_monitor = performance_monitor
        self._health_checker = health_checker
        self._logger = logging.getLogger(self.__class__.__name__)
        
        # Estado del servicio
        self._start_time = time.time()
        self._is_healthy = True
        self._active_requests = 0
    
    async def initialize(self) -> None:
        """Inicializar el servicio."""
        try:
            # Validar configuración
            config_errors = self._config_service.validate_config()
            if config_errors:
                self._logger.error(f"Configuration errors: {config_errors}")
                self._is_healthy = False
                return
            
            # Inicializar componentes
            await self._performance_monitor.start_monitoring()
            
            # Registrar health checks
            self._health_checker.register_health_check(
                "analysis_service", 
                self._check_service_health
            )
            
            self._logger.info("AnalysisService initialized successfully")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize AnalysisService: {e}")
            self._is_healthy = False
            raise
    
    async def shutdown(self) -> None:
        """Cerrar el servicio ordenadamente."""
        try:
            await self._performance_monitor.stop_monitoring()
            self._logger.info("AnalysisService shutdown completed")
        except Exception as e:
            self._logger.error(f"Error during shutdown: {e}")
    
    def get_default_tier(self) -> ProcessingTier:
        """Obtener tier por defecto desde configuración."""
        return self._config_service.get_processing_tier()
    
    def get_cache_strategy(self) -> CacheStrategy:
        """Obtener estrategia de cache."""
        return self._config_service.get_cache_strategy()
    
    def is_optimization_enabled(self, optimization: str) -> bool:
        """Verificar si una optimización está habilitada."""
        return self._config_service.is_optimization_enabled(optimization)
    
    async async def record_request_start(self) -> None:
        """Registrar inicio de request."""
        self._active_requests += 1
        self._metrics_collector.record_gauge('active_requests', self._active_requests)
    
    async async def record_request_end(self) -> None:
        """Registrar fin de request."""
        self._active_requests = max(0, self._active_requests - 1)
        self._metrics_collector.record_gauge('active_requests', self._active_requests)
    
    async def _check_service_health(self) -> Dict[str, Any]:
        """Health check interno del servicio."""
        return {
            'status': 'healthy' if self._is_healthy else 'unhealthy',
            'uptime': time.time() - self._start_time,
            'active_requests': self._active_requests,
            'memory_usage': self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Obtener uso de memoria."""
        try:
            return self._performance_monitor.get_memory_usage()
        except Exception:
            return {'error': 'unable_to_get_memory_usage'}


class CacheService:
    """Servicio para gestión avanzada de cache."""
    
    def __init__(
        self,
        cache_repository: ICacheRepository,
        cache_key_generator: ICacheKeyGenerator,
        metrics_collector: IMetricsCollector
    ):
        
    """__init__ function."""
self._cache_repository = cache_repository
        self._cache_key_generator = cache_key_generator
        self._metrics_collector = metrics_collector
        self._logger = logging.getLogger(self.__class__.__name__)
        
        # Estadísticas
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'evictions': 0
        }
    
    async async def handle_cache_request(self, request: CacheRequest) -> Dict[str, Any]:
        """
        Manejar request de cache.
        
        Args:
            request: CacheRequest con operación solicitada
            
        Returns:
            Resultado de la operación
        """
        try:
            if request.operation == 'get':
                result = await self._get_from_cache(request.key)
                return {'success': True, 'data': result, 'cache_hit': result is not None}
            
            elif request.operation == 'set':
                await self._set_in_cache(request.key, request.value, request.ttl)
                return {'success': True, 'message': 'Value set in cache'}
            
            elif request.operation == 'delete':
                deleted = await self._delete_from_cache(request.key)
                return {'success': True, 'deleted': deleted}
            
            elif request.operation == 'clear':
                await self._clear_cache()
                return {'success': True, 'message': 'Cache cleared'}
            
            elif request.operation == 'invalidate_pattern':
                count = await self._invalidate_pattern(request.pattern)
                return {'success': True, 'invalidated_count': count}
            
            else:
                return {'success': False, 'error': f'Unknown operation: {request.operation}'}
                
        except Exception as e:
            self._cache_stats['errors'] += 1
            self._logger.error(f"Cache operation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _get_from_cache(self, key: str) -> Optional[AnalysisResult]:
        """Obtener del cache con métricas."""
        try:
            result = await self._cache_repository.get(key)
            if result:
                self._cache_stats['hits'] += 1
                self._metrics_collector.increment_counter('cache_hits')
            else:
                self._cache_stats['misses'] += 1
                self._metrics_collector.increment_counter('cache_misses')
            return result
        except Exception as e:
            self._cache_stats['errors'] += 1
            raise e
    
    async def _set_in_cache(self, key: str, value: Any, ttl: Optional[int]) -> None:
        """Guardar en cache con métricas."""
        try:
            await self._cache_repository.set(key, value, ttl)
            self._metrics_collector.increment_counter('cache_sets')
        except Exception as e:
            self._cache_stats['errors'] += 1
            raise e
    
    async def _delete_from_cache(self, key: str) -> bool:
        """Eliminar del cache."""
        try:
            deleted = await self._cache_repository.delete(key)
            if deleted:
                self._metrics_collector.increment_counter('cache_deletes')
            return deleted
        except Exception as e:
            self._cache_stats['errors'] += 1
            raise e
    
    async def _clear_cache(self) -> None:
        """Limpiar cache completo."""
        try:
            await self._cache_repository.clear()
            self._metrics_collector.increment_counter('cache_clears')
        except Exception as e:
            self._cache_stats['errors'] += 1
            raise e
    
    async def _invalidate_pattern(self, pattern: str) -> int:
        """Invalidar keys que coincidan con patrón."""
        try:
            count = await self._cache_repository.invalidate_pattern(pattern)
            self._metrics_collector.increment_counter('cache_invalidations', count)
            return count
        except Exception as e:
            self._cache_stats['errors'] += 1
            raise e
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache."""
        hit_rate = 0
        total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
        if total_requests > 0:
            hit_rate = self._cache_stats['hits'] / total_requests
        
        return {
            **self._cache_stats,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    async def optimize_cache(self) -> Dict[str, Any]:
        """Optimizar cache (limpieza, rebalanceo, etc.)."""
        try:
            # Obtener estadísticas del repositorio
            repo_stats = self._cache_repository.get_stats()
            
            # Realizar health check
            health = await self._cache_repository.health_check()
            
            optimization_performed = []
            
            # Si el cache está muy lleno, hacer limpieza
            if repo_stats.get('memory_usage', 0) > 0.8:  # 80% lleno
                await self._cache_repository.invalidate_pattern("nlp:*:old")
                optimization_performed.append("memory_cleanup")
            
            return {
                'success': True,
                'optimizations': optimization_performed,
                'health': health,
                'stats': repo_stats
            }
            
        except Exception as e:
            self._logger.error(f"Cache optimization failed: {e}")
            return {'success': False, 'error': str(e)}


class MetricsService:
    """Servicio para gestión de métricas y monitoreo."""
    
    def __init__(
        self,
        metrics_collector: IMetricsCollector,
        performance_monitor: IPerformanceMonitor,
        health_checker: IHealthChecker
    ):
        
    """__init__ function."""
self._metrics_collector = metrics_collector
        self._performance_monitor = performance_monitor
        self._health_checker = health_checker
        self._logger = logging.getLogger(self.__class__.__name__)
    
    async async def handle_metrics_request(self, request: MetricsRequest) -> MetricsResponse:
        """
        Manejar request de métricas.
        
        Args:
            request: MetricsRequest con parámetros
            
        Returns:
            MetricsResponse con datos
        """
        try:
            # Obtener métricas básicas
            metrics_summary = await self._metrics_collector.get_metrics_summary()
            
            # Filtrar métricas si se especificaron nombres
            if request.metric_names:
                filtered_metrics = {
                    name: metrics_summary.get(name)
                    for name in request.metric_names
                    if name in metrics_summary
                }
            else:
                filtered_metrics = metrics_summary
            
            # Obtener historial si se solicitó
            if request.include_history and request.metric_names:
                for metric_name in request.metric_names:
                    history = await self._metrics_collector.get_metric_history(
                        metric_name, request.time_range
                    )
                    filtered_metrics[f"{metric_name}_history"] = history
            
            return MetricsResponse(
                timestamp=time.time(),
                metrics=filtered_metrics,
                format_type=request.format_type,
                time_range=request.time_range
            )
            
        except Exception as e:
            self._logger.error(f"Metrics request failed: {e}")
            return MetricsResponse(
                timestamp=time.time(),
                metrics={'error': str(e)},
                format_type=request.format_type
            )
    
    async async def handle_health_check_request(self, request: HealthCheckRequest) -> HealthCheckResponse:
        """
        Manejar request de health check.
        
        Args:
            request: HealthCheckRequest con parámetros
            
        Returns:
            HealthCheckResponse con estado
        """
        try:
            if request.component:
                # Health check de componente específico
                component_health = await self._health_checker.check_component_health(request.component)
                
                return HealthCheckResponse(
                    status=component_health.get('status', 'unknown'),
                    timestamp=time.time(),
                    uptime_seconds=component_health.get('uptime', 0),
                    components={request.component: component_health}
                )
            else:
                # Health check completo del sistema
                system_health = await self._health_checker.check_system_health()
                
                # Obtener métricas si se solicitaron
                metrics = None
                if request.include_metrics:
                    metrics = await self._metrics_collector.get_metrics_summary()
                
                # Obtener performance si es deep check
                if request.deep_check:
                    performance_report = await self._performance_monitor.get_performance_report()
                    system_health['performance'] = performance_report
                
                return HealthCheckResponse(
                    status=system_health.get('status', 'unknown'),
                    timestamp=time.time(),
                    uptime_seconds=system_health.get('uptime', 0),
                    components=system_health.get('components', {}),
                    metrics=metrics,
                    errors=system_health.get('errors', [])
                )
                
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return HealthCheckResponse(
                status='unhealthy',
                timestamp=time.time(),
                uptime_seconds=0,
                errors=[str(e)]
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Obtener resumen de performance."""
        try:
            return {
                'cpu_usage': self._performance_monitor.get_cpu_usage(),
                'memory_usage': self._performance_monitor.get_memory_usage(),
                'disk_usage': self._performance_monitor.get_disk_usage()
            }
        except Exception as e:
            self._logger.error(f"Performance summary failed: {e}")
            return {'error': str(e)}
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generar reporte completo de performance."""
        try:
            performance_report = await self._performance_monitor.get_performance_report()
            metrics_summary = await self._metrics_collector.get_metrics_summary()
            system_health = await self._health_checker.check_system_health()
            
            return {
                'timestamp': time.time(),
                'performance': performance_report,
                'metrics': metrics_summary,
                'health': system_health,
                'summary': {
                    'overall_status': system_health.get('status', 'unknown'),
                    'total_requests': metrics_summary.get('analysis_total', 0),
                    'avg_latency': metrics_summary.get('avg_analysis_duration_ms', 0),
                    'cache_hit_rate': metrics_summary.get('cache_hit_rate', 0)
                }
            }
        except Exception as e:
            self._logger.error(f"Performance report generation failed: {e}")
            return {'error': str(e), 'timestamp': time.time()}


class ConfigurationService:
    """Servicio para gestión de configuración."""
    
    def __init__(self, config_service: IConfigurationService):
        
    """__init__ function."""
self._config_service = config_service
        self._logger = logging.getLogger(self.__class__.__name__)
    
    async async def handle_configuration_request(self, request: ConfigurationRequest) -> ConfigurationResponse:
        """
        Manejar request de configuración.
        
        Args:
            request: ConfigurationRequest con operación
            
        Returns:
            ConfigurationResponse con resultado
        """
        try:
            if request.operation == 'get':
                if request.key:
                    value = self._config_service.get_config_value(request.key)
                    return ConfigurationResponse(
                        success=True,
                        data={request.key: value}
                    )
                else:
                    all_config = self._config_service.get_all_config()
                    return ConfigurationResponse(
                        success=True,
                        data=all_config
                    )
            
            elif request.operation == 'set':
                if not request.key or request.value is None:
                    return ConfigurationResponse(
                        success=False,
                        errors=['Key and value are required for set operation']
                    )
                
                self._config_service.set_config_value(request.key, request.value)
                return ConfigurationResponse(
                    success=True,
                    data={'message': f'Configuration {request.key} updated'}
                )
            
            elif request.operation == 'validate':
                validation_errors = self._config_service.validate_config()
                return ConfigurationResponse(
                    success=len(validation_errors) == 0,
                    validation_errors=validation_errors
                )
            
            elif request.operation == 'reload':
                reloaded = self._config_service.reload_config()
                return ConfigurationResponse(
                    success=reloaded,
                    data={'reloaded': reloaded}
                )
            
            elif request.operation == 'get_all':
                all_config = self._config_service.get_all_config()
                return ConfigurationResponse(
                    success=True,
                    data=all_config
                )
            
            else:
                return ConfigurationResponse(
                    success=False,
                    errors=[f'Unknown operation: {request.operation}']
                )
                
        except Exception as e:
            self._logger.error(f"Configuration request failed: {e}")
            return ConfigurationResponse(
                success=False,
                errors=[str(e)]
            ) 