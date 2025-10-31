from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from loguru import logger
import statistics
from concurrent.futures import ThreadPoolExecutor
import psutil
from ..models import SEOScrapeRequest, SEOScrapeResponse
from ..core.metrics import PerformanceTracker, MetricsCollector
            from urllib.parse import urlparse
from typing import Any, List, Dict, Optional
import logging
"""
Batch Processing Service para el servicio SEO ultra-optimizado.
Procesamiento en lote con control de concurrencia y métricas.
"""




@dataclass
class BatchResult:
    """Resultado de procesamiento en lote."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    processing_time: float
    avg_processing_time: float
    success_rate: float
    memory_usage: float
    results: List[SEOScrapeResponse]
    errors: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class BatchProcessingService:
    """Servicio de procesamiento en lote ultra-optimizado."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        self.performance_tracker = PerformanceTracker(self.config.get('performance', {}))
        self.metrics_collector = MetricsCollector()
        
        # Configuraciones por defecto
        self.default_max_concurrent = self.config.get('default_max_concurrent', 10)
        self.default_batch_size = self.config.get('default_batch_size', 100)
        self.retry_attempts = self.config.get('retry_attempts', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        
        # Estadísticas
        self.total_batches_processed = 0
        self.total_requests_processed = 0
        self.total_successful_requests = 0
        self.total_failed_requests = 0
    
    async def process_batch(
        self,
        requests: List[SEOScrapeRequest],
        seo_service,
        max_concurrent: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> BatchResult:
        """Procesa un lote de requests con control de concurrencia."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        max_concurrent = max_concurrent or self.default_max_concurrent
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Crear tasks
        tasks = []
        for i, request in enumerate(requests):
            task = self._process_single_request(
                request, seo_service, semaphore, i, progress_callback
            )
            tasks.append(task)
        
        # Ejecutar tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar resultados
        successful_results = []
        failed_results = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_info = {
                    'index': i,
                    'url': requests[i].url,
                    'error': str(result),
                    'error_type': type(result).__name__
                }
                errors.append(error_info)
                failed_results.append(SEOScrapeResponse(
                    url=requests[i].url,
                    success=False,
                    error=str(result)
                ))
            elif isinstance(result, SEOScrapeResponse):
                if result.success:
                    successful_results.append(result)
                else:
                    failed_results.append(result)
                    errors.append({
                        'index': i,
                        'url': result.url,
                        'error': result.error,
                        'error_type': 'seo_error'
                    })
        
        # Calcular métricas
        processing_time = time.perf_counter() - start_time
        end_memory = self._get_memory_usage()
        memory_usage = end_memory - start_memory
        
        total_requests = len(requests)
        successful_count = len(successful_results)
        failed_count = len(failed_results)
        success_rate = successful_count / total_requests if total_requests > 0 else 0
        
        # Actualizar estadísticas globales
        self._update_global_stats(total_requests, successful_count, failed_count)
        
        # Crear resultado
        batch_result = BatchResult(
            total_requests=total_requests,
            successful_requests=successful_count,
            failed_requests=failed_count,
            processing_time=processing_time,
            avg_processing_time=processing_time / total_requests if total_requests > 0 else 0,
            success_rate=success_rate,
            memory_usage=memory_usage,
            results=successful_results + failed_results,
            errors=errors,
            metrics=self._calculate_batch_metrics(successful_results, processing_time)
        )
        
        # Registrar métricas
        self.performance_tracker.record_metric('batch_processing_time', processing_time)
        self.performance_tracker.record_metric('batch_success_rate', success_rate)
        self.performance_tracker.record_metric('batch_memory_usage', memory_usage)
        
        logger.info(f"Batch processed: {successful_count}/{total_requests} successful "
                   f"({success_rate:.2%}) in {processing_time:.3f}s")
        
        return batch_result
    
    async async def _process_single_request(
        self,
        request: SEOScrapeRequest,
        seo_service,
        semaphore: asyncio.Semaphore,
        index: int,
        progress_callback: Optional[Callable] = None
    ) -> SEOScrapeResponse:
        """Procesa una request individual con retry y semáforo."""
        async with semaphore:
            for attempt in range(self.retry_attempts):
                try:
                    result = await seo_service.scrape(request)
                    
                    # Callback de progreso
                    if progress_callback:
                        progress_callback(index, result.success, attempt + 1)
                    
                    return result
                    
                except Exception as e:
                    if attempt == self.retry_attempts - 1:
                        # Último intento, re-lanzar excepción
                        raise e
                    else:
                        # Esperar antes del siguiente intento
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        logger.warning(f"Retry {attempt + 1} for {request.url}: {e}")
    
    def _calculate_batch_metrics(self, successful_results: List[SEOScrapeResponse], total_time: float) -> Dict[str, Any]:
        """Calcula métricas detalladas del lote."""
        if not successful_results:
            return {}
        
        # Métricas de tiempo
        processing_times = [r.metrics.processing_time for r in successful_results if r.metrics]
        load_times = [r.metrics.load_time for r in successful_results if r.metrics]
        memory_usages = [r.metrics.memory_usage for r in successful_results if r.metrics]
        
        # Métricas de SEO
        seo_scores = []
        for result in successful_results:
            if result.data and 'analysis' in result.data:
                score = result.data['analysis'].get('score', 0)
                seo_scores.append(score)
        
        # Métricas de caché
        cache_hits = sum(1 for r in successful_results if r.metrics and r.metrics.cache_hit)
        cache_hit_rate = cache_hits / len(successful_results) if successful_results else 0
        
        return {
            'timing': {
                'avg_processing_time': statistics.mean(processing_times) if processing_times else 0,
                'min_processing_time': min(processing_times) if processing_times else 0,
                'max_processing_time': max(processing_times) if processing_times else 0,
                'avg_load_time': statistics.mean(load_times) if load_times else 0,
                'total_time': total_time
            },
            'memory': {
                'avg_memory_usage': statistics.mean(memory_usages) if memory_usages else 0,
                'total_memory_usage': sum(memory_usages) if memory_usages else 0
            },
            'seo': {
                'avg_score': statistics.mean(seo_scores) if seo_scores else 0,
                'min_score': min(seo_scores) if seo_scores else 0,
                'max_score': max(seo_scores) if seo_scores else 0,
                'score_distribution': self._calculate_score_distribution(seo_scores)
            },
            'cache': {
                'hit_rate': cache_hit_rate,
                'hits': cache_hits,
                'misses': len(successful_results) - cache_hits
            }
        }
    
    def _calculate_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Calcula distribución de scores SEO."""
        if not scores:
            return {}
        
        distribution = {
            'excellent': 0,  # 90-100
            'good': 0,       # 70-89
            'fair': 0,       # 50-69
            'poor': 0        # 0-49
        }
        
        for score in scores:
            if score >= 90:
                distribution['excellent'] += 1
            elif score >= 70:
                distribution['good'] += 1
            elif score >= 50:
                distribution['fair'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution
    
    def _update_global_stats(self, total: int, successful: int, failed: int):
        """Actualiza estadísticas globales."""
        self.total_batches_processed += 1
        self.total_requests_processed += total
        self.total_successful_requests += successful
        self.total_failed_requests += failed
    
    def _get_memory_usage(self) -> float:
        """Obtiene el uso de memoria actual."""
        return psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    async def process_large_batch(
        self,
        requests: List[SEOScrapeRequest],
        seo_service,
        batch_size: Optional[int] = None,
        max_concurrent: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[BatchResult]:
        """Procesa un lote grande dividiéndolo en sub-lotes."""
        batch_size = batch_size or self.default_batch_size
        max_concurrent = max_concurrent or self.default_max_concurrent
        
        # Dividir requests en sub-lotes
        sub_batches = [
            requests[i:i + batch_size] 
            for i in range(0, len(requests), batch_size)
        ]
        
        logger.info(f"Processing large batch: {len(requests)} requests in {len(sub_batches)} sub-batches")
        
        # Procesar sub-lotes
        results = []
        for i, sub_batch in enumerate(sub_batches):
            logger.info(f"Processing sub-batch {i + 1}/{len(sub_batches)} ({len(sub_batch)} requests)")
            
            result = await self.process_batch(
                sub_batch, seo_service, max_concurrent, progress_callback
            )
            results.append(result)
            
            # Pequeña pausa entre sub-lotes para evitar sobrecarga
            if i < len(sub_batches) - 1:
                await asyncio.sleep(0.1)
        
        return results
    
    async def process_with_priority(
        self,
        requests: List[SEOScrapeRequest],
        seo_service,
        priority_function: Callable[[SEOScrapeRequest], int],
        max_concurrent: Optional[int] = None
    ) -> BatchResult:
        """Procesa requests con prioridad."""
        max_concurrent = max_concurrent or self.default_max_concurrent
        
        # Ordenar requests por prioridad
        sorted_requests = sorted(requests, key=priority_function, reverse=True)
        
        logger.info(f"Processing {len(requests)} requests with priority")
        
        return await self.process_batch(sorted_requests, seo_service, max_concurrent)
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de procesamiento en lote."""
        overall_success_rate = (
            self.total_successful_requests / self.total_requests_processed 
            if self.total_requests_processed > 0 else 0
        )
        
        return {
            'total_batches': self.total_batches_processed,
            'total_requests': self.total_requests_processed,
            'successful_requests': self.total_successful_requests,
            'failed_requests': self.total_failed_requests,
            'overall_success_rate': overall_success_rate,
            'avg_requests_per_batch': (
                self.total_requests_processed / self.total_batches_processed 
                if self.total_batches_processed > 0 else 0
            )
        }
    
    def reset_stats(self) -> Any:
        """Resetea estadísticas globales."""
        self.total_batches_processed = 0
        self.total_requests_processed = 0
        self.total_successful_requests = 0
        self.total_failed_requests = 0
        logger.info("Batch processing stats reset")
    
    async def validate_batch(self, requests: List[SEOScrapeRequest]) -> Dict[str, Any]:
        """Valida un lote de requests antes del procesamiento."""
        validation_results = {
            'valid_requests': [],
            'invalid_requests': [],
            'duplicates': [],
            'total_count': len(requests)
        }
        
        seen_urls = set()
        
        for i, request in enumerate(requests):
            # Validar URL
            if not self._is_valid_url(request.url):
                validation_results['invalid_requests'].append({
                    'index': i,
                    'url': request.url,
                    'reason': 'invalid_url'
                })
                continue
            
            # Verificar duplicados
            if request.url in seen_urls:
                validation_results['duplicates'].append({
                    'index': i,
                    'url': request.url
                })
                continue
            
            seen_urls.add(request.url)
            validation_results['valid_requests'].append(request)
        
        validation_results['valid_count'] = len(validation_results['valid_requests'])
        validation_results['invalid_count'] = len(validation_results['invalid_requests'])
        validation_results['duplicate_count'] = len(validation_results['duplicates'])
        
        return validation_results
    
    def _is_valid_url(self, url: str) -> bool:
        """Valida si una URL es válida."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    async def estimate_processing_time(
        self,
        requests: List[SEOScrapeRequest],
        max_concurrent: Optional[int] = None
    ) -> Dict[str, float]:
        """Estima el tiempo de procesamiento para un lote."""
        max_concurrent = max_concurrent or self.default_max_concurrent
        
        # Obtener métricas históricas
        avg_processing_time = self.performance_tracker.metrics.get('avg_processing_time', 2.0)
        avg_success_rate = self.performance_tracker.metrics.get('success_rate', 0.95)
        
        # Calcular estimaciones
        total_requests = len(requests)
        estimated_successful = int(total_requests * avg_success_rate)
        estimated_failed = total_requests - estimated_successful
        
        # Tiempo estimado considerando concurrencia
        estimated_time = (total_requests * avg_processing_time) / max_concurrent
        
        # Agregar tiempo para retries
        retry_time = estimated_failed * self.retry_delay * self.retry_attempts
        
        return {
            'estimated_total_time': estimated_time + retry_time,
            'estimated_processing_time': estimated_time,
            'estimated_retry_time': retry_time,
            'estimated_successful_requests': estimated_successful,
            'estimated_failed_requests': estimated_failed,
            'requests_per_second': max_concurrent / avg_processing_time
        } 