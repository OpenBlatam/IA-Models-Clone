from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import time
import asyncio
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin, urlparse
from loguru import logger
import psutil
from ..core.interfaces import HTMLParser, HTTPClient, CacheManager, SEOAnalyzer, PerformanceTracker
from ..core.parsers import ParserFactory
from ..core.http_client import HTTPClientFactory
from ..core.cache_manager import CacheManagerFactory
from ..core.analyzer import AnalyzerFactory
from ..core.metrics import PerformanceTracker, MetricsCollector
from ..models import SEOScrapeRequest, SEOScrapeResponse, SEOMetrics
            from .selenium_service import SeleniumService
        import hashlib
        import orjson
from typing import Any, List, Dict, Optional
import logging
"""
Servicio SEO principal ultra-optimizado con arquitectura refactorizada.
Implementación modular con inyección de dependencias.
"""




class UltraOptimizedSEOService:
    """Servicio SEO ultra-optimizado con arquitectura modular."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        self.performance_tracker = PerformanceTracker(self.config.get('performance', {}))
        self.metrics_collector = MetricsCollector()
        
        # Inicializar componentes usando factories
        self._initialize_components()
        
        # Configurar Selenium si está habilitado
        self.selenium_service = None
        if self.config.get('enable_selenium', False):
            self._setup_selenium()
    
    def _initialize_components(self) -> Any:
        """Inicializa todos los componentes del servicio."""
        # Parser
        parser_config = self.config.get('parser', {})
        self.parser = ParserFactory.create_parser(
            parser_config.get('type', 'auto'),
            parser_config
        )
        
        # HTTP Client
        http_config = self.config.get('http_client', {})
        self.http_client = HTTPClientFactory.create_client(
            http_config.get('type', 'ultra_fast'),
            http_config
        )
        
        # Cache Manager
        cache_config = self.config.get('cache', {})
        self.cache_manager = CacheManagerFactory.create_cache_manager(
            cache_config.get('type', 'ultra_optimized'),
            cache_config
        )
        
        # SEO Analyzer
        analyzer_config = self.config.get('analyzer', {})
        self.analyzer = AnalyzerFactory.create_analyzer(
            analyzer_config.get('type', 'ultra_fast'),
            analyzer_config
        )
        
        logger.info(f"SEO Service initialized with:")
        logger.info(f"  Parser: {self.parser.get_parser_name()}")
        logger.info(f"  HTTP Client: {type(self.http_client).__name__}")
        logger.info(f"  Cache Manager: {type(self.cache_manager).__name__}")
        logger.info(f"  Analyzer: {self.analyzer.get_analyzer_name()}")
    
    def _setup_selenium(self) -> Any:
        """Configura el servicio Selenium."""
        try:
            self.selenium_service = SeleniumService(self.config.get('selenium', {}))
            logger.info("Selenium service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Selenium service: {e}")
    
    async def scrape(self, request: SEOScrapeRequest) -> SEOScrapeResponse:
        """Scraping SEO ultra-optimizado con métricas detalladas."""
        self.performance_tracker.start_timer('total_request')
        start_memory = self._get_memory_usage()
        
        try:
            # Normalizar URL
            normalized_url = self._normalize_url(request.url)
            
            # Verificar caché
            cache_key = self._generate_cache_key(normalized_url, request.options)
            cached_result = self.cache_manager.get(cache_key)
            
            if cached_result and not request.force_refresh:
                self.performance_tracker.end_timer('total_request')
                return self._create_cached_response(normalized_url, cached_result, start_memory)
            
            # Realizar análisis
            result = await self._perform_analysis(normalized_url, request.options)
            
            # Guardar en caché
            self.cache_manager.set(cache_key, result)
            
            # Crear respuesta
            response = self._create_response(normalized_url, result, start_memory, cache_hit=False)
            
            # Registrar métricas
            self.performance_tracker.record_request(
                success=True,
                processing_time=response.metrics.processing_time,
                memory_usage=response.metrics.memory_usage
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in SEO scraping: {e}")
            self.performance_tracker.end_timer('total_request')
            
            return self._create_error_response(request.url, str(e), start_memory)
    
    async def _perform_analysis(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza análisis SEO completo."""
        self.performance_tracker.start_timer('analysis')
        
        # Obtener contenido HTML
        html_content = await self._get_html_content(url, options)
        if not html_content:
            raise Exception("No se pudo obtener contenido HTML")
        
        # Parsear HTML
        self.performance_tracker.start_timer('parsing')
        seo_data = self.parser.parse(html_content, url)
        self.performance_tracker.end_timer('parsing')
        
        # Analizar con LangChain
        self.performance_tracker.start_timer('ai_analysis')
        analysis = await self.analyzer.analyze(seo_data, url)
        self.performance_tracker.end_timer('ai_analysis')
        
        # Combinar resultados
        result = {
            **seo_data,
            "analysis": analysis,
            "url": url,
            "timestamp": time.time(),
            "parser_used": self.parser.get_parser_name(),
            "analyzer_used": self.analyzer.get_analyzer_name()
        }
        
        self.performance_tracker.end_timer('analysis')
        return result
    
    async def _get_html_content(self, url: str, options: Dict[str, Any]) -> Optional[str]:
        """Obtiene contenido HTML usando el método más apropiado."""
        use_selenium = options.get("use_selenium", False)
        
        if use_selenium and self.selenium_service:
            try:
                return self.selenium_service.get_page_source(url)
            except Exception as e:
                logger.error(f"Selenium error: {e}")
        
        # Usar HTTP client como fallback
        return await self.http_client.fetch(url)
    
    def _normalize_url(self, url: str) -> str:
        """Normaliza URL ultra-rápido."""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        return url.rstrip('/')
    
    def _generate_cache_key(self, url: str, options: Dict[str, Any]) -> str:
        """Genera clave de caché única."""
        
        # Crear string único basado en URL y opciones
        key_data = {
            'url': url,
            'options': {k: v for k, v in options.items() if k != 'force_refresh'}
        }
        
        key_string = orjson.dumps(key_data, sort_keys=True).decode()
        return f"seo_analysis:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _get_memory_usage(self) -> float:
        """Obtiene el uso de memoria actual."""
        return psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    def _create_response(self, url: str, result: Dict[str, Any], start_memory: float, cache_hit: bool = False) -> SEOScrapeResponse:
        """Crea respuesta de scraping."""
        processing_time = self.performance_tracker.end_timer('total_request')
        end_memory = self._get_memory_usage()
        
        metrics = SEOMetrics(
            load_time=result.get("load_time", 0.0),
            memory_usage=end_memory - start_memory,
            cache_hit=cache_hit,
            processing_time=processing_time,
            elements_extracted=len(result.get("images", [])) + len(result.get("links", [])),
            compression_ratio=self.cache_manager.get_stats().get("compression_ratio", 0.0),
            network_latency=result.get("network_latency", 0.0)
        )
        
        return SEOScrapeResponse(
            url=url,
            success=True,
            data=result,
            metrics=metrics
        )
    
    def _create_cached_response(self, url: str, cached_result: Dict[str, Any], start_memory: float) -> SEOScrapeResponse:
        """Crea respuesta desde caché."""
        processing_time = self.performance_tracker.end_timer('total_request')
        end_memory = self._get_memory_usage()
        
        metrics = SEOMetrics(
            load_time=0.0,
            memory_usage=end_memory - start_memory,
            cache_hit=True,
            processing_time=processing_time,
            elements_extracted=len(cached_result.get("images", [])) + len(cached_result.get("links", [])),
            compression_ratio=self.cache_manager.get_stats().get("compression_ratio", 0.0),
            network_latency=0.0
        )
        
        return SEOScrapeResponse(
            url=url,
            success=True,
            data=cached_result,
            metrics=metrics
        )
    
    def _create_error_response(self, url: str, error: str, start_memory: float) -> SEOScrapeResponse:
        """Crea respuesta de error."""
        processing_time = self.performance_tracker.end_timer('total_request')
        end_memory = self._get_memory_usage()
        
        metrics = SEOMetrics(
            load_time=0.0,
            memory_usage=end_memory - start_memory,
            cache_hit=False,
            processing_time=processing_time,
            elements_extracted=0,
            compression_ratio=0.0,
            network_latency=0.0
        )
        
        return SEOScrapeResponse(
            url=url,
            success=False,
            error=error,
            metrics=metrics
        )
    
    async def batch_scrape(self, requests: List[SEOScrapeRequest], max_concurrent: int = 10) -> List[SEOScrapeResponse]:
        """Realiza scraping en lote ultra-optimizado."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(request: SEOScrapeRequest) -> SEOScrapeResponse:
            async with semaphore:
                return await self.scrape(request)
        
        tasks = [process_single(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar resultados
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(SEOScrapeResponse(
                    url=requests[i].url,
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché."""
        return self.cache_manager.get_stats()
    
    def clear_cache(self) -> int:
        """Limpia el caché."""
        return self.cache_manager.clear()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de rendimiento."""
        return self.performance_tracker.get_performance_summary()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del sistema."""
        return self.metrics_collector.collect_system_metrics()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del servicio."""
        return {
            'performance': self.get_performance_stats(),
            'cache': self.get_cache_stats(),
            'system': self.get_system_metrics(),
            'components': {
                'parser': self.parser.get_parser_name(),
                'http_client': type(self.http_client).__name__,
                'cache_manager': type(self.cache_manager).__name__,
                'analyzer': self.analyzer.get_analyzer_name(),
                'selenium_enabled': self.selenium_service is not None
            }
        }
    
    async def close(self) -> Any:
        """Cierra recursos del servicio."""
        await self.http_client.close()
        if self.selenium_service:
            self.selenium_service.close()
        
        logger.info("SEO Service resources closed")


# Instancia global del servicio
seo_service = UltraOptimizedSEOService()


async def scrape(request: SEOScrapeRequest) -> SEOScrapeResponse:
    """Función de scraping ultra-optimizada."""
    return await seo_service.scrape(request) 