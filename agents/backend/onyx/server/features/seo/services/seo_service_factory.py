from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, Any, Optional
from loguru import logger
from ..core.interfaces import HTMLParserInterface, CacheInterface, HTTPClientInterface, AnalyzerInterface
from ..core.ultra_optimized_parser import UltraOptimizedParser
from ..core.ultra_optimized_cache import UltraOptimizedCache
from ..core.ultra_optimized_http_client import UltraOptimizedHTTPClient
from ..core.ultra_optimized_analyzer import UltraOptimizedAnalyzer
from ..services.selenium_service import SeleniumService
from ..services.batch_service import BatchProcessingService
from ..services.seo_service import UltraOptimizedSEOService
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Factory for SEO Service with dependency injection.
Creates and configures all dependencies for the SEO service.
"""




class SEOServiceFactory:
    """Factory for creating SEO service with all dependencies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        self._dependencies = {}
        self._initialized = False
    
    def create_parser(self) -> HTMLParserInterface:
        """Create HTML parser with configuration."""
        parser_config = self.config.get('parser', {})
        parser = UltraOptimizedParser(parser_config)
        logger.info("HTML Parser created")
        return parser
    
    def create_cache(self) -> CacheInterface:
        """Create cache manager with configuration."""
        cache_config = self.config.get('cache', {})
        cache = UltraOptimizedCache(cache_config)
        logger.info("Cache Manager created")
        return cache
    
    async def create_http_client(self) -> HTTPClientInterface:
        """Create HTTP client with configuration."""
        http_config = self.config.get('http_client', {})
        http_client = UltraOptimizedHTTPClient(http_config)
        logger.info("HTTP Client created")
        return http_client
    
    def create_analyzer(self) -> AnalyzerInterface:
        """Create analyzer with configuration."""
        analyzer_config = self.config.get('analyzer', {})
        analyzer = UltraOptimizedAnalyzer(analyzer_config)
        logger.info("Analyzer created")
        return analyzer
    
    def create_selenium_service(self) -> Any:
        """Create Selenium service with configuration."""
        selenium_config = self.config.get('selenium', {})
        selenium_service = SeleniumService(selenium_config)
        logger.info("Selenium Service created")
        return selenium_service
    
    def create_batch_service(self) -> Any:
        """Create batch processing service with configuration."""
        batch_config = self.config.get('batch_service', {})
        batch_service = BatchProcessingService(batch_config)
        logger.info("Batch Processing Service created")
        return batch_service
    
    def create_seo_service(self) -> UltraOptimizedSEOService:
        """Create main SEO service with all dependencies."""
        # Create all dependencies
        parser = self.create_parser()
        cache = self.create_cache()
        http_client = self.create_http_client()
        analyzer = self.create_analyzer()
        selenium_service = self.create_selenium_service()
        batch_service = self.create_batch_service()
        
        # Create main service
        seo_config = self.config.get('seo_service', {})
        seo_service = UltraOptimizedSEOService(
            parser=parser,
            cache=cache,
            http_client=http_client,
            analyzer=analyzer,
            selenium_service=selenium_service,
            batch_service=batch_service,
            config=seo_config
        )
        
        logger.info("SEO Service created with all dependencies")
        return seo_service
    
    def create_with_dependencies(self) -> Dict[str, Any]:
        """Create all services with dependencies."""
        if self._initialized:
            return self._dependencies
        
        # Create all components
        self._dependencies = {
            'parser': self.create_parser(),
            'cache': self.create_cache(),
            'http_client': self.create_http_client(),
            'analyzer': self.create_analyzer(),
            'selenium_service': self.create_selenium_service(),
            'batch_service': self.create_batch_service(),
            'seo_service': self.create_seo_service()
        }
        
        self._initialized = True
        logger.info("All SEO service dependencies created")
        return self._dependencies
    
    def get_dependency(self, name: str):
        """Get a specific dependency by name."""
        if not self._initialized:
            self.create_with_dependencies()
        
        if name not in self._dependencies:
            raise ValueError(f"Dependency '{name}' not found")
        
        return self._dependencies[name]
    
    def get_seo_service(self) -> UltraOptimizedSEOService:
        """Get the main SEO service."""
        return self.get_dependency('seo_service')
    
    def get_cache(self) -> CacheInterface:
        """Get the cache manager."""
        return self.get_dependency('cache')
    
    async def get_http_client(self) -> HTTPClientInterface:
        """Get the HTTP client."""
        return self.get_dependency('http_client')
    
    def get_parser(self) -> HTMLParserInterface:
        """Get the HTML parser."""
        return self.get_dependency('parser')
    
    def get_analyzer(self) -> AnalyzerInterface:
        """Get the analyzer."""
        return self.get_dependency('analyzer')
    
    def get_batch_service(self) -> Optional[Dict[str, Any]]:
        """Get the batch processing service."""
        return self.get_dependency('batch_service')
    
    def get_selenium_service(self) -> Optional[Dict[str, Any]]:
        """Get the Selenium service."""
        return self.get_dependency('selenium_service')
    
    async def cleanup(self) -> Any:
        """Cleanup all dependencies."""
        if not self._initialized:
            return
        
        # Close HTTP client
        if 'http_client' in self._dependencies:
            await self._dependencies['http_client'].close()
        
        # Close Selenium service
        if 'selenium_service' in self._dependencies:
            self._dependencies['selenium_service'].close()
        
        # Close SEO service
        if 'seo_service' in self._dependencies:
            await self._dependencies['seo_service'].close()
        
        logger.info("All SEO service dependencies cleaned up")
        self._initialized = False
        self._dependencies = {}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all dependencies."""
        if not self._initialized:
            return {'status': 'not_initialized'}
        
        health_status = {
            'status': 'healthy',
            'dependencies': {}
        }
        
        # Check each dependency
        for name, dependency in self._dependencies.items():
            try:
                if hasattr(dependency, 'health_check'):
                    health = dependency.health_check()
                    if asyncio.iscoroutine(health):
                        # Handle async health checks
                        health_status['dependencies'][name] = {'status': 'async_check_needed'}
                    else:
                        health_status['dependencies'][name] = health
                else:
                    health_status['dependencies'][name] = {'status': 'no_health_check'}
            except Exception as e:
                health_status['dependencies'][name] = {'status': 'error', 'error': str(e)}
                health_status['status'] = 'degraded'
        
        return health_status
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics of all dependencies."""
        if not self._initialized:
            return {'status': 'not_initialized'}
        
        stats = {
            'factory': {
                'initialized': self._initialized,
                'dependencies_count': len(self._dependencies)
            },
            'dependencies': {}
        }
        
        # Get stats from each dependency
        for name, dependency in self._dependencies.items():
            try:
                if hasattr(dependency, 'get_stats'):
                    stats['dependencies'][name] = dependency.get_stats()
                elif hasattr(dependency, 'get_performance_stats'):
                    stats['dependencies'][name] = dependency.get_performance_stats()
                else:
                    stats['dependencies'][name] = {'status': 'no_stats_available'}
            except Exception as e:
                stats['dependencies'][name] = {'status': 'error', 'error': str(e)}
        
        return stats


# Global factory instance
_global_factory: Optional[SEOServiceFactory] = None


def get_factory(config: Optional[Dict[str, Any]] = None) -> SEOServiceFactory:
    """Get global factory instance."""
    global _global_factory
    if _global_factory is None:
        _global_factory = SEOServiceFactory(config)
    return _global_factory


def get_seo_service(config: Optional[Dict[str, Any]] = None) -> UltraOptimizedSEOService:
    """Get SEO service instance."""
    factory = get_factory(config)
    return factory.get_seo_service()


async def cleanup_factory():
    """Cleanup global factory."""
    global _global_factory
    if _global_factory:
        await _global_factory.cleanup()
        _global_factory = None 