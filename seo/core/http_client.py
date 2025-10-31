from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import time
import asyncio
from typing import Optional, Dict, Any
from loguru import logger
import httpx
import cchardet
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio_throttle
from .interfaces import HTTPClient
            import psutil
from typing import Any, List, Dict, Optional
import logging
"""
HTTP Client ultra-optimizado para el servicio SEO.
Implementación con connection pooling, throttling y retry inteligente.
"""




class UltraFastHTTPClient(HTTPClient):
    """Cliente HTTP ultra-optimizado con connection pooling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.session = None
        self.config = config or {}
        self.throttler = asyncio_throttle.Throttler(
            rate_limit=self.config.get('rate_limit', 100),
            period=self.config.get('period', 60)
        )
        self._setup_session()
    
    def _setup_session(self) -> Any:
        """Configura sesión HTTP ultra-optimizada."""
        limits = httpx.Limits(
            max_keepalive_connections=self.config.get('max_keepalive', 20),
            max_connections=self.config.get('max_connections', 100)
        )
        timeout = httpx.Timeout(
            self.config.get('timeout', 10.0),
            connect=self.config.get('connect_timeout', 5.0)
        )
        
        headers = {
            'User-Agent': self.config.get('user_agent', 
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        }
        
        self.session = httpx.AsyncClient(
            limits=limits,
            timeout=timeout,
            headers=headers,
            follow_redirects=self.config.get('follow_redirects', True),
            http2=self.config.get('http2', True)
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async async def fetch(self, url: str) -> Optional[str]:
        """Obtiene contenido HTML con throttling y retry."""
        async with self.throttler:
            try:
                response = await self.session.get(url)
                response.raise_for_status()
                
                # Detectar encoding automáticamente
                encoding = cchardet.detect(response.content)['encoding'] or 'utf-8'
                return response.content.decode(encoding, errors='ignore')
                
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def measure_load_time(self, url: str) -> Optional[float]:
        """Mide tiempo de carga ultra-optimizado."""
        start_time = time.perf_counter()
        try:
            response = await self.session.get(url)
            response.raise_for_status()
            return time.perf_counter() - start_time
        except Exception as e:
            logger.error(f"Error measuring load time for {url}: {e}")
            return None
    
    async async def fetch_with_metrics(self, url: str) -> Dict[str, Any]:
        """Obtiene contenido HTML con métricas detalladas."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        try:
            content = await self.fetch(url)
            load_time = time.perf_counter() - start_time
            end_memory = self._get_memory_usage()
            
            return {
                'content': content,
                'load_time': load_time,
                'memory_delta': end_memory - start_memory,
                'success': True
            }
        except Exception as e:
            return {
                'content': None,
                'load_time': time.perf_counter() - start_time,
                'memory_delta': 0,
                'success': False,
                'error': str(e)
            }
    
    async async def batch_fetch(self, urls: list, max_concurrent: int = 10) -> Dict[str, Any]:
        """Obtiene múltiples URLs en paralelo."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async async def fetch_single(url: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.fetch_with_metrics(url)
        
        tasks = [fetch_single(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar resultados
        successful = []
        failed = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed.append({'url': urls[i], 'error': str(result)})
            elif result.get('success'):
                successful.append(result)
            else:
                failed.append({'url': urls[i], 'error': result.get('error', 'Unknown error')})
        
        return {
            'successful': successful,
            'failed': failed,
            'total': len(urls),
            'success_rate': len(successful) / len(urls) if urls else 0
        }
    
    def _get_memory_usage(self) -> float:
        """Obtiene el uso de memoria actual."""
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0
    
    async def close(self) -> Any:
        """Cierra la sesión HTTP."""
        if self.session:
            await self.session.aclose()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cliente HTTP."""
        return {
            'session_active': self.session is not None,
            'throttle_rate': self.throttler.rate_limit,
            'throttle_period': self.throttler.period
        }


class HTTPClientFactory:
    """Factory para crear clientes HTTP."""
    
    @staticmethod
    def create_client(client_type: str = "ultra_fast", config: Optional[Dict[str, Any]] = None) -> HTTPClient:
        """Crea un cliente HTTP basado en el tipo especificado."""
        if client_type == "ultra_fast":
            return UltraFastHTTPClient(config)
        else:
            raise ValueError(f"Unknown HTTP client type: {client_type}") 