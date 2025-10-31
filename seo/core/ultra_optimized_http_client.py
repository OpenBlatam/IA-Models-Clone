from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import time
import asyncio
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from loguru import logger
import httpx
import httpcore
import anyio
from urllib.parse import urlparse, urljoin
import orjson
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import aiofiles
import aiofiles.os
from .interfaces import HTTPClientInterface
            import aiofiles.os
        import hashlib
from typing import Any, List, Dict, Optional
import logging
"""
HTTP Client ultra-optimizado usando las librerías más rápidas disponibles.
Httpx + HTTPCore + AnyIO con optimizaciones avanzadas.
"""




@dataclass
class HTTPResponse:
    """Respuesta HTTP ultra-optimizada."""
    status_code: int
    headers: Dict[str, str]
    content: bytes
    text: str
    url: str
    elapsed: float
    encoding: str = "utf-8"


@dataclass
class HTTPStats:
    """Estadísticas del cliente HTTP ultra-optimizado."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time: float = 0.0
    avg_response_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    compression_savings: float = 0.0


class UltraOptimizedHTTPClient(HTTPClientInterface):
    """Cliente HTTP ultra-optimizado con múltiples optimizaciones."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        
        # Configuraciones de conexión
        self.max_connections = self.config.get('max_connections', 200)
        self.max_keepalive_connections = self.config.get('max_keepalive_connections', 50)
        self.keepalive_expiry = self.config.get('keepalive_expiry', 30.0)
        self.timeout = self.config.get('timeout', 15.0)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        
        # Configuraciones de rendimiento
        self.enable_http2 = self.config.get('enable_http2', True)
        self.enable_compression = self.config.get('enable_compression', True)
        self.follow_redirects = self.config.get('follow_redirects', True)
        self.max_redirects = self.config.get('max_redirects', 5)
        
        # Configuraciones de cache
        self.enable_cache = self.config.get('enable_cache', True)
        self.cache_ttl = self.config.get('cache_ttl', 3600)
        self.cache_dir = self.config.get('cache_dir', '/tmp/http_cache')
        
        # Headers por defecto optimizados
        self.default_headers = {
            'User-Agent': 'SEO-Analyzer/2.0.0 (Ultra-Optimized)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br' if self.enable_compression else 'identity',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Cliente HTTP optimizado
        self.client = self._create_optimized_client()
        
        # Estadísticas
        self.stats = HTTPStats()
        self.start_time = time.time()
        
        # Cache de archivos
        if self.enable_cache:
            self._setup_cache()
    
    def _create_optimized_client(self) -> httpx.AsyncClient:
        """Crea cliente HTTP ultra-optimizado."""
        # Configurar transport optimizado
        transport = httpcore.AsyncHTTPTransport(
            retries=self.max_retries,
            http1=True,
            http2=self.enable_http2,
            verify=True,
            cert=None,
            trust_env=True
        )
        
        # Configurar límites optimizados
        limits = httpx.Limits(
            max_connections=self.max_connections,
            max_keepalive_connections=self.max_keepalive_connections,
            keepalive_expiry=self.keepalive_expiry
        )
        
        # Crear cliente
        return httpx.AsyncClient(
            transport=transport,
            limits=limits,
            timeout=self.timeout,
            follow_redirects=self.follow_redirects,
            max_redirects=self.max_redirects,
            headers=self.default_headers,
            http2=self.enable_http2
        )
    
    def _setup_cache(self) -> Any:
        """Configura cache de archivos."""
        try:
            asyncio.create_task(aiofiles.os.makedirs(self.cache_dir, exist_ok=True))
            logger.info(f"HTTP cache directory created: {self.cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to setup HTTP cache: {e}")
            self.enable_cache = False
    
    def _get_cache_key(self, url: str, headers: Optional[Dict[str, str]] = None) -> str:
        """Genera clave de cache para URL."""
        cache_data = f"{url}:{orjson.dumps(headers or {}).decode()}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[HTTPResponse]:
        """Obtiene respuesta del cache."""
        if not self.enable_cache:
            return None
        
        try:
            cache_file = f"{self.cache_dir}/{cache_key}"
            async with aiofiles.open(cache_file, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                cached_data = await f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                if cached_data:
                    response_data = orjson.loads(cached_data)
                    self.stats.cache_hits += 1
                    return HTTPResponse(**response_data)
        except Exception as e:
            logger.debug(f"Cache miss for {cache_key}: {e}")
        
        self.stats.cache_misses += 1
        return None
    
    async def _save_to_cache(self, cache_key: str, response: HTTPResponse):
        """Guarda respuesta en cache."""
        if not self.enable_cache:
            return
        
        try:
            cache_file = f"{self.cache_dir}/{cache_key}"
            response_data = {
                'status_code': response.status_code,
                'headers': response.headers,
                'content': response.content,
                'text': response.text,
                'url': response.url,
                'elapsed': response.elapsed,
                'encoding': response.encoding
            }
            
            async with aiofiles.open(cache_file, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                await f.write(orjson.dumps(response_data))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        except Exception as e:
            logger.debug(f"Failed to save to cache: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException))
    )
    async def get(self, url: str, headers: Optional[Dict[str, str]] = None, 
                  timeout: Optional[float] = None) -> HTTPResponse:
        """Realiza GET request ultra-optimizado."""
        start_time = time.perf_counter()
        self.stats.total_requests += 1
        
        # Verificar cache
        cache_key = self._get_cache_key(url, headers)
        cached_response = await self._get_from_cache(cache_key)
        if cached_response:
            return cached_response
        
        # Headers combinados
        request_headers = {**self.default_headers, **(headers or {})}
        
        try:
            # Realizar request
            response = await self.client.get(
                url,
                headers=request_headers,
                timeout=timeout or self.timeout
            )
            
            # Procesar respuesta
            elapsed = time.perf_counter() - start_time
            http_response = HTTPResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                content=response.content,
                text=response.text,
                url=str(response.url),
                elapsed=elapsed,
                encoding=response.encoding or "utf-8"
            )
            
            # Guardar en cache si es exitosa
            if response.status_code == 200:
                await self._save_to_cache(cache_key, http_response)
            
            self.stats.successful_requests += 1
            self.stats.total_time += elapsed
            self.stats.avg_response_time = self.stats.total_time / self.stats.successful_requests
            
            return http_response
            
        except Exception as e:
            self.stats.failed_requests += 1
            logger.error(f"HTTP GET failed for {url}: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException))
    )
    async def post(self, url: str, data: Any = None, json: Any = None,
                   headers: Optional[Dict[str, str]] = None,
                   timeout: Optional[float] = None) -> HTTPResponse:
        """Realiza POST request ultra-optimizado."""
        start_time = time.perf_counter()
        self.stats.total_requests += 1
        
        # Headers combinados
        request_headers = {**self.default_headers, **(headers or {})}
        
        try:
            # Realizar request
            response = await self.client.post(
                url,
                data=data,
                json=json,
                headers=request_headers,
                timeout=timeout or self.timeout
            )
            
            # Procesar respuesta
            elapsed = time.perf_counter() - start_time
            http_response = HTTPResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                content=response.content,
                text=response.text,
                url=str(response.url),
                elapsed=elapsed,
                encoding=response.encoding or "utf-8"
            )
            
            self.stats.successful_requests += 1
            self.stats.total_time += elapsed
            self.stats.avg_response_time = self.stats.total_time / self.stats.successful_requests
            
            return http_response
            
        except Exception as e:
            self.stats.failed_requests += 1
            logger.error(f"HTTP POST failed for {url}: {e}")
            raise
    
    async def get_many(self, urls: List[str], 
                      headers: Optional[Dict[str, str]] = None,
                      max_concurrent: Optional[int] = None) -> List[HTTPResponse]:
        """Realiza múltiples GET requests en paralelo."""
        max_concurrent = max_concurrent or min(len(urls), self.max_connections)
        
        # Crear semáforo para limitar concurrencia
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def get_with_semaphore(url: str) -> HTTPResponse:
            async with semaphore:
                return await self.get(url, headers)
        
        # Ejecutar requests en paralelo
        tasks = [get_with_semaphore(url) for url in urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar resultados
        successful_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Failed to fetch {urls[i]}: {response}")
            else:
                successful_responses.append(response)
        
        return successful_responses
    
    async def head(self, url: str, headers: Optional[Dict[str, str]] = None,
                   timeout: Optional[float] = None) -> HTTPResponse:
        """Realiza HEAD request ultra-optimizado."""
        start_time = time.perf_counter()
        self.stats.total_requests += 1
        
        # Headers combinados
        request_headers = {**self.default_headers, **(headers or {})}
        
        try:
            # Realizar request
            response = await self.client.head(
                url,
                headers=request_headers,
                timeout=timeout or self.timeout
            )
            
            # Procesar respuesta
            elapsed = time.perf_counter() - start_time
            http_response = HTTPResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                content=b'',
                text='',
                url=str(response.url),
                elapsed=elapsed
            )
            
            self.stats.successful_requests += 1
            self.stats.total_time += elapsed
            self.stats.avg_response_time = self.stats.total_time / self.stats.successful_requests
            
            return http_response
            
        except Exception as e:
            self.stats.failed_requests += 1
            logger.error(f"HTTP HEAD failed for {url}: {e}")
            raise
    
    async def check_url_health(self, url: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Verifica la salud de una URL."""
        try:
            start_time = time.perf_counter()
            
            # Realizar HEAD request
            response = await self.head(url, timeout=timeout)
            
            elapsed = time.perf_counter() - start_time
            
            return {
                'url': url,
                'status': 'healthy',
                'status_code': response.status_code,
                'response_time': elapsed,
                'headers': response.headers
            }
            
        except Exception as e:
            return {
                'url': url,
                'status': 'unhealthy',
                'error': str(e),
                'response_time': None
            }
    
    async async def download_file(self, url: str, filepath: str,
                           chunk_size: int = 8192) -> bool:
        """Descarga archivo ultra-optimizado."""
        try:
            async with self.client.stream('GET', url) as response:
                if response.status_code == 200:
                    async with aiofiles.open(filepath, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                            await f.write(chunk)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    return True
                else:
                    logger.error(f"Failed to download {url}: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Download failed for {url}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas del cliente HTTP."""
        total_requests = self.stats.total_requests
        success_rate = (self.stats.successful_requests / total_requests * 100) if total_requests > 0 else 0
        cache_hit_rate = (self.stats.cache_hits / (self.stats.cache_hits + self.stats.cache_misses) * 100) if (self.stats.cache_hits + self.stats.cache_misses) > 0 else 0
        
        return {
            'total_requests': total_requests,
            'successful_requests': self.stats.successful_requests,
            'failed_requests': self.stats.failed_requests,
            'success_rate': success_rate,
            'avg_response_time': self.stats.avg_response_time,
            'total_time': self.stats.total_time,
            'cache_hits': self.stats.cache_hits,
            'cache_misses': self.stats.cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'compression_savings': self.stats.compression_savings,
            'uptime': time.time() - self.start_time,
            'max_connections': self.max_connections,
            'enable_http2': self.enable_http2,
            'enable_compression': self.enable_compression,
            'enable_cache': self.enable_cache
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verifica la salud del cliente HTTP."""
        health = {
            'status': 'healthy',
            'client': 'ok',
            'cache': 'ok',
            'connection_pool': 'ok'
        }
        
        try:
            # Verificar cliente
            test_url = 'https://httpbin.org/get'
            response = await self.get(test_url, timeout=5.0)
            if response.status_code != 200:
                health['client'] = 'error'
                health['status'] = 'degraded'
        except Exception as e:
            health['client'] = f'error: {e}'
            health['status'] = 'degraded'
        
        # Verificar cache
        if self.enable_cache:
            try:
                cache_key = self._get_cache_key('test')
                await self._save_to_cache(cache_key, HTTPResponse(
                    status_code=200,
                    headers={},
                    content=b'test',
                    text='test',
                    url='test',
                    elapsed=0.0
                ))
                cached = await self._get_from_cache(cache_key)
                if not cached:
                    health['cache'] = 'error'
                    health['status'] = 'degraded'
            except Exception as e:
                health['cache'] = f'error: {e}'
                health['status'] = 'degraded'
        
        return health
    
    async def close(self) -> Any:
        """Cierra el cliente HTTP."""
        if self.client:
            await self.client.aclose()
            logger.info("HTTP client closed")
    
    async def __aenter__(self) -> Any:
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Context manager exit."""
        await self.close() 