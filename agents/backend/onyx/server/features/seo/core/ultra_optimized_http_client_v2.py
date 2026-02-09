from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import time
import asyncio
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from loguru import logger
import httpx
import httpcore
import orjson
import zstandard as zstd
from urllib.parse import urlparse, urljoin
import ssl
from contextlib import asynccontextmanager
from .interfaces import HTTPClientInterface
        import hashlib
from typing import Any, List, Dict, Optional
import logging
"""
Ultra-Optimized HTTP Client v2.0
Using the fastest HTTP libraries: httpx + httpcore + h2
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
    average_response_time: float = 0.0
    total_bytes_downloaded: int = 0
    total_bytes_uploaded: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    compression_savings: float = 0.0


class UltraOptimizedHTTPClientV2(HTTPClientInterface):
    """Cliente HTTP ultra-optimizado con las librerías más rápidas."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        
        # Configuraciones del cliente
        self.timeout = self.config.get('timeout', 15.0)
        self.max_connections = self.config.get('max_connections', 200)
        self.max_keepalive_connections = self.config.get('max_keepalive_connections', 20)
        self.keepalive_expiry = self.config.get('keepalive_expiry', 30.0)
        self.enable_http2 = self.config.get('enable_http2', True)
        self.enable_compression = self.config.get('enable_compression', True)
        self.retry_attempts = self.config.get('retry_attempts', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        self.max_redirects = self.config.get('max_redirects', 5)
        self.verify_ssl = self.config.get('verify_ssl', True)
        
        # Configuraciones de rate limiting
        self.rate_limit = self.config.get('rate_limit', 200)  # requests per second
        self.rate_limit_window = self.config.get('rate_limit_window', 1.0)
        
        # Configuraciones de cache
        self.enable_cache = self.config.get('enable_cache', True)
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        
        # Headers por defecto
        self.default_headers = {
            'User-Agent': 'SEO-Service/2.0 (Ultra-Optimized)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Actualizar headers por defecto con configuración
        if self.config.get('headers'):
            self.default_headers.update(self.config['headers'])
        
        # Inicializar cliente HTTP
        self._init_http_client()
        
        # Compresor Zstandard para datos grandes
        if self.enable_compression:
            self.compressor = zstd.ZstdCompressor(level=3)
            self.decompressor = zstd.ZstdDecompressor()
        
        # Cache de respuestas
        self.response_cache = {}
        self.cache_size = self.config.get('cache_size', 1000)
        
        # Rate limiting
        self.request_times = []
        self.rate_limit_semaphore = asyncio.Semaphore(self.rate_limit)
        
        # Estadísticas
        self.stats = HTTPStats()
        
        logger.info("Ultra-Optimized HTTP Client v2.0 initialized")
    
    async def _init_http_client(self) -> Any:
        """Inicializar cliente HTTP ultra-optimizado."""
        # Configurar transport HTTP
        transport_config = {
            'retries': self.retry_attempts,
            'http1': True,
            'http2': self.enable_http2,
            'verify': self.verify_ssl,
            'max_connections': self.max_connections,
            'max_keepalive_connections': self.max_keepalive_connections,
            'keepalive_expiry': self.keepalive_expiry,
        }
        
        # Configurar límites
        limits = httpx.Limits(
            max_connections=self.max_connections,
            max_keepalive_connections=self.max_keepalive_connections,
            keepalive_expiry=self.keepalive_expiry
        )
        
        # Crear cliente
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=limits,
            http2=self.enable_http2,
            verify=self.verify_ssl,
            follow_redirects=True,
            max_redirects=self.max_redirects,
            default_headers=self.default_headers,
            transport=httpcore.AsyncHTTPTransport(**transport_config)
        )
        
        logger.info(f"HTTP client initialized: http2={self.enable_http2}, max_connections={self.max_connections}")
    
    async def get(self, url: str, headers: Optional[Dict[str, str]] = None, 
                  timeout: Optional[float] = None) -> HTTPResponse:
        """GET request ultra-optimizado."""
        return await self._make_request('GET', url, headers=headers, timeout=timeout)
    
    async def post(self, url: str, data: Any = None, json: Any = None,
                   headers: Optional[Dict[str, str]] = None,
                   timeout: Optional[float] = None) -> HTTPResponse:
        """POST request ultra-optimizado."""
        return await self._make_request('POST', url, data=data, json=json, headers=headers, timeout=timeout)
    
    async def head(self, url: str, headers: Optional[Dict[str, str]] = None,
                   timeout: Optional[float] = None) -> HTTPResponse:
        """HEAD request ultra-optimizado."""
        return await self._make_request('HEAD', url, headers=headers, timeout=timeout)
    
    async async def _make_request(self, method: str, url: str, **kwargs) -> HTTPResponse:
        """Realizar request con optimizaciones."""
        start_time = time.perf_counter()
        
        # Rate limiting
        async with self.rate_limit_semaphore:
            await self._enforce_rate_limit()
        
        # Verificar cache para GET requests
        if method == 'GET' and self.enable_cache:
            cache_key = self._generate_cache_key(method, url, kwargs.get('headers'))
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                self.stats.cache_hits += 1
                return cached_response
        
        self.stats.cache_misses += 1
        
        # Realizar request
        try:
            response = await self.client.request(method, url, **kwargs)
            
            # Procesar respuesta
            http_response = await self._process_response(response, start_time)
            
            # Cachear respuesta para GET requests exitosos
            if method == 'GET' and self.enable_cache and response.status_code == 200:
                self._cache_response(cache_key, http_response)
            
            # Actualizar estadísticas
            self._update_stats(http_response, True)
            
            return http_response
            
        except Exception as e:
            logger.error(f"HTTP request failed: {method} {url} - {e}")
            self._update_stats(None, False)
            raise
    
    async def get_many(self, urls: List[str], headers: Optional[Dict[str, str]] = None,
                       max_concurrent: Optional[int] = None) -> List[HTTPResponse]:
        """Múltiples GET requests en paralelo."""
        if max_concurrent is None:
            max_concurrent = min(len(urls), self.max_connections)
        
        # Crear semáforo para limitar concurrencia
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async async def fetch_url(url: str) -> HTTPResponse:
            async with semaphore:
                return await self.get(url, headers=headers)
        
        # Ejecutar requests en paralelo
        tasks = [fetch_url(url) for url in urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar resultados
        results = []
        for response in responses:
            if isinstance(response, Exception):
                logger.error(f"Request failed: {response}")
                results.append(None)
            else:
                results.append(response)
        
        return results
    
    async def check_url_health(self, url: str, timeout: float = 5.0) -> bool:
        """Verificar salud de URL."""
        try:
            response = await self.head(url, timeout=timeout)
            return 200 <= response.status_code < 400
        except Exception:
            return False
    
    async async def download_file(self, url: str, filepath: str, chunk_size: int = 8192) -> bool:
        """Descargar archivo con streaming."""
        try:
            async with self.client.stream('GET', url) as response:
                if response.status_code != 200:
                    return False
                
                with open(filepath, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                        f.write(chunk)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        self.stats.total_bytes_downloaded += len(chunk)
                
                return True
                
        except Exception as e:
            logger.error(f"File download failed: {url} - {e}")
            return False
    
    async def _process_response(self, response: httpx.Response, start_time: float) -> HTTPResponse:
        """Procesar respuesta HTTP."""
        elapsed = time.perf_counter() - start_time
        
        # Obtener contenido
        content = response.content
        text = response.text
        
        # Comprimir contenido si es grande
        if self.enable_compression and len(content) > 1024:
            try:
                compressed_content = self.compressor.compress(content)
                compression_ratio = (len(content) - len(compressed_content)) / len(content)
                self.stats.compression_savings += compression_ratio
                content = compressed_content
            except Exception as e:
                logger.warning(f"Content compression failed: {e}")
        
        return HTTPResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            content=content,
            text=text,
            url=str(response.url),
            elapsed=elapsed,
            encoding=response.encoding or 'utf-8'
        )
    
    def _generate_cache_key(self, method: str, url: str, headers: Optional[Dict[str, str]]) -> str:
        """Generar clave de cache."""
        
        # Normalizar URL
        parsed_url = urlparse(url)
        normalized_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
        if parsed_url.query:
            normalized_url += f"?{parsed_url.query}"
        
        # Crear clave
        key_data = f"{method}:{normalized_url}"
        if headers:
            # Ordenar headers para consistencia
            sorted_headers = sorted(headers.items())
            key_data += f":{orjson.dumps(sorted_headers)}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[HTTPResponse]:
        """Obtener respuesta del cache."""
        if cache_key in self.response_cache:
            cached_data = self.response_cache[cache_key]
            if time.time() < cached_data['expires']:
                return cached_data['response']
            else:
                del self.response_cache[cache_key]
        
        return None
    
    def _cache_response(self, cache_key: str, response: HTTPResponse):
        """Cachear respuesta."""
        if len(self.response_cache) >= self.cache_size:
            # Remover elemento más antiguo
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
        
        self.response_cache[cache_key] = {
            'response': response,
            'expires': time.time() + self.cache_ttl
        }
    
    async def _enforce_rate_limit(self) -> Any:
        """Aplicar rate limiting."""
        current_time = time.time()
        
        # Limpiar tiempos antiguos
        self.request_times = [t for t in self.request_times if current_time - t < self.rate_limit_window]
        
        # Verificar si necesitamos esperar
        if len(self.request_times) >= self.rate_limit:
            wait_time = self.rate_limit_window - (current_time - self.request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Registrar tiempo de request
        self.request_times.append(current_time)
    
    def _update_stats(self, response: Optional[HTTPResponse], success: bool):
        """Actualizar estadísticas."""
        self.stats.total_requests += 1
        
        if success:
            self.stats.successful_requests += 1
            if response:
                self.stats.total_bytes_downloaded += len(response.content)
                self.stats.average_response_time = (
                    (self.stats.average_response_time * (self.stats.total_requests - 1) + response.elapsed) 
                    / self.stats.total_requests
                )
        else:
            self.stats.failed_requests += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cliente HTTP."""
        return {
            'client_type': 'ultra_optimized_v2',
            'total_requests': self.stats.total_requests,
            'successful_requests': self.stats.successful_requests,
            'failed_requests': self.stats.failed_requests,
            'success_rate': self.stats.successful_requests / max(self.stats.total_requests, 1),
            'average_response_time': self.stats.average_response_time,
            'total_bytes_downloaded': self.stats.total_bytes_downloaded,
            'total_bytes_uploaded': self.stats.total_bytes_uploaded,
            'cache_hits': self.stats.cache_hits,
            'cache_misses': self.stats.cache_misses,
            'cache_hit_ratio': self.stats.cache_hits / max(self.stats.cache_hits + self.stats.cache_misses, 1),
            'compression_savings': self.stats.compression_savings,
            'rate_limit': self.rate_limit,
            'max_connections': self.max_connections,
            'http2_enabled': self.enable_http2,
            'compression_enabled': self.enable_compression
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check del cliente HTTP."""
        try:
            # Test simple request
            response = await self.get('https://httpbin.org/get', timeout=5.0)
            
            return {
                'status': 'healthy',
                'client_type': 'ultra_optimized_v2',
                'test_response_time': response.elapsed,
                'test_status_code': response.status_code,
                'http2_enabled': self.enable_http2,
                'max_connections': self.max_connections
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'client_type': 'ultra_optimized_v2',
                'error': str(e)
            }
    
    async def close(self) -> Any:
        """Cerrar cliente HTTP."""
        try:
            await self.client.aclose()
            logger.info("HTTP client closed")
        except Exception as e:
            logger.error(f"Error closing HTTP client: {e}")
    
    async def __aenter__(self) -> Any:
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Context manager exit."""
        await self.close() 