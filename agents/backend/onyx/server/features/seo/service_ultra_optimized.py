from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import time
import asyncio
import httpx
from selectolax.parser import HTMLParser as SelectolaxParser
from lxml import html
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from urllib.parse import urljoin, urlparse
import re
from typing import Dict, List, Any, Optional, Tuple
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiofiles
import orjson
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential
import tracemalloc
from dataclasses import dataclass
from abc import ABC, abstractmethod
import cchardet
import regex
import msgpack
import zstandard
from loguru import logger
import psutil
import asyncio_throttle
from .models import SEOScrapeRequest, SEOScrapeResponse, SEOAnalysis
from typing import Any, List, Dict, Optional
"""
Servicio SEO Ultra-Optimizado con las librerías más rápidas disponibles.
Máximo rendimiento y eficiencia de memoria.
"""



# Configurar tracemalloc para monitoreo de memoria
tracemalloc.start()

# Configurar loguru para logging ultra-eficiente
logger.remove()
logger.add(
    "logs/seo_service.log",
    rotation="100 MB",
    compression="zstd",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
)


@dataclass
class SEOMetrics:
    """Métricas de rendimiento ultra-optimizadas."""
    load_time: float
    memory_usage: float
    cache_hit: bool
    processing_time: float
    elements_extracted: int
    compression_ratio: float
    network_latency: float


class UltraFastHTMLParser(ABC):
    """Interfaz abstracta para parsers HTML ultra-rápidos."""
    
    @abstractmethod
    def parse(self, html_content: str, url: str) -> Dict[str, Any]:
        """Parsea el contenido HTML con máxima velocidad."""
        pass


class SelectolaxUltraParser(UltraFastHTMLParser):
    """Parser ultra-rápido usando selectolax (más rápido que lxml)."""
    
    def parse(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extrae información SEO usando selectolax (máxima velocidad)."""
        seo_data = self._initialize_seo_data()
        
        try:
            # Usar selectolax para parsing ultra-rápido
            parser = SelectolaxParser(html_content)
            self._extract_basic_info_selectolax(parser, seo_data)
            self._extract_meta_tags_selectolax(parser, seo_data)
            self._extract_headers_selectolax(parser, seo_data)
            self._extract_images_selectolax(parser, seo_data)
            self._extract_links_selectolax(parser, seo_data, url)
            self._extract_content_selectolax(parser, seo_data)
            self._extract_structured_data_selectolax(parser, seo_data)
            self._calculate_performance_metrics_selectolax(parser, seo_data)
            
        except Exception as e:
            logger.error(f"Error en parsing selectolax: {e}")
            return self._fallback_parse(html_content, url)
        
        return seo_data
    
    def _initialize_seo_data(self) -> Dict[str, Any]:
        """Inicializa la estructura de datos SEO optimizada."""
        return {
            "title": "",
            "meta_description": "",
            "h1_tags": [],
            "h2_tags": [],
            "h3_tags": [],
            "images": [],
            "links": [],
            "keywords": [],
            "content_length": 0,
            "social_media_tags": {},
            "canonical_url": "",
            "robots_meta": "",
            "language": "",
            "charset": "",
            "structured_data": [],
            "performance_metrics": {}
        }
    
    def _extract_basic_info_selectolax(self, parser: SelectolaxParser, seo_data: Dict[str, Any]):
        """Extrae información básica usando selectolax."""
        # Título - ultra-rápido
        title_elem = parser.css_first('title')
        if title_elem:
            seo_data["title"] = title_elem.text().strip()
        
        # Canonical URL
        canonical_elem = parser.css_first('link[rel="canonical"]')
        if canonical_elem:
            seo_data["canonical_url"] = canonical_elem.attributes.get('href', '')
        
        # Language y charset
        html_elem = parser.css_first('html')
        if html_elem:
            seo_data["language"] = html_elem.attributes.get('lang', '')
            seo_data["charset"] = html_elem.attributes.get('charset', '')
    
    def _extract_meta_tags_selectolax(self, parser: SelectolaxParser, seo_data: Dict[str, Any]):
        """Extrae meta tags usando selectolax (ultra-rápido)."""
        meta_elements = parser.css('meta')
        for meta in meta_elements:
            name = meta.attributes.get('name', '').lower()
            property_attr = meta.attributes.get('property', '').lower()
            content = meta.attributes.get('content', '')

            if name == 'description':
                seo_data["meta_description"] = content
            elif name == 'keywords':
                seo_data["keywords"] = [kw.strip() for kw in content.split(',') if kw.strip()]
            elif name == 'robots':
                seo_data["robots_meta"] = content
            elif property_attr.startswith('og:'):
                seo_data["social_media_tags"][property_attr] = content
            elif name.startswith('twitter:'):
                seo_data["social_media_tags"][name] = content
    
    def _extract_headers_selectolax(self, parser: SelectolaxParser, seo_data: Dict[str, Any]):
        """Extrae headers usando selectolax."""
        seo_data["h1_tags"] = [h.text().strip() for h in parser.css('h1')]
        seo_data["h2_tags"] = [h.text().strip() for h in parser.css('h2')]
        seo_data["h3_tags"] = [h.text().strip() for h in parser.css('h3')]
    
    def _extract_images_selectolax(self, parser: SelectolaxParser, seo_data: Dict[str, Any]):
        """Extrae información de imágenes ultra-optimizada."""
        img_elements = parser.css('img')[:15]  # Limitar a 15 imágenes
        for img in img_elements:
            seo_data["images"].append({
                "src": img.attributes.get('src', ''),
                "alt": img.attributes.get('alt', ''),
                "title": img.attributes.get('title', ''),
                "loading": img.attributes.get('loading', ''),
                "width": img.attributes.get('width', ''),
                "height": img.attributes.get('height', '')
            })
    
    def _extract_links_selectolax(self, parser: SelectolaxParser, seo_data: Dict[str, Any], url: str):
        """Extrae enlaces ultra-optimizados."""
        link_elements = parser.css('a[href]')[:30]  # Limitar a 30 enlaces
        base_domain = urlparse(url).netloc
        
        for link in link_elements:
            href = link.attributes.get('href')
            if href:
                full_url = urljoin(url, href)
                link_text = link.text().strip()[:80]
                seo_data["links"].append({
                    "url": full_url,
                    "text": link_text,
                    "title": link.attributes.get('title', ''),
                    "is_internal": urlparse(full_url).netloc == base_domain,
                    "rel": link.attributes.get('rel', '').split() if link.attributes.get('rel') else []
                })
    
    def _extract_content_selectolax(self, parser: SelectolaxParser, seo_data: Dict[str, Any]):
        """Extrae contenido principal ultra-optimizado."""
        content_selectors = [
            'main',
            'article',
            '[role="main"]',
            '.content',
            '#content'
        ]
        
        main_content = None
        for selector in content_selectors:
            elements = parser.css(selector)
            if elements:
                main_content = elements[0]
                break
        
        if main_content:
            content_text = main_content.text()
            seo_data["content_length"] = len(content_text)
    
    def _extract_structured_data_selectolax(self, parser: SelectolaxParser, seo_data: Dict[str, Any]):
        """Extrae datos estructurados usando selectolax."""
        script_elements = parser.css('script[type="application/ld+json"]')
        for script in script_elements:
            try:
                data = orjson.loads(script.text())
                seo_data["structured_data"].append(data)
            except:
                continue
    
    def _calculate_performance_metrics_selectolax(self, parser: SelectolaxParser, seo_data: Dict[str, Any]):
        """Calcula métricas de rendimiento."""
        seo_data["performance_metrics"] = {
            "total_elements": len(parser.css('*')),
            "images_count": len(parser.css('img')),
            "links_count": len(parser.css('a')),
            "scripts_count": len(parser.css('script')),
            "stylesheets_count": len(parser.css('link[rel="stylesheet"]'))
        }
    
    def _fallback_parse(self, html_content: str, url: str) -> Dict[str, Any]:
        """Fallback usando lxml si selectolax falla."""
        try:
            tree = html.fromstring(html_content)
            return self._parse_with_lxml(tree, url)
        except:
            return self._initialize_seo_data()
    
    def _parse_with_lxml(self, tree, url: str) -> Dict[str, Any]:
        """Parsing con lxml como fallback."""
        seo_data = self._initialize_seo_data()
        
        # Extraer datos básicos con lxml
        title_elements = tree.xpath('//title/text()')
        if title_elements:
            seo_data["title"] = title_elements[0].strip()
        
        return seo_data


class UltraFastHTTPClient:
    """Cliente HTTP ultra-optimizado con connection pooling."""
    
    def __init__(self) -> Any:
        self.session = None
        self.throttler = asyncio_throttle.Throttler(rate_limit=100, period=60)
        self._setup_session()
    
    def _setup_session(self) -> Any:
        """Configura sesión HTTP ultra-optimizada."""
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        timeout = httpx.Timeout(10.0, connect=5.0)
        
        self.session = httpx.AsyncClient(
            limits=limits,
            timeout=timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            },
            follow_redirects=True,
            http2=True
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
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
                return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def measure_load_time(self, url: str) -> Optional[float]:
        """Mide tiempo de carga ultra-optimizado."""
        start_time = time.perf_counter()
        try:
            response = await self.session.get(url)
            response.raise_for_status()
            return time.perf_counter() - start_time
        except:
            return None
    
    async def close(self) -> Any:
        """Cierra la sesión HTTP."""
        if self.session:
            await self.session.aclose()


class UltraOptimizedCacheManager:
    """Gestor de caché ultra-optimizado con compresión."""
    
    def __init__(self, maxsize: int = 2000, ttl: int = 3600):
        
    """__init__ function."""
self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.compressor = zstandard.ZstdCompressor(level=3)
        self.decompressor = zstandard.ZstdDecompressor()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "compression_ratio": 0.0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene datos del caché con descompresión."""
        try:
            compressed_data = self.cache.get(key)
            if compressed_data:
                self.stats["hits"] += 1
                decompressed_data = self.decompressor.decompress(compressed_data)
                return orjson.loads(decompressed_data)
            else:
                self.stats["misses"] += 1
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.stats["misses"] += 1
            return None
    
    def set(self, key: str, value: Any):
        """Almacena datos en caché con compresión."""
        try:
            json_data = orjson.dumps(value)
            compressed_data = self.compressor.compress(json_data)
            
            # Calcular ratio de compresión
            original_size = len(json_data)
            compressed_size = len(compressed_data)
            compression_ratio = (original_size - compressed_size) / original_size
            self.stats["compression_ratio"] = compression_ratio
            
            self.cache[key] = compressed_data
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def clear(self) -> int:
        """Limpia el caché y retorna elementos eliminados."""
        size = len(self.cache)
        self.cache.clear()
        return size
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "compression_ratio": self.stats["compression_ratio"],
            "cache_size": len(self.cache),
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024  # MB
        }


class UltraFastSEOAnalyzer:
    """Analizador SEO ultra-optimizado con LangChain."""
    
    def __init__(self, api_key: Optional[str] = None):
        
    """__init__ function."""
self.api_key = api_key
        self.llm = None
        self._setup_langchain(api_key)
    
    def _setup_langchain(self, api_key: str):
        """Configura LangChain ultra-optimizado."""
        if api_key:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                    max_tokens=1000,
                    request_timeout=30
                )
            except Exception as e:
                logger.error(f"Error setting up LangChain: {e}")
    
    async def analyze(self, seo_data: Dict[str, Any], url: str) -> Dict[str, Any]:
        """Analiza datos SEO con LangChain ultra-optimizado."""
        if not self.llm:
            return self._fallback_analysis(seo_data, url)
        
        try:
            prompt = self._create_optimized_prompt(seo_data, url)
            response = await self.llm.ainvoke(prompt)
            
            # Parsear respuesta con orjson (más rápido)
            analysis = orjson.loads(response.content)
            return analysis
            
        except Exception as e:
            logger.error(f"Error in LangChain analysis: {e}")
            return self._fallback_analysis(seo_data, url)
    
    def _create_optimized_prompt(self, seo_data: Dict[str, Any], url: str) -> ChatPromptTemplate:
        """Crea prompt optimizado para análisis SEO."""
        template = """
        Analiza los siguientes datos SEO de {url} y proporciona recomendaciones en formato JSON:
        
        Datos SEO:
        - Título: {title}
        - Meta descripción: {meta_description}
        - Headers H1: {h1_count} elementos
        - Headers H2: {h2_count} elementos
        - Imágenes: {images_count} elementos
        - Enlaces: {links_count} elementos
        - Longitud del contenido: {content_length} caracteres
        
        Proporciona análisis en este formato JSON:
        {{
            "score": 85,
            "recommendations": ["recomendación 1", "recomendación 2"],
            "strengths": ["fortaleza 1", "fortaleza 2"],
            "weaknesses": ["debilidad 1", "debilidad 2"],
            "priority_actions": ["acción 1", "acción 2"]
        }}
        """
        
        return ChatPromptTemplate.from_template(template)
    
    def _fallback_analysis(self, seo_data: Dict[str, Any], url: str) -> Dict[str, Any]:
        """Análisis de fallback ultra-optimizado."""
        score = 50
        recommendations = []
        strengths = []
        weaknesses = []
        
        # Análisis básico ultra-rápido
        if seo_data.get("title"):
            score += 10
            strengths.append("Título presente")
        else:
            weaknesses.append("Falta título")
            recommendations.append("Agregar título SEO")
        
        if seo_data.get("meta_description"):
            score += 10
            strengths.append("Meta descripción presente")
        else:
            weaknesses.append("Falta meta descripción")
            recommendations.append("Agregar meta descripción")
        
        if len(seo_data.get("h1_tags", [])) > 0:
            score += 5
            strengths.append("Headers H1 presentes")
        else:
            weaknesses.append("Falta header H1")
            recommendations.append("Agregar header H1 principal")
        
        if seo_data.get("content_length", 0) > 300:
            score += 10
            strengths.append("Contenido sustancial")
        else:
            weaknesses.append("Contenido insuficiente")
            recommendations.append("Aumentar contenido")
        
        return {
            "score": min(score, 100),
            "recommendations": recommendations[:5],
            "strengths": strengths[:5],
            "weaknesses": weaknesses[:5],
            "priority_actions": recommendations[:3]
        }


class UltraOptimizedSEOService:
    """Servicio SEO ultra-optimizado con máxima eficiencia."""
    
    def __init__(self) -> Any:
        self.http_client = UltraFastHTTPClient()
        self.cache_manager = UltraOptimizedCacheManager()
        self.analyzer = UltraFastSEOAnalyzer()
        self.parser = SelectolaxUltraParser()
        self.selenium_manager = None
        self._setup_selenium()
    
    def _setup_selenium(self) -> Any:
        """Configura Selenium ultra-optimizado."""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-images')
            chrome_options.add_argument('--disable-javascript')
            chrome_options.add_argument('--disable-plugins')
            chrome_options.add_argument('--disable-web-security')
            chrome_options.add_argument('--disable-features=VizDisplayCompositor')
            chrome_options.add_argument('--memory-pressure-off')
            chrome_options.add_argument('--max_old_space_size=4096')
            
            service = Service(ChromeDriverManager().install())
            self.selenium_manager = webdriver.Chrome(service=service, options=chrome_options)
            
        except Exception as e:
            logger.error(f"Error setting up Selenium: {e}")
    
    async def scrape(self, request: SEOScrapeRequest) -> SEOScrapeResponse:
        """Scraping SEO ultra-optimizado."""
        start_time = time.perf_counter()
        memory_start = psutil.Process().memory_info().rss
        
        try:
            # Normalizar URL
            normalized_url = self._normalize_url(request.url)
            
            # Verificar caché
            cache_key = f"seo_analysis:{normalized_url}"
            cached_result = self.cache_manager.get(cache_key)
            
            if cached_result and not request.force_refresh:
                return SEOScrapeResponse(
                    url=normalized_url,
                    success=True,
                    data=cached_result,
                    metrics=SEOMetrics(
                        load_time=0.0,
                        memory_usage=psutil.Process().memory_info().rss / 1024 / 1024,
                        cache_hit=True,
                        processing_time=time.perf_counter() - start_time,
                        elements_extracted=0,
                        compression_ratio=self.cache_manager.stats["compression_ratio"],
                        network_latency=0.0
                    )
                )
            
            # Realizar análisis
            result = await self._perform_ultra_analysis(normalized_url, request.options)
            
            # Guardar en caché
            self.cache_manager.set(cache_key, result)
            
            processing_time = time.perf_counter() - start_time
            memory_usage = (psutil.Process().memory_info().rss - memory_start) / 1024 / 1024
            
            return SEOScrapeResponse(
                url=normalized_url,
                success=True,
                data=result,
                metrics=SEOMetrics(
                    load_time=result.get("load_time", 0.0),
                    memory_usage=memory_usage,
                    cache_hit=False,
                    processing_time=processing_time,
                    elements_extracted=len(result.get("images", [])) + len(result.get("links", [])),
                    compression_ratio=self.cache_manager.stats["compression_ratio"],
                    network_latency=result.get("network_latency", 0.0)
                )
            )
            
        except Exception as e:
            logger.error(f"Error in SEO scraping: {e}")
            return SEOScrapeResponse(
                url=request.url,
                success=False,
                error=str(e),
                metrics=SEOMetrics(
                    load_time=0.0,
                    memory_usage=psutil.Process().memory_info().rss / 1024 / 1024,
                    cache_hit=False,
                    processing_time=time.perf_counter() - start_time,
                    elements_extracted=0,
                    compression_ratio=0.0,
                    network_latency=0.0
                )
            )
    
    def _normalize_url(self, url: str) -> str:
        """Normaliza URL ultra-rápido."""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        return url.rstrip('/')
    
    async def _perform_ultra_analysis(self, url: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza análisis SEO ultra-optimizado."""
        # Obtener contenido HTML
        html_content = await self._get_html_content_ultra(url, options)
        if not html_content:
            raise Exception("No se pudo obtener contenido HTML")
        
        # Parsear HTML ultra-rápido
        seo_data = self.parser.parse(html_content, url)
        
        # Analizar con LangChain
        analysis = await self.analyzer.analyze(seo_data, url)
        
        # Combinar resultados
        result = {
            **seo_data,
            "analysis": analysis,
            "url": url,
            "timestamp": time.time(),
            "load_time": options.get("load_time", 0.0),
            "network_latency": options.get("network_latency", 0.0)
        }
        
        return result
    
    async def _get_html_content_ultra(self, url: str, options: Dict[str, Any]) -> Optional[str]:
        """Obtiene contenido HTML ultra-optimizado."""
        if options.get("use_selenium", False) and self.selenium_manager:
            try:
                self.selenium_manager.get(url)
                return self.selenium_manager.page_source
            except Exception as e:
                logger.error(f"Selenium error: {e}")
        
        # Usar HTTP client ultra-optimizado
        return await self.http_client.fetch(url)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché."""
        return self.cache_manager.get_stats()
    
    def clear_cache(self) -> int:
        """Limpia el caché."""
        return self.cache_manager.clear()
    
    async def close(self) -> Any:
        """Cierra recursos del servicio."""
        await self.http_client.close()
        if self.selenium_manager:
            self.selenium_manager.quit()


# Instancia global ultra-optimizada
seo_service = UltraOptimizedSEOService()


async def scrape(request: SEOScrapeRequest) -> SEOScrapeResponse:
    """Función de scraping ultra-optimizada."""
    return await seo_service.scrape(request) 