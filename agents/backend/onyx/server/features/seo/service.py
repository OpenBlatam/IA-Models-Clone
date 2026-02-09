from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import time
import asyncio
import httpx
from bs4 import BeautifulSoup
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
from .models import SEOScrapeRequest, SEOScrapeResponse, SEOAnalysis
from typing import Any, List, Dict, Optional
"""
Servicio SEO Ultra-Optimizado con arquitectura modular y refactorizada.
"""



# Configurar logging optimizado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurar tracemalloc para monitoreo de memoria
tracemalloc.start()


@dataclass
class SEOMetrics:
    """Métricas de rendimiento del análisis SEO."""
    load_time: float
    memory_usage: float
    cache_hit: bool
    processing_time: float
    elements_extracted: int


class HTMLParser(ABC):
    """Interfaz abstracta para parsers de HTML."""
    
    @abstractmethod
    def parse(self, html_content: str, url: str) -> Dict[str, Any]:
        """Parsea el contenido HTML y extrae datos SEO."""
        pass


class LXMLParser(HTMLParser):
    """Parser ultra-rápido usando lxml."""
    
    def parse(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extrae información SEO usando lxml (más rápido que BeautifulSoup)."""
        seo_data = self._initialize_seo_data()
        
        try:
            tree = html.fromstring(html_content)
            self._extract_basic_info(tree, seo_data)
            self._extract_meta_tags(tree, seo_data)
            self._extract_headers(tree, seo_data)
            self._extract_images(tree, seo_data)
            self._extract_links(tree, seo_data, url)
            self._extract_content(tree, seo_data)
            self._extract_structured_data(tree, seo_data)
            self._calculate_performance_metrics(tree, seo_data)
            
        except Exception as e:
            logger.error(f"Error en parsing lxml: {e}")
            return self._fallback_parse(html_content, url)
        
        return seo_data
    
    def _initialize_seo_data(self) -> Dict[str, Any]:
        """Inicializa la estructura de datos SEO."""
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
    
    def _extract_basic_info(self, tree, seo_data: Dict[str, Any]):
        """Extrae información básica de la página."""
        # Título
        title_elements = tree.xpath('//title/text()')
        if title_elements:
            seo_data["title"] = title_elements[0].strip()
        
        # Canonical URL
        canonical_elements = tree.xpath('//link[@rel="canonical"]/@href')
        if canonical_elements:
            seo_data["canonical_url"] = canonical_elements[0]
        
        # Language y charset
        html_element = tree.xpath('//html')[0] if tree.xpath('//html') else None
        if html_element:
            seo_data["language"] = html_element.get('lang', '')
            seo_data["charset"] = html_element.get('charset', '')
    
    def _extract_meta_tags(self, tree, seo_data: Dict[str, Any]):
        """Extrae meta tags usando XPath."""
        meta_elements = tree.xpath('//meta')
        for meta in meta_elements:
            name = meta.get('name', '').lower()
            property_attr = meta.get('property', '').lower()
            content = meta.get('content', '')

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
    
    def _extract_headers(self, tree, seo_data: Dict[str, Any]):
        """Extrae headers usando XPath."""
        seo_data["h1_tags"] = [h.strip() for h in tree.xpath('//h1/text()')]
        seo_data["h2_tags"] = [h.strip() for h in tree.xpath('//h2/text()')]
        seo_data["h3_tags"] = [h.strip() for h in tree.xpath('//h3/text()')]
    
    def _extract_images(self, tree, seo_data: Dict[str, Any]):
        """Extrae información de imágenes optimizada."""
        img_elements = tree.xpath('//img')[:20]  # Limitar a 20 imágenes
        for img in img_elements:
            seo_data["images"].append({
                "src": img.get('src', ''),
                "alt": img.get('alt', ''),
                "title": img.get('title', ''),
                "loading": img.get('loading', ''),
                "width": img.get('width', ''),
                "height": img.get('height', '')
            })
    
    def _extract_links(self, tree, seo_data: Dict[str, Any], url: str):
        """Extrae enlaces optimizados."""
        link_elements = tree.xpath('//a[@href]')[:50]  # Limitar a 50 enlaces
        base_domain = urlparse(url).netloc
        
        for link in link_elements:
            href = link.get('href')
            if href:
                full_url = urljoin(url, href)
                link_text = ''.join(link.xpath('.//text()')).strip()[:100]
                seo_data["links"].append({
                    "url": full_url,
                    "text": link_text,
                    "title": link.get('title', ''),
                    "is_internal": urlparse(full_url).netloc == base_domain,
                    "rel": link.get('rel', '').split() if link.get('rel') else []
                })
    
    def _extract_content(self, tree, seo_data: Dict[str, Any]):
        """Extrae contenido principal optimizado."""
        content_selectors = [
            '//main',
            '//article',
            '//*[@role="main"]',
            '//div[contains(@class, "content")]',
            '//div[@id="content"]'
        ]
        
        main_content = None
        for selector in content_selectors:
            elements = tree.xpath(selector)
            if elements:
                main_content = elements[0]
                break
        
        if not main_content:
            main_content = tree.xpath('//body')[0] if tree.xpath('//body') else None

        if main_content:
            text_content = ' '.join(main_content.xpath('.//text()'))
            seo_data["content_length"] = len(text_content.strip())
    
    def _extract_structured_data(self, tree, seo_data: Dict[str, Any]):
        """Extrae structured data (JSON-LD)."""
        json_ld_scripts = tree.xpath('//script[@type="application/ld+json"]/text()')
        for script in json_ld_scripts:
            try:
                structured_data = orjson.loads(script)
                seo_data["structured_data"].append(structured_data)
            except:
                continue
    
    def _calculate_performance_metrics(self, tree, seo_data: Dict[str, Any]):
        """Calcula métricas de rendimiento básicas."""
        seo_data["performance_metrics"] = {
            "total_elements": len(tree.xpath('//*')),
            "script_count": len(tree.xpath('//script')),
            "style_count": len(tree.xpath('//style')),
            "link_count": len(tree.xpath('//link')),
            "meta_count": len(tree.xpath('//meta'))
        }
    
    def _fallback_parse(self, html_content: str, url: str) -> Dict[str, Any]:
        """Fallback usando BeautifulSoup si lxml falla."""
        return BeautifulSoupParser().parse(html_content, url)


class BeautifulSoupParser(HTMLParser):
    """Parser de fallback usando BeautifulSoup."""
    
    def parse(self, html_content: str, url: str) -> Dict[str, Any]:
        """Parsea HTML usando BeautifulSoup como fallback."""
        soup = BeautifulSoup(html_content, 'html.parser')
        seo_data = self._initialize_seo_data()
        
        # Implementación básica de fallback
        title_tag = soup.find('title')
        if title_tag:
            seo_data["title"] = title_tag.get_text().strip()
        
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            seo_data["meta_description"] = meta_desc.get('content', '')
        
        return seo_data


class HTTPClient:
    """Cliente HTTP asíncrono optimizado."""
    
    def __init__(self) -> Any:
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async async def fetch(self, url: str) -> Optional[str]:
        """Obtiene HTML de forma asíncrona con retry."""
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def measure_load_time(self, url: str) -> Optional[float]:
        """Mide el tiempo de carga de forma asíncrona."""
        try:
            start_time = time.time()
            response = await self.client.get(url)
            response.raise_for_status()
            await response.aread()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            end_time = time.time()
            return end_time - start_time
        except:
            return None
    
    async def close(self) -> Any:
        """Cierra el cliente HTTP."""
        await self.client.aclose()


class SeleniumManager:
    """Gestor de Selenium optimizado."""
    
    def __init__(self) -> Any:
        self.driver = None
        self._setup_driver()
    
    def _setup_driver(self) -> Any:
        """Configura Selenium con optimizaciones avanzadas."""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--disable-images")
            chrome_options.add_argument("--disable-javascript")
            chrome_options.add_argument("--disable-css")
            chrome_options.add_argument("--disable-fonts")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            chrome_options.add_argument("--memory-pressure-off")
            chrome_options.add_argument("--max_old_space_size=4096")
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(30)
            logger.info("Selenium configurado correctamente")
        except Exception as e:
            logger.warning(f"Error configurando Selenium: {e}")
            self.driver = None
    
    def get_page_source(self, url: str) -> Optional[str]:
        """Obtiene el HTML de una página usando Selenium."""
        try:
            if not self.driver:
                return None
            
            self.driver.get(url)
            return self.driver.page_source
        except Exception as e:
            logger.error(f"Error con Selenium: {e}")
            return None
    
    def check_mobile_friendly(self, url: str) -> bool:
        """Verifica compatibilidad móvil optimizada."""
        try:
            if not self.driver:
                return True
            
            self.driver.set_window_size(375, 667)
            self.driver.get(url)
            
            viewport_meta = self.driver.find_element(By.CSS_SELECTOR, 'meta[name="viewport"]')
            return viewport_meta is not None
        except:
            return True
    
    def close(self) -> Any:
        """Cierra el driver de Selenium."""
        if self.driver:
            self.driver.quit()


class CacheManager:
    """Gestor de cache optimizado."""
    
    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        
    """__init__ function."""
self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor del cache."""
        value = self.cache.get(key)
        if value is not None:
            self.hits += 1
        else:
            self.misses += 1
        return value
    
    def set(self, key: str, value: Any):
        """Guarda un valor en el cache."""
        self.cache[key] = value
    
    def clear(self) -> int:
        """Limpia el cache y retorna el número de elementos eliminados."""
        size = len(self.cache)
        self.cache.clear()
        return size
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.cache.maxsize,
            "ttl": self.cache.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2)
        }


class SEOAnalyzer:
    """Analizador SEO con LangChain."""
    
    def __init__(self, api_key: Optional[str] = None):
        
    """__init__ function."""
self.llm = None
        if api_key:
            self._setup_langchain(api_key)
    
    def _setup_langchain(self, api_key: str):
        """Configura LangChain."""
        try:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                api_key=api_key,
                max_retries=3,
                timeout=30
            )
            logger.info("LangChain configurado correctamente")
        except Exception as e:
            logger.warning(f"Error configurando LangChain: {e}")
    
    async def analyze(self, seo_data: Dict[str, Any], url: str) -> Dict[str, Any]:
        """Analiza datos SEO con LangChain de forma asíncrona."""
        if not self.llm:
            return self._fallback_analysis(seo_data, url)
        
        try:
            response = await asyncio.to_thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                self.llm.invoke,
                self._create_prompt(seo_data, url)
            )
            
            try:
                analysis = orjson.loads(response.content)
                return analysis
            except:
                return self._fallback_analysis(seo_data, url)
                
        except Exception as e:
            logger.error(f"Error en análisis LangChain: {e}")
            return self._fallback_analysis(seo_data, url)
    
    def _create_prompt(self, seo_data: Dict[str, Any], url: str) -> ChatPromptTemplate:
        """Crea el prompt optimizado para LangChain."""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Analiza SEO y responde solo en JSON válido:
            {
                "seo_score": 0-100,
                "recommendations": ["lista de 3-5 recomendaciones"],
                "technical_issues": ["lista de problemas"],
                "analysis_summary": "resumen en 200 chars"
            }"""),
            ("human", """
            URL: {url}
            Título: {title}
            Meta: {meta_description}
            H1: {h1_count}, H2: {h2_count}, H3: {h3_count}
            Imágenes: {image_count} (alt: {images_with_alt})
            Enlaces: {link_count} (int: {internal_links}, ext: {external_links})
            Contenido: {content_length} chars
            Keywords: {keywords}
            Social: {social_tags}
            Canonical: {canonical}
            Robots: {robots}
            """)
        ])
        
        # Preparar datos optimizados
        h1_count = len(seo_data["h1_tags"])
        h2_count = len(seo_data["h2_tags"])
        h3_count = len(seo_data["h3_tags"])
        image_count = len(seo_data["images"])
        images_with_alt = len([img for img in seo_data["images"] if img["alt"]])
        link_count = len(seo_data["links"])
        internal_links = len([link for link in seo_data["links"] if link["is_internal"]])
        external_links = link_count - internal_links
        social_tags = len(seo_data["social_media_tags"])
        
        return prompt_template.format_messages(
            url=url,
            title=seo_data["title"][:100],
            meta_description=seo_data["meta_description"][:200],
            h1_count=h1_count,
            h2_count=h2_count,
            h3_count=h3_count,
            image_count=image_count,
            images_with_alt=images_with_alt,
            link_count=link_count,
            internal_links=internal_links,
            external_links=external_links,
            content_length=seo_data["content_length"],
            keywords=", ".join(seo_data["keywords"][:10]),
            social_tags=social_tags,
            canonical=seo_data["canonical_url"],
            robots=seo_data["robots_meta"]
        )
    
    def _fallback_analysis(self, seo_data: Dict[str, Any], url: str) -> Dict[str, Any]:
        """Análisis básico optimizado sin LangChain."""
        score = 50
        recommendations = []
        issues = []
        
        # Análisis básico optimizado
        if not seo_data["title"]:
            issues.append("Falta título de página")
            score -= 10
        
        if not seo_data["meta_description"]:
            recommendations.append("Agregar meta descripción")
            score -= 5
        
        if len(seo_data["h1_tags"]) == 0:
            issues.append("No hay tags H1")
            score -= 10
        elif len(seo_data["h1_tags"]) > 1:
            issues.append("Múltiples tags H1")
            score -= 5

        if seo_data["content_length"] < 300:
            recommendations.append("Agregar más contenido")
            score -= 10

        images_without_alt = len([img for img in seo_data["images"] if not img["alt"]])
        if images_without_alt > 0:
            recommendations.append(f"Agregar alt text a {images_without_alt} imágenes")

        # Análisis adicional de rendimiento
        perf_metrics = seo_data.get("performance_metrics", {})
        if perf_metrics.get("script_count", 0) > 20:
            recommendations.append("Reducir número de scripts")
            score -= 5

        return {
            "seo_score": max(0, score),
            "recommendations": recommendations[:3],
            "technical_issues": issues,
            "analysis_summary": f"Análisis básico: {score}/100 puntos"
        }


class SEOService:
    """Servicio SEO ultra-optimizado con arquitectura modular."""
    
    def __init__(self) -> Any:
        self.http_client = HTTPClient()
        self.selenium_manager = SeleniumManager()
        self.cache_manager = CacheManager()
        self.seo_analyzer = SEOAnalyzer(os.getenv("OPENAI_API_KEY"))
        self.html_parser = LXMLParser()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def scrape(self, request: SEOScrapeRequest) -> SEOScrapeResponse:
        """Realiza scraping SEO ultra-optimizado de una URL."""
        try:
            if not request.url:
                return SEOScrapeResponse(success=False, error="URL vacía")

            url = self._normalize_url(request.url)
            logger.info(f"Iniciando análisis SEO optimizado de: {url}")

            # Verificar cache
            cache_key = f"{url}_{hash(str(request.options))}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"Resultado obtenido de cache para: {url}")
                return cached_result

            # Realizar análisis
            start_time = time.time()
            result = await self._perform_analysis(url, request.options)
            processing_time = time.time() - start_time
            
            if result.success:
                # Guardar en cache
                self.cache_manager.set(cache_key, result)
                logger.info(f"Análisis completado en {processing_time:.2f}s para: {url}")
            
            return result

        except Exception as e:
            logger.error(f"Error durante el scraping: {str(e)}")
            return SEOScrapeResponse(
                success=False,
                error=f"Error durante el scraping: {str(e)}"
            )
    
    def _normalize_url(self, url: str) -> str:
        """Normaliza la URL."""
        if not url.startswith(('http://', 'https://')):
            return 'https://' + url
        return url
    
    async def _perform_analysis(self, url: str, options: Dict[str, Any]) -> SEOScrapeResponse:
        """Realiza el análisis completo de la URL."""
        # Medir tiempo de carga
        load_time = await self.http_client.measure_load_time(url)
        
        # Obtener HTML
        html_content = await self._get_html_content(url, options)
        if not html_content:
            return SEOScrapeResponse(
                success=False,
                error=f"No se pudo acceder a la URL: {url}"
            )

        # Extraer datos SEO
        seo_data = self.html_parser.parse(html_content, url)
        
        # Analizar con IA
        langchain_analysis = await self.seo_analyzer.analyze(seo_data, url)
        
        # Verificar compatibilidad móvil
        mobile_friendly = self.selenium_manager.check_mobile_friendly(url)
        
        # Crear objeto SEOAnalysis
        analysis = SEOAnalysis(
            title=seo_data["title"],
            meta_description=seo_data["meta_description"],
            h1_tags=seo_data["h1_tags"],
            h2_tags=seo_data["h2_tags"],
            h3_tags=seo_data["h3_tags"],
            images=seo_data["images"],
            links=seo_data["links"],
            keywords=seo_data["keywords"],
            content_length=seo_data["content_length"],
            load_time=load_time,
            seo_score=langchain_analysis.get("seo_score"),
            recommendations=langchain_analysis.get("recommendations", []),
            technical_issues=langchain_analysis.get("technical_issues", []),
            mobile_friendly=mobile_friendly,
            page_speed=self._calculate_page_speed(load_time),
            social_media_tags=seo_data["social_media_tags"]
        )

        return SEOScrapeResponse(
            success=True,
            data=analysis,
            raw_html=html_content[:5000] if html_content else None,
            analysis_summary=langchain_analysis.get("analysis_summary", "")
        )
    
    async def _get_html_content(self, url: str, options: Dict[str, Any]) -> Optional[str]:
        """Obtiene el contenido HTML de la URL."""
        # Intentar con Selenium si está habilitado
        if options.get('use_selenium', False):
            html_content = self.selenium_manager.get_page_source(url)
            if html_content:
                return html_content
        
        # Fallback a HTTP client
        return await self.http_client.fetch(url)
    
    def _calculate_page_speed(self, load_time: Optional[float]) -> str:
        """Calcula la velocidad de la página."""
        if not load_time:
            return "Desconocida"
        elif load_time < 2:
            return "Muy rápida"
        elif load_time < 3:
            return "Rápida"
        elif load_time < 5:
            return "Normal"
        else:
            return "Lenta"
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache."""
        return self.cache_manager.get_stats()
    
    def clear_cache(self) -> int:
        """Limpia el cache y retorna elementos eliminados."""
        return self.cache_manager.clear()
    
    async def close(self) -> Any:
        """Cierra todos los recursos."""
        await self.http_client.close()
        self.selenium_manager.close()
        self.executor.shutdown(wait=False)


# Función estática para compatibilidad
async def scrape(request: SEOScrapeRequest) -> SEOScrapeResponse:
    """Realiza scraping SEO ultra-optimizado de una URL."""
    service = SEOService()
    try:
        return await service.scrape(request)
    finally:
        await service.close() 