"""
Scraper web avanzado para extraer contenido de páginas web
Utiliza múltiples estrategias y librerías para máxima efectividad
"""

import logging
import re
import json
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urljoin, urlparse
from datetime import datetime
import asyncio

import httpx
from bs4 import BeautifulSoup
import trafilatura
from readability import Document
from newspaper import Article
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from cachetools import TTLCache
import chardet

from .metadata_extractors import (
    extract_text_from_element,
    extract_meta_tag,
    extract_attribute_from_element,
    extract_prefixed_meta_tags,
    extract_keywords_list
)
from .element_extractors import extract_links, extract_images
from .extraction_helpers import safe_extract
from .value_extractors import get_value_or_alternative, get_value_with_fallback

logger = logging.getLogger(__name__)


class AdvancedWebScraper:
    """
    Scraper avanzado con múltiples estrategias de extracción:
    - Trafilatura: Extracción de artículos y contenido principal
    - Readability: Limpieza de contenido HTML
    - Newspaper3k: Extracción de noticias y metadatos
    - BeautifulSoup: Parsing HTML avanzado
    - Retry logic: Reintentos automáticos
    - Caché: Almacenamiento temporal de resultados
    """
    
    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        cache_ttl: int = 3600,
        use_cache: bool = True
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_cache = use_cache
        self._client: Optional[httpx.AsyncClient] = None
        
        # Caché con TTL (Time To Live)
        if use_cache:
            self._cache: TTLCache = TTLCache(maxsize=100, ttl=cache_ttl)
        else:
            self._cache = None
        
        # User agents rotativos para evitar bloqueos
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15"
        ]
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Obtener o crear cliente HTTP con configuración avanzada"""
        if self._client is None:
            import random
            user_agent = random.choice(self.user_agents)
            
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0, read=20.0),
                follow_redirects=True,
                max_redirects=10,
                headers={
                    "User-Agent": user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
                    "Accept-Encoding": "gzip, deflate, br",
                    "DNT": "1",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                },
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
            )
        return self._client
    
    async def close(self):
        """Cerrar cliente HTTP"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _detect_encoding(self, content: bytes) -> str:
        """Detectar encoding del contenido"""
        try:
            result = chardet.detect(content)
            return result.get('encoding', 'utf-8')
        except Exception:
            return 'utf-8'
    
    def _normalize_url(self, url: str, base_url: str = None) -> str:
        """Normalizar URLs relativas a absolutas"""
        if base_url:
            return urljoin(base_url, url)
        return url
    
    def _extract_published_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Extraer fecha de publicación de varios formatos"""
        date_selectors = [
            {'name': 'article:published_time'},
            {'property': 'article:published_time'},
            {'name': 'date'},
            {'property': 'date'},
        ]
        for selector in date_selectors:
            date_tag = soup.find('meta', attrs=selector)
            if date_tag:
                date_str = date_tag.get('content', '')
                if date_str:
                    try:
                        from dateutil import parser
                        return parser.parse(date_str).isoformat()
                    except Exception:
                        pass
        return None
    
    def _extract_author(self, content_data: Dict[str, Any], metadata: Dict[str, Any]) -> Any:
        """Extraer autor con manejo de diferentes formatos"""
        authors = content_data.get('authors')
        if isinstance(authors, list):
            return authors if authors else [metadata.get('author', '')]
        return authors if authors else metadata.get('author', '')
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extraer metadatos avanzados de la página"""
        metadata = {
            "title": extract_text_from_element(soup, "title"),
            "description": extract_meta_tag(soup, "description", "name"),
            "keywords": extract_keywords_list(soup),
            "author": extract_meta_tag(soup, "author", "name"),
            "published_date": self._extract_published_date(soup),
            "og": extract_prefixed_meta_tags(soup, "og:", "property"),
            "twitter": extract_prefixed_meta_tags(soup, "twitter:", "name"),
            "canonical": extract_attribute_from_element(
                soup, 'link[rel="canonical"]', 'href'
            ),
            "language": extract_attribute_from_element(soup, "html", "lang") or "en"
        }
        
        return metadata
    
    def _extract_structured_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extraer datos estructurados (JSON-LD, Microdata, RDFa)"""
        structured_data = {
            "json_ld": [],
            "microdata": [],
            "rdfa": []
        }
        
        # JSON-LD
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                import json
                data = json.loads(script.string)
                structured_data["json_ld"].append(data)
            except:
                pass
        
        # Microdata (básico)
        items = soup.find_all(attrs={'itemscope': True})
        for item in items[:5]:  # Limitar a 5 items
            item_data = {}
            item_type = item.get('itemtype', '')
            if item_type:
                item_data['type'] = item_type
            props = item.find_all(attrs={'itemprop': True})
            for prop in props:
                prop_name = prop.get('itemprop', '')
                prop_value = prop.get('content') or prop.get_text(strip=True)
                if prop_name and prop_value:
                    item_data[prop_name] = prop_value
            if item_data:
                structured_data["microdata"].append(item_data)
        
        return structured_data
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException))
    )
    async def _fetch_url(self, url: str) -> httpx.Response:
        """Fetch URL con retry logic"""
        client = await self._get_client()
        response = await client.get(url)
        response.raise_for_status()
        return response
    
    def _extract_with_trafilatura(self, html: str, url: str) -> Dict[str, Any]:
        """Extraer contenido usando Trafilatura (mejor para artículos)"""
        return safe_extract(
            lambda: json.loads(trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=True,
                include_images=True,
                include_links=True,
                output_format='json'
            ) or '{}'),
            tool_name="Trafilatura"
        )
    
    def _extract_with_readability(self, html: str) -> Dict[str, Any]:
        """Extraer contenido usando Readability (limpieza de HTML)"""
        return safe_extract(
            lambda: {
                "title": Document(html).title(),
                "content": Document(html).summary(),
                "short_title": Document(html).short_title()
            },
            tool_name="Readability"
        )
    
    def _extract_with_newspaper(self, url: str, html: str) -> Dict[str, Any]:
        """Extraer contenido usando Newspaper3k (mejor para noticias)"""
        return safe_extract(
            lambda: self._build_newspaper_result(url, html),
            tool_name="Newspaper3k"
        )
    
    def _build_newspaper_result(self, url: str, html: str) -> Dict[str, Any]:
        """Construir resultado de Newspaper3k"""
        article = Article(url)
        article.set_html(html)
        article.parse()
        
        return {
            "title": article.title,
            "text": article.text,
            "authors": article.authors,
            "publish_date": article.publish_date.isoformat() if article.publish_date else None,
            "top_image": article.top_image,
            "images": article.images,
            "movies": article.movies,
            "keywords": article.keywords,
            "summary": article.summary
        }
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extraer contenido principal usando heurísticas"""
        # Intentar encontrar el contenido principal
        main_selectors = [
            'main',
            'article',
            '[role="main"]',
            '.content',
            '.post-content',
            '.entry-content',
            '#content',
            '.article-body',
            '.post-body'
        ]
        
        for selector in main_selectors:
            element = soup.select_one(selector)
            if element:
                # Remover elementos no deseados
                for tag in element.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    tag.decompose()
                text = element.get_text(separator=' ', strip=True)
                if len(text) > 200:  # Contenido significativo
                    return text
        
        # Fallback: usar todo el body
        body = soup.find('body')
        if body:
            for tag in body.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            return body.get_text(separator=' ', strip=True)
        
        return ""
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extraer tablas de la página"""
        tables = []
        for table in soup.find_all('table'):
            try:
                rows = []
                headers = []
                
                # Extraer headers
                thead = table.find('thead')
                if thead:
                    header_row = thead.find('tr')
                    if header_row:
                        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                
                # Extraer filas
                tbody = table.find('tbody') or table
                for tr in tbody.find_all('tr'):
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if cells:
                        rows.append(cells)
                
                if rows:
                    tables.append({
                        "headers": headers if headers else None,
                        "rows": rows,
                        "row_count": len(rows),
                        "column_count": len(headers) if headers else len(rows[0]) if rows else 0
                    })
            except Exception as e:
                logger.debug(f"Error extrayendo tabla: {e}")
                continue
        
        return tables
    
    def _extract_videos(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """Extraer videos de la página"""
        videos = []
        
        # Video tags HTML5
        for video in soup.find_all('video'):
            src = video.get('src', '')
            if not src:
                source = video.find('source')
                if source:
                    src = source.get('src', '')
            
            if src:
                videos.append({
                    "type": "html5",
                    "src": self._normalize_url(src, base_url),
                    "poster": self._normalize_url(video.get('poster', ''), base_url) if video.get('poster') else None,
                    "width": video.get('width'),
                    "height": video.get('height'),
                    "duration": video.get('duration')
                })
        
        # iframes (YouTube, Vimeo, etc.)
        for iframe in soup.find_all('iframe'):
            src = iframe.get('src', '')
            if src and any(domain in src.lower() for domain in ['youtube', 'vimeo', 'dailymotion', 'twitch']):
                videos.append({
                    "type": "embed",
                    "src": src,
                    "width": iframe.get('width'),
                    "height": iframe.get('height')
                })
        
        # Open Graph video
        og_video = soup.find('meta', property='og:video')
        if og_video:
            videos.append({
                "type": "og_video",
                "src": og_video.get('content', '')
            })
        
        return videos
    
    def _extract_quotes(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extraer citas y blockquotes"""
        quotes = []
        
        for blockquote in soup.find_all('blockquote'):
            text = blockquote.get_text(strip=True)
            cite = blockquote.get('cite', '')
            author = None
            
            # Buscar autor en footer o cite
            footer = blockquote.find('footer')
            if footer:
                author = footer.get_text(strip=True)
            cite_tag = blockquote.find('cite')
            if cite_tag:
                author = cite_tag.get_text(strip=True)
            
            if text:
                quotes.append({
                    "text": text,
                    "author": author,
                    "cite": cite
                })
        
        # Citas inline con <q>
        for q in soup.find_all('q'):
            text = q.get_text(strip=True)
            cite = q.get('cite', '')
            if text:
                quotes.append({
                    "text": text,
                    "cite": cite,
                    "type": "inline"
                })
        
        return quotes
    
    def _extract_code_blocks(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extraer bloques de código"""
        code_blocks = []
        
        for pre in soup.find_all('pre'):
            code = pre.find('code')
            if code:
                language = code.get('class', [])
                if language:
                    # Extraer lenguaje de class como 'language-python'
                    lang = [cls.replace('language-', '') for cls in language if 'language-' in cls]
                    language = lang[0] if lang else None
                else:
                    language = None
                
                code_blocks.append({
                    "code": code.get_text(),
                    "language": language,
                    "line_count": len(code.get_text().split('\n'))
                })
        
        return code_blocks
    
    def _extract_forms(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extraer formularios de la página"""
        forms = []
        
        for form in soup.find_all('form'):
            form_data = {
                "action": form.get('action', ''),
                "method": form.get('method', 'get').upper(),
                "fields": []
            }
            
            # Extraer campos
            for input_field in form.find_all(['input', 'textarea', 'select']):
                field_data = {
                    "type": input_field.get('type', input_field.name),
                    "name": input_field.get('name', ''),
                    "label": ""
                }
                
                # Buscar label asociado
                field_id = input_field.get('id', '')
                if field_id:
                    label = soup.find('label', {'for': field_id})
                    if label:
                        field_data["label"] = label.get_text(strip=True)
                
                # Label padre
                if not field_data["label"]:
                    parent_label = input_field.find_parent('label')
                    if parent_label:
                        field_data["label"] = parent_label.get_text(strip=True)
                
                form_data["fields"].append(field_data)
            
            if form_data["fields"]:
                forms.append(form_data)
        
        return forms
    
    def _extract_feeds(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """Extraer feeds RSS/Atom"""
        feeds = []
        
        # Buscar links de feeds
        for link in soup.find_all('link', type=['application/rss+xml', 'application/atom+xml', 'application/xml']):
            href = link.get('href', '')
            if href:
                feeds.append({
                    "type": link.get('type', ''),
                    "title": link.get('title', ''),
                    "url": self._normalize_url(href, base_url)
                })
        
        return feeds
    
    def _analyze_content_quality(self, text: str) -> Dict[str, Any]:
        """Analizar calidad del contenido extraído"""
        try:
            import textstat
            
            return {
                "word_count": textstat.lexicon_count(text),
                "sentence_count": textstat.sentence_count(text),
                "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
                "avg_sentence_length": textstat.avg_sentence_length(text),
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "automated_readability_index": textstat.automated_readability_index(text),
                "coleman_liau_index": textstat.coleman_liau_index(text)
            }
        except ImportError:
            # Fallback básico si textstat no está disponible
            words = text.split()
            sentences = text.split('.')
            return {
                "word_count": len(words),
                "sentence_count": len([s for s in sentences if s.strip()]),
                "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
                "avg_sentence_length": len(words) / len(sentences) if sentences else 0
            }
        except Exception as e:
            logger.debug(f"Error analizando calidad: {e}")
            return {}
    
    def _detect_language_advanced(self, text: str) -> Dict[str, Any]:
        """Detección avanzada de idioma"""
        try:
            from langdetect import detect, detect_langs, LangDetectException
            
            if not text or len(text.strip()) < 10:
                return {"language": "unknown", "confidence": 0.0}
            
            try:
                language = detect(text)
                languages = detect_langs(text)
                confidence = languages[0].prob if languages else 0.0
                
                return {
                    "language": language,
                    "confidence": confidence,
                    "all_detections": [
                        {"language": lang.lang, "confidence": lang.prob}
                        for lang in languages[:3]
                    ]
                }
            except LangDetectException:
                return {"language": "unknown", "confidence": 0.0}
        except ImportError:
            # Fallback básico
            return {"language": "en", "confidence": 0.5}
        except Exception as e:
            logger.debug(f"Error detectando idioma: {e}")
            return {"language": "unknown", "confidence": 0.0}
    
    async def scrape(
        self,
        url: str,
        use_javascript: bool = False,
        extract_strategy: str = "auto"
    ) -> Dict[str, Any]:
        """
        Extrae contenido de una página web usando múltiples estrategias
        
        Args:
            url: URL de la página a extraer
            use_javascript: Si True, usa Playwright para renderizar JavaScript (más lento)
            extract_strategy: Estrategia de extracción ("auto", "trafilatura", "readability", "newspaper", "beautifulsoup")
            
        Returns:
            Dict con contenido extraído y metadatos
        """
        # Verificar caché
        if self._cache and url in self._cache:
            logger.info(f"Retornando resultado desde caché para {url}")
            return self._cache[url]
        
        try:
            # Fetch HTML
            if use_javascript:
                html, _ = await self._fetch_with_playwright(url, take_screenshot=False)
            else:
                response = await self._fetch_url(url)
                encoding = self._detect_encoding(response.content)
                html = response.text if encoding == 'utf-8' else response.content.decode(encoding, errors='ignore')
            
            # Parse con BeautifulSoup
            soup = BeautifulSoup(html, 'lxml')
            
            # Extraer metadatos
            metadata = self._extract_metadata(soup, url)
            
            # Extraer datos estructurados
            structured_data = self._extract_structured_data(soup)
            
            # Determinar mejor estrategia de extracción
            if extract_strategy == "auto":
                # Heurística: detectar tipo de contenido
                if soup.find('article') or soup.find('main'):
                    extract_strategy = "trafilatura"
                elif 'news' in url.lower() or soup.find(class_=re.compile(r'news|article', re.I)):
                    extract_strategy = "newspaper"
                else:
                    extract_strategy = "trafilatura"
            
            # Extraer contenido según estrategia
            content_data = {}
            if extract_strategy == "trafilatura":
                content_data = self._extract_with_trafilatura(html, url)
            elif extract_strategy == "readability":
                content_data = self._extract_with_readability(html)
            elif extract_strategy == "newspaper":
                content_data = self._extract_with_newspaper(url, html)
            
            # Fallback a BeautifulSoup si otras estrategias fallan
            if not content_data.get('text') and not content_data.get('content'):
                main_content = self._extract_main_content(soup)
                content_data['content'] = main_content
                if not content_data.get('title'):
                    content_data['title'] = metadata.get('title', '')
            
            # Extraer enlaces e imágenes usando helpers
            links = extract_links(soup, url, limit=100)
            images = extract_images(soup, url, limit=50)
            
            # Extraer contenido adicional avanzado
            tables = self._extract_tables(soup)
            videos = self._extract_videos(soup, url)
            quotes = self._extract_quotes(soup)
            code_blocks = self._extract_code_blocks(soup)
            forms = self._extract_forms(soup)
            feeds = self._extract_feeds(soup, url)
            
            # Análisis de contenido
            main_content = content_data.get('text') or content_data.get('content', '')
            content_quality = self._analyze_content_quality(main_content)
            language_detection = self._detect_language_advanced(main_content)
            
            # Construir resultado usando helpers para valores con fallback
            result = {
                "url": url,
                "title": get_value_or_alternative(
                    content_data, "title", metadata, "title", ""
                ),
                "description": get_value_or_alternative(
                    content_data, "description", metadata, "description", ""
                ),
                "content": get_value_with_fallback(
                    [content_data], ["text", "content"], ""
                ),
                "author": self._extract_author(content_data, metadata),
                "published_date": get_value_or_alternative(
                    content_data, "publish_date", metadata, "published_date"
                ),
                "language": metadata.get('language', 'en'),
                "links": links[:100],  # Limitar a 100 enlaces
                "images": images[:50],  # Limitar a 50 imágenes
                "metadata": metadata,
                "structured_data": structured_data,
                "extraction_method": extract_strategy,
                "extracted_at": datetime.utcnow().isoformat(),
                "content_length": len(content_data.get('text') or content_data.get('content', '')),
                "links_count": len(links),
                "images_count": len(images),
                "tables": tables[:10],  # Limitar a 10 tablas
                "videos": videos[:10],  # Limitar a 10 videos
                "quotes": quotes[:20],  # Limitar a 20 citas
                "code_blocks": code_blocks[:20],  # Limitar a 20 bloques de código
                "forms": forms[:10],  # Limitar a 10 formularios
                "feeds": feeds,
                "content_quality": content_quality,
                "language_detection": language_detection
            }
            
            # Agregar campos específicos según el método usado
            if extract_strategy == "newspaper":
                result["keywords"] = content_data.get('keywords', [])
                result["top_image"] = content_data.get('top_image')
                result["summary"] = content_data.get('summary')
            elif extract_strategy == "trafilatura":
                result["raw_text"] = content_data.get('raw_text', '')
                result["comments"] = content_data.get('comments', '')
            
            # Guardar en caché
            if self._cache:
                self._cache[url] = result
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Error HTTP al hacer scraping de {url}: {e.response.status_code}")
            raise Exception(f"Error al acceder a la URL: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error al hacer scraping de {url}: {e}")
            raise Exception(f"Error al extraer contenido: {str(e)}")
    
    async def _fetch_with_playwright(self, url: str, take_screenshot: bool = False) -> Tuple[str, Optional[bytes]]:
        """Fetch URL usando Playwright para renderizar JavaScript"""
        try:
            from playwright.async_api import async_playwright
            
            screenshot_data = None
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, wait_until='networkidle', timeout=30000)
                html = await page.content()
                
                # Tomar screenshot si se solicita
                if take_screenshot:
                    screenshot_data = await page.screenshot(full_page=True, type='png')
                
                await browser.close()
                return html, screenshot_data
        except ImportError:
            logger.warning("Playwright no está instalado. Instalando con: pip install playwright && playwright install")
            raise Exception("Playwright no está disponible. Use use_javascript=False o instale Playwright.")
        except Exception as e:
            logger.error(f"Error con Playwright: {e}")
            raise
    
    async def scrape_batch(
        self,
        urls: List[str],
        use_javascript: bool = False,
        extract_strategy: str = "auto",
        max_concurrent: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Scrapear múltiples URLs en paralelo
        
        Args:
            urls: Lista de URLs a scrapear
            use_javascript: Si True, usa Playwright
            extract_strategy: Estrategia de extracción
            max_concurrent: Máximo de requests concurrentes
            
        Returns:
            Dict con resultados indexados por URL
        """
        import asyncio
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}
        
        async def scrape_with_semaphore(url: str):
            async with semaphore:
                try:
                    result = await self.scrape(url, use_javascript, extract_strategy)
                    return url, result
                except Exception as e:
                    logger.error(f"Error scrapeando {url}: {e}")
                    return url, {"error": str(e)}
        
        tasks = [scrape_with_semaphore(url) for url in urls]
        completed = await asyncio.gather(*tasks)
        
        for url, result in completed:
            results[url] = result
        
        return results


# Alias para compatibilidad hacia atrás
WebScraper = AdvancedWebScraper
