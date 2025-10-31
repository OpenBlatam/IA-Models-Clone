from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from loguru import logger
import orjson
from selectolax.parser import HTMLParser, Node
from lxml import html, etree
import re
from urllib.parse import urljoin, urlparse
import zstandard as zstd
from .interfaces import HTMLParserInterface
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Parser ultra-optimizado usando las librerías más rápidas disponibles.
Selectolax + Orjson + LXML con fallback automático.
"""




@dataclass
class ParsedData:
    """Datos parseados ultra-optimizados."""
    title: str = ""
    meta_description: str = ""
    meta_keywords: str = ""
    h1_tags: List[str] = None
    h2_tags: List[str] = None
    h3_tags: List[str] = None
    links: List[Dict[str, str]] = None
    images: List[Dict[str, str]] = None
    scripts: List[Dict[str, str]] = None
    styles: List[Dict[str, str]] = None
    social_tags: Dict[str, str] = None
    structured_data: List[Dict[str, Any]] = None
    text_content: str = ""
    word_count: int = 0
    processing_time: float = 0.0
    parser_used: str = ""


class UltraOptimizedParser(HTMLParserInterface):
    """Parser ultra-optimizado con múltiples estrategias."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        self.selectolax_parser = None
        self.lxml_parser = None
        self.compressor = zstd.ZstdCompressor(level=3)
        self.decompressor = zstd.ZstdDecompressor()
        
        # Configuraciones
        self.max_content_size = self.config.get('max_content_size', 50 * 1024 * 1024)  # 50MB
        self.extract_text = self.config.get('extract_text', True)
        self.extract_links = self.config.get('extract_links', True)
        self.extract_images = self.config.get('extract_images', True)
        self.extract_scripts = self.config.get('extract_scripts', False)
        self.extract_styles = self.config.get('extract_styles', False)
        self.extract_structured_data = self.config.get('extract_structured_data', True)
        
        # Selectores optimizados
        self.selectors = {
            'title': 'title',
            'meta_description': 'meta[name="description"]',
            'meta_keywords': 'meta[name="keywords"]',
            'h1': 'h1',
            'h2': 'h2',
            'h3': 'h3',
            'links': 'a[href]',
            'images': 'img[src]',
            'scripts': 'script[src]',
            'styles': 'link[rel="stylesheet"]',
            'social_og': 'meta[property^="og:"]',
            'social_twitter': 'meta[name^="twitter:"]',
            'structured_data': 'script[type="application/ld+json"]'
        }
    
    def parse(self, html_content: str, base_url: str = "") -> ParsedData:
        """Parsea HTML ultra-optimizado con múltiples estrategias."""
        start_time = time.perf_counter()
        
        # Validar tamaño del contenido
        if len(html_content) > self.max_content_size:
            logger.warning(f"HTML content too large: {len(html_content)} bytes")
            html_content = html_content[:self.max_content_size]
        
        # Intentar con Selectolax primero (más rápido)
        try:
            parsed_data = self._parse_with_selectolax(html_content, base_url)
            parsed_data.parser_used = "selectolax"
        except Exception as e:
            logger.warning(f"Selectolax failed: {e}, falling back to LXML")
            try:
                parsed_data = self._parse_with_lxml(html_content, base_url)
                parsed_data.parser_used = "lxml"
            except Exception as e2:
                logger.error(f"LXML also failed: {e2}, using basic parsing")
                parsed_data = self._parse_basic(html_content, base_url)
                parsed_data.parser_used = "basic"
        
        # Calcular tiempo de procesamiento
        parsed_data.processing_time = time.perf_counter() - start_time
        
        return parsed_data
    
    def _parse_with_selectolax(self, html_content: str, base_url: str) -> ParsedData:
        """Parsea usando Selectolax (más rápido)."""
        tree = HTMLParser(html_content)
        
        # Extraer datos básicos
        title = self._extract_text_safe(tree.css_first('title'))
        meta_description = self._extract_attribute_safe(
            tree.css_first('meta[name="description"]'), 'content'
        )
        meta_keywords = self._extract_attribute_safe(
            tree.css_first('meta[name="keywords"]'), 'content'
        )
        
        # Extraer headers
        h1_tags = [h.text() for h in tree.css('h1') if h.text()]
        h2_tags = [h.text() for h in tree.css('h2') if h.text()]
        h3_tags = [h.text() for h in tree.css('h3') if h.text()]
        
        # Extraer links
        links = []
        if self.extract_links:
            for link in tree.css('a[href]'):
                href = link.attributes.get('href', '')
                if href:
                    full_url = urljoin(base_url, href)
                    links.append({
                        'url': full_url,
                        'text': link.text().strip()[:100],
                        'title': link.attributes.get('title', ''),
                        'rel': link.attributes.get('rel', '')
                    })
        
        # Extraer imágenes
        images = []
        if self.extract_images:
            for img in tree.css('img[src]'):
                src = img.attributes.get('src', '')
                if src:
                    full_url = urljoin(base_url, src)
                    images.append({
                        'src': full_url,
                        'alt': img.attributes.get('alt', ''),
                        'title': img.attributes.get('title', ''),
                        'width': img.attributes.get('width', ''),
                        'height': img.attributes.get('height', '')
                    })
        
        # Extraer scripts
        scripts = []
        if self.extract_scripts:
            for script in tree.css('script[src]'):
                src = script.attributes.get('src', '')
                if src:
                    full_url = urljoin(base_url, src)
                    scripts.append({
                        'src': full_url,
                        'type': script.attributes.get('type', ''),
                        'async': script.attributes.get('async', ''),
                        'defer': script.attributes.get('defer', '')
                    })
        
        # Extraer estilos
        styles = []
        if self.extract_styles:
            for style in tree.css('link[rel="stylesheet"]'):
                href = style.attributes.get('href', '')
                if href:
                    full_url = urljoin(base_url, href)
                    styles.append({
                        'href': full_url,
                        'media': style.attributes.get('media', ''),
                        'type': style.attributes.get('type', '')
                    })
        
        # Extraer tags sociales
        social_tags = self._extract_social_tags_selectolax(tree)
        
        # Extraer datos estructurados
        structured_data = []
        if self.extract_structured_data:
            for script in tree.css('script[type="application/ld+json"]'):
                try:
                    data = orjson.loads(script.text())
                    structured_data.append(data)
                except:
                    continue
        
        # Extraer contenido de texto
        text_content = ""
        word_count = 0
        if self.extract_text:
            text_content = self._extract_text_content_selectolax(tree)
            word_count = len(text_content.split())
        
        return ParsedData(
            title=title,
            meta_description=meta_description,
            meta_keywords=meta_keywords,
            h1_tags=h1_tags,
            h2_tags=h2_tags,
            h3_tags=h3_tags,
            links=links,
            images=images,
            scripts=scripts,
            styles=styles,
            social_tags=social_tags,
            structured_data=structured_data,
            text_content=text_content,
            word_count=word_count
        )
    
    def _parse_with_lxml(self, html_content: str, base_url: str) -> ParsedData:
        """Parsea usando LXML como fallback."""
        tree = html.fromstring(html_content)
        
        # Extraer datos básicos
        title = self._extract_text_safe_lxml(tree.xpath('//title'))
        meta_description = self._extract_attribute_safe_lxml(
            tree.xpath('//meta[@name="description"]/@content')
        )
        meta_keywords = self._extract_attribute_safe_lxml(
            tree.xpath('//meta[@name="keywords"]/@content')
        )
        
        # Extraer headers
        h1_tags = [h.text_content().strip() for h in tree.xpath('//h1') if h.text_content().strip()]
        h2_tags = [h.text_content().strip() for h in tree.xpath('//h2') if h.text_content().strip()]
        h3_tags = [h.text_content().strip() for h in tree.xpath('//h3') if h.text_content().strip()]
        
        # Extraer links
        links = []
        if self.extract_links:
            for link in tree.xpath('//a[@href]'):
                href = link.get('href', '')
                if href:
                    full_url = urljoin(base_url, href)
                    links.append({
                        'url': full_url,
                        'text': link.text_content().strip()[:100],
                        'title': link.get('title', ''),
                        'rel': link.get('rel', '')
                    })
        
        # Extraer imágenes
        images = []
        if self.extract_images:
            for img in tree.xpath('//img[@src]'):
                src = img.get('src', '')
                if src:
                    full_url = urljoin(base_url, src)
                    images.append({
                        'src': full_url,
                        'alt': img.get('alt', ''),
                        'title': img.get('title', ''),
                        'width': img.get('width', ''),
                        'height': img.get('height', '')
                    })
        
        # Extraer scripts
        scripts = []
        if self.extract_scripts:
            for script in tree.xpath('//script[@src]'):
                src = script.get('src', '')
                if src:
                    full_url = urljoin(base_url, src)
                    scripts.append({
                        'src': full_url,
                        'type': script.get('type', ''),
                        'async': script.get('async', ''),
                        'defer': script.get('defer', '')
                    })
        
        # Extraer estilos
        styles = []
        if self.extract_styles:
            for style in tree.xpath('//link[@rel="stylesheet"]'):
                href = style.get('href', '')
                if href:
                    full_url = urljoin(base_url, href)
                    styles.append({
                        'href': full_url,
                        'media': style.get('media', ''),
                        'type': style.get('type', '')
                    })
        
        # Extraer tags sociales
        social_tags = self._extract_social_tags_lxml(tree)
        
        # Extraer datos estructurados
        structured_data = []
        if self.extract_structured_data:
            for script in tree.xpath('//script[@type="application/ld+json"]'):
                try:
                    data = orjson.loads(script.text_content())
                    structured_data.append(data)
                except:
                    continue
        
        # Extraer contenido de texto
        text_content = ""
        word_count = 0
        if self.extract_text:
            text_content = self._extract_text_content_lxml(tree)
            word_count = len(text_content.split())
        
        return ParsedData(
            title=title,
            meta_description=meta_description,
            meta_keywords=meta_keywords,
            h1_tags=h1_tags,
            h2_tags=h2_tags,
            h3_tags=h3_tags,
            links=links,
            images=images,
            scripts=scripts,
            styles=styles,
            social_tags=social_tags,
            structured_data=structured_data,
            text_content=text_content,
            word_count=word_count
        )
    
    def _parse_basic(self, html_content: str, base_url: str) -> ParsedData:
        """Parsea básico usando regex como último recurso."""
        # Extraer título
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
        
        # Extraer meta description
        desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']', html_content, re.IGNORECASE)
        meta_description = desc_match.group(1) if desc_match else ""
        
        # Extraer meta keywords
        keywords_match = re.search(r'<meta[^>]*name=["\']keywords["\'][^>]*content=["\']([^"\']*)["\']', html_content, re.IGNORECASE)
        meta_keywords = keywords_match.group(1) if keywords_match else ""
        
        # Extraer headers básicos
        h1_tags = re.findall(r'<h1[^>]*>(.*?)</h1>', html_content, re.IGNORECASE | re.DOTALL)
        h2_tags = re.findall(r'<h2[^>]*>(.*?)</h2>', html_content, re.IGNORECASE | re.DOTALL)
        h3_tags = re.findall(r'<h3[^>]*>(.*?)</h3>', html_content, re.IGNORECASE | re.DOTALL)
        
        # Limpiar HTML tags
        h1_tags = [re.sub(r'<[^>]+>', '', h).strip() for h in h1_tags]
        h2_tags = [re.sub(r'<[^>]+>', '', h).strip() for h in h2_tags]
        h3_tags = [re.sub(r'<[^>]+>', '', h).strip() for h in h3_tags]
        
        return ParsedData(
            title=title,
            meta_description=meta_description,
            meta_keywords=meta_keywords,
            h1_tags=h1_tags,
            h2_tags=h2_tags,
            h3_tags=h3_tags,
            links=[],
            images=[],
            scripts=[],
            styles=[],
            social_tags={},
            structured_data=[],
            text_content="",
            word_count=0
        )
    
    def _extract_text_safe(self, node: Optional[Node]) -> str:
        """Extrae texto de forma segura de un nodo Selectolax."""
        if node and hasattr(node, 'text'):
            return node.text().strip()
        return ""
    
    def _extract_attribute_safe(self, node: Optional[Node], attr: str) -> str:
        """Extrae atributo de forma segura de un nodo Selectolax."""
        if node and hasattr(node, 'attributes'):
            return node.attributes.get(attr, "")
        return ""
    
    def _extract_text_safe_lxml(self, elements: List) -> str:
        """Extrae texto de forma segura de elementos LXML."""
        if elements and len(elements) > 0:
            return elements[0].strip()
        return ""
    
    def _extract_attribute_safe_lxml(self, attributes: List) -> str:
        """Extrae atributo de forma segura de elementos LXML."""
        if attributes and len(attributes) > 0:
            return attributes[0].strip()
        return ""
    
    def _extract_social_tags_selectolax(self, tree: HTMLParser) -> Dict[str, str]:
        """Extrae tags sociales usando Selectolax."""
        social_tags = {}
        
        # Open Graph tags
        for meta in tree.css('meta[property^="og:"]'):
            property_name = meta.attributes.get('property', '')
            content = meta.attributes.get('content', '')
            if property_name and content:
                social_tags[property_name] = content
        
        # Twitter tags
        for meta in tree.css('meta[name^="twitter:"]'):
            name = meta.attributes.get('name', '')
            content = meta.attributes.get('content', '')
            if name and content:
                social_tags[name] = content
        
        return social_tags
    
    def _extract_social_tags_lxml(self, tree) -> Dict[str, str]:
        """Extrae tags sociales usando LXML."""
        social_tags = {}
        
        # Open Graph tags
        for meta in tree.xpath('//meta[starts-with(@property, "og:")]'):
            property_name = meta.get('property', '')
            content = meta.get('content', '')
            if property_name and content:
                social_tags[property_name] = content
        
        # Twitter tags
        for meta in tree.xpath('//meta[starts-with(@name, "twitter:")]'):
            name = meta.get('name', '')
            content = meta.get('content', '')
            if name and content:
                social_tags[name] = content
        
        return social_tags
    
    def _extract_text_content_selectolax(self, tree: HTMLParser) -> str:
        """Extrae contenido de texto usando Selectolax."""
        # Remover scripts y estilos
        for script in tree.css('script'):
            script.decompose()
        for style in tree.css('style'):
            style.decompose()
        
        # Extraer texto del body
        body = tree.css_first('body')
        if body:
            return body.text().strip()
        
        return ""
    
    def _extract_text_content_lxml(self, tree) -> str:
        """Extrae contenido de texto usando LXML."""
        # Remover scripts y estilos
        for script in tree.xpath('//script'):
            script.getparent().remove(script)
        for style in tree.xpath('//style'):
            style.getparent().remove(style)
        
        # Extraer texto del body
        body_elements = tree.xpath('//body')
        if body_elements:
            return body_elements[0].text_content().strip()
        
        return ""
    
    def compress_data(self, data: Dict[str, Any]) -> bytes:
        """Comprime datos usando Zstandard."""
        return self.compressor.compress(orjson.dumps(data))
    
    def decompress_data(self, compressed_data: bytes) -> Dict[str, Any]:
        """Descomprime datos usando Zstandard."""
        return orjson.loads(self.decompressor.decompress(compressed_data))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de rendimiento del parser."""
        return {
            'parser_type': 'ultra_optimized',
            'max_content_size': self.max_content_size,
            'extract_text': self.extract_text,
            'extract_links': self.extract_links,
            'extract_images': self.extract_images,
            'extract_scripts': self.extract_scripts,
            'extract_styles': self.extract_styles,
            'extract_structured_data': self.extract_structured_data
        } 