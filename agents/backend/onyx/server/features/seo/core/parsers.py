from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import time
from typing import Dict, List, Any, Optional
from loguru import logger
import cchardet
from .interfaces import HTMLParser
            from selectolax.parser import HTMLParser as SelectolaxParser
        from urllib.parse import urljoin, urlparse
        import orjson
            from lxml import html
            from lxml import html
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
HTML Parsers ultra-optimizados para el servicio SEO.
Implementaciones de máxima velocidad y eficiencia.
"""




class SelectolaxUltraParser(HTMLParser):
    """Parser ultra-rápido usando selectolax (más rápido que lxml)."""
    
    def __init__(self) -> Any:
        try:
            self.SelectolaxParser = SelectolaxParser
        except ImportError:
            logger.warning("Selectolax no disponible, usando fallback")
            self.SelectolaxParser = None
    
    def parse(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extrae información SEO usando selectolax (máxima velocidad)."""
        if not self.SelectolaxParser:
            return self._fallback_parse(html_content, url)
        
        seo_data = self._initialize_seo_data()
        
        try:
            # Usar selectolax para parsing ultra-rápido
            parser = self.SelectolaxParser(html_content)
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
    
    def get_parser_name(self) -> str:
        return "selectolax_ultra"
    
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
    
    def _extract_basic_info_selectolax(self, parser, seo_data: Dict[str, Any]):
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
    
    def _extract_meta_tags_selectolax(self, parser, seo_data: Dict[str, Any]):
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
    
    def _extract_headers_selectolax(self, parser, seo_data: Dict[str, Any]):
        """Extrae headers usando selectolax."""
        seo_data["h1_tags"] = [h.text().strip() for h in parser.css('h1')]
        seo_data["h2_tags"] = [h.text().strip() for h in parser.css('h2')]
        seo_data["h3_tags"] = [h.text().strip() for h in parser.css('h3')]
    
    def _extract_images_selectolax(self, parser, seo_data: Dict[str, Any]):
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
    
    def _extract_links_selectolax(self, parser, seo_data: Dict[str, Any], url: str):
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
    
    def _extract_content_selectolax(self, parser, seo_data: Dict[str, Any]):
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
    
    def _extract_structured_data_selectolax(self, parser, seo_data: Dict[str, Any]):
        """Extrae datos estructurados usando selectolax."""
        
        script_elements = parser.css('script[type="application/ld+json"]')
        for script in script_elements:
            try:
                data = orjson.loads(script.text())
                seo_data["structured_data"].append(data)
            except:
                continue
    
    def _calculate_performance_metrics_selectolax(self, parser, seo_data: Dict[str, Any]):
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


class LXMLFallbackParser(HTMLParser):
    """Parser de fallback usando lxml."""
    
    def parse(self, html_content: str, url: str) -> Dict[str, Any]:
        """Parsea HTML usando lxml como fallback."""
        try:
            tree = html.fromstring(html_content)
            return self._parse_lxml(tree, url)
        except Exception as e:
            logger.error(f"Error en parsing lxml: {e}")
            return self._initialize_seo_data()
    
    def get_parser_name(self) -> str:
        return "lxml_fallback"
    
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
    
    def _parse_lxml(self, tree, url: str) -> Dict[str, Any]:
        """Parsing completo con lxml."""
        seo_data = self._initialize_seo_data()
        
        # Extraer información básica
        title_elements = tree.xpath('//title/text()')
        if title_elements:
            seo_data["title"] = title_elements[0].strip()
        
        # Canonical URL
        canonical_elements = tree.xpath('//link[@rel="canonical"]/@href')
        if canonical_elements:
            seo_data["canonical_url"] = canonical_elements[0]
        
        # Meta tags
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
        
        # Headers
        seo_data["h1_tags"] = [h.strip() for h in tree.xpath('//h1/text()')]
        seo_data["h2_tags"] = [h.strip() for h in tree.xpath('//h2/text()')]
        seo_data["h3_tags"] = [h.strip() for h in tree.xpath('//h3/text()')]
        
        return seo_data


class ParserFactory:
    """Factory para crear parsers HTML."""
    
    @staticmethod
    def create_parser(parser_type: str = "auto") -> HTMLParser:
        """Crea un parser basado en el tipo especificado."""
        if parser_type == "selectolax":
            return SelectolaxUltraParser()
        elif parser_type == "lxml":
            return LXMLFallbackParser()
        else:
            # Auto-detect: intentar selectolax primero
            try:
                return SelectolaxUltraParser()
            except:
                return LXMLFallbackParser() 