from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import time
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from loguru import logger
import orjson
import zstandard as zstd
import lz4.frame
import snappy
from selectolax.parser import HTMLParser as SelectolaxParser
from lxml import html, etree
import regex as re
from urllib.parse import urljoin, urlparse
import numba
from concurrent.futures import ThreadPoolExecutor
import threading
from .interfaces import HTMLParserInterface, ParsedData
        import hashlib
from typing import Any, List, Dict, Optional
import logging
"""
Ultra-Optimized HTML Parser v3.0
Using the fastest parsing libraries available: Selectolax + LXML + Zstandard + Numba
"""




@dataclass
class ParsingStats:
    """Estadísticas de parsing ultra-optimizado v3."""
    parser_used: str = ""
    parsing_time: float = 0.0
    compression_ratio: float = 0.0
    original_size: int = 0
    compressed_size: int = 0
    elements_found: int = 0
    cache_hit: bool = False
    compression_algorithm: str = ""


class UltraOptimizedParserV3(HTMLParserInterface):
    """Parser ultra-optimizado v3 con las librerías más rápidas."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        
        # Configuraciones de parsing
        self.primary_parser = self.config.get('primary_parser', 'selectolax')
        self.fallback_parser = self.config.get('fallback_parser', 'lxml')
        self.timeout = self.config.get('timeout', 10.0)
        self.max_size = self.config.get('max_size', 100 * 1024 * 1024)  # 100MB
        self.enable_compression = self.config.get('enable_compression', True)
        self.compression_algorithm = self.config.get('compression_algorithm', 'zstandard')
        self.compression_level = self.config.get('compression_level', 3)
        
        # Configuraciones de extracción
        self.extract_metadata = self.config.get('extract_metadata', True)
        self.extract_links = self.config.get('extract_links', True)
        self.extract_images = self.config.get('extract_images', True)
        self.extract_text = self.config.get('extract_text', True)
        self.extract_headers = self.config.get('extract_headers', True)
        self.extract_forms = self.config.get('extract_forms', True)
        
        # Configuraciones de optimización
        self.enable_parallel_processing = self.config.get('enable_parallel_processing', True)
        self.max_workers = self.config.get('max_workers', 4)
        self.enable_numba_optimization = self.config.get('enable_numba_optimization', True)
        
        # Compresores
        if self.enable_compression:
            self._init_compressors()
        
        # Cache de parsing
        self.parsing_cache = {}
        self.cache_size = self.config.get('cache_size', 2000)
        self.cache_lock = threading.RLock()
        
        # Thread pool para procesamiento paralelo
        if self.enable_parallel_processing:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Estadísticas
        self.stats = {
            'total_parses': 0,
            'cache_hits': 0,
            'selectolax_parses': 0,
            'lxml_parses': 0,
            'compression_savings': 0,
            'average_parse_time': 0.0,
            'parallel_processing_used': 0
        }
        
        # Compilar funciones Numba
        if self.enable_numba_optimization:
            self._compile_numba_functions()
        
        logger.info("Ultra-Optimized Parser v3.0 initialized")
    
    def _init_compressors(self) -> Any:
        """Inicializar compresores."""
        self.compressors = {
            'zstandard': {
                'compressor': zstd.ZstdCompressor(level=self.compression_level),
                'decompressor': zstd.ZstdDecompressor()
            },
            'lz4': {
                'compressor': lz4.frame,
                'decompressor': lz4.frame
            },
            'snappy': {
                'compressor': snappy,
                'decompressor': snappy
            }
        }
    
    def _compile_numba_functions(self) -> Any:
        """Compilar funciones optimizadas con Numba."""
        @numba.jit(nopython=True, cache=True)
        def fast_text_clean(text) -> Any:
            """Limpieza rápida de texto con Numba."""
            result = ""
            for char in text:
                if char.isprintable() and char != '\x00':
                    result += char
            return result
        
        @numba.jit(nopython=True, cache=True)
        def fast_word_count(text) -> Any:
            """Conteo rápido de palabras con Numba."""
            count = 0
            in_word = False
            for char in text:
                if char.isalnum():
                    if not in_word:
                        count += 1
                        in_word = True
                else:
                    in_word = False
            return count
        
        self.fast_text_clean = fast_text_clean
        self.fast_word_count = fast_word_count
    
    def parse(self, html_content: str, url: Optional[str] = None) -> ParsedData:
        """Parse HTML ultra-optimizado v3 con fallback automático."""
        start_time = time.perf_counter()
        
        # Validar entrada
        if not html_content or len(html_content) > self.max_size:
            raise ValueError(f"Invalid HTML content: size={len(html_content) if html_content else 0}")
        
        # Generar clave de cache
        cache_key = self._generate_cache_key(html_content, url)
        
        # Verificar cache con lock
        with self.cache_lock:
            if cache_key in self.parsing_cache:
                self.stats['cache_hits'] += 1
                cached_result = self.parsing_cache[cache_key]
                cached_result.stats.cache_hit = True
                return cached_result
        
        # Intentar parsing con Selectolax (más rápido)
        try:
            parsed_data = self._parse_with_selectolax(html_content, url)
            self.stats['selectolax_parses'] += 1
            parser_used = 'selectolax'
        except Exception as e:
            logger.warning(f"Selectolax parsing failed: {e}, falling back to LXML")
            try:
                parsed_data = self._parse_with_lxml(html_content, url)
                self.stats['lxml_parses'] += 1
                parser_used = 'lxml'
            except Exception as e2:
                logger.error(f"LXML parsing also failed: {e2}")
                raise ValueError(f"All parsers failed: {e}, {e2}")
        
        # Calcular estadísticas
        parsing_time = time.perf_counter() - start_time
        elements_found = self._count_elements(parsed_data)
        
        # Comprimir datos si está habilitado
        original_size = len(str(parsed_data))
        compressed_size = original_size
        compression_algorithm = "none"
        
        if self.enable_compression and original_size > 1024:
            try:
                compressed_data, compression_algorithm = self._compress_data(parsed_data)
                compressed_size = len(compressed_data)
                compression_ratio = (original_size - compressed_size) / original_size
            except Exception as e:
                logger.warning(f"Compression failed: {e}")
                compression_ratio = 0.0
        else:
            compression_ratio = 0.0
        
        # Crear estadísticas
        stats = ParsingStats(
            parser_used=parser_used,
            parsing_time=parsing_time,
            compression_ratio=compression_ratio,
            original_size=original_size,
            compressed_size=compressed_size,
            elements_found=elements_found,
            cache_hit=False,
            compression_algorithm=compression_algorithm
        )
        
        parsed_data.stats = stats
        
        # Actualizar estadísticas globales
        self.stats['total_parses'] += 1
        self.stats['compression_savings'] += compression_ratio
        self.stats['average_parse_time'] = (
            (self.stats['average_parse_time'] * (self.stats['total_parses'] - 1) + parsing_time) 
            / self.stats['total_parses']
        )
        
        # Guardar en cache con lock
        with self.cache_lock:
            self._cache_result(cache_key, parsed_data)
        
        logger.debug(f"Parsed {len(html_content)} bytes in {parsing_time:.3f}s using {parser_used}")
        
        return parsed_data
    
    def _parse_with_selectolax(self, html_content: str, url: Optional[str] = None) -> ParsedData:
        """Parse usando Selectolax (más rápido) con optimizaciones."""
        # Parse con Selectolax
        parser = SelectolaxParser(html_content)
        
        # Extraer datos básicos
        title = self._extract_title_selectolax(parser)
        meta_description = self._extract_meta_description_selectolax(parser)
        meta_keywords = self._extract_meta_keywords_selectolax(parser)
        
        # Extraer headers en paralelo si está habilitado
        if self.enable_parallel_processing:
            header_futures = []
            for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                future = self.thread_pool.submit(self._extract_headers_selectolax, parser, tag)
                header_futures.append((tag, future))
            
            headers = {}
            for tag, future in header_futures:
                headers[tag] = future.result()
            
            h1_tags = headers['h1']
            h2_tags = headers['h2']
            h3_tags = headers['h3']
            h4_tags = headers['h4']
            h5_tags = headers['h5']
            h6_tags = headers['h6']
            
            self.stats['parallel_processing_used'] += 1
        else:
            h1_tags = self._extract_headers_selectolax(parser, 'h1')
            h2_tags = self._extract_headers_selectolax(parser, 'h2')
            h3_tags = self._extract_headers_selectolax(parser, 'h3')
            h4_tags = self._extract_headers_selectolax(parser, 'h4')
            h5_tags = self._extract_headers_selectolax(parser, 'h5')
            h6_tags = self._extract_headers_selectolax(parser, 'h6')
        
        # Extraer links
        links = self._extract_links_selectolax(parser, url)
        
        # Extraer imágenes
        images = self._extract_images_selectolax(parser, url)
        
        # Extraer texto con optimización Numba
        text_content = self._extract_text_selectolax(parser)
        
        # Extraer formularios
        forms = self._extract_forms_selectolax(parser)
        
        # Extraer metadatos adicionales
        metadata = self._extract_metadata_selectolax(parser)
        
        return ParsedData(
            title=title,
            meta_description=meta_description,
            meta_keywords=meta_keywords,
            h1_tags=h1_tags,
            h2_tags=h2_tags,
            h3_tags=h3_tags,
            h4_tags=h4_tags,
            h5_tags=h5_tags,
            h6_tags=h6_tags,
            links=links,
            images=images,
            text_content=text_content,
            forms=forms,
            metadata=metadata,
            url=url,
            parser_used='selectolax'
        )
    
    def _parse_with_lxml(self, html_content: str, url: Optional[str] = None) -> ParsedData:
        """Parse usando LXML (fallback) con optimizaciones."""
        # Parse con LXML
        tree = html.fromstring(html_content)
        
        # Extraer datos básicos
        title = self._extract_title_lxml(tree)
        meta_description = self._extract_meta_description_lxml(tree)
        meta_keywords = self._extract_meta_keywords_lxml(tree)
        
        # Extraer headers en paralelo si está habilitado
        if self.enable_parallel_processing:
            header_futures = []
            for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                future = self.thread_pool.submit(self._extract_headers_lxml, tree, tag)
                header_futures.append((tag, future))
            
            headers = {}
            for tag, future in header_futures:
                headers[tag] = future.result()
            
            h1_tags = headers['h1']
            h2_tags = headers['h2']
            h3_tags = headers['h3']
            h4_tags = headers['h4']
            h5_tags = headers['h5']
            h6_tags = headers['h6']
            
            self.stats['parallel_processing_used'] += 1
        else:
            h1_tags = self._extract_headers_lxml(tree, 'h1')
            h2_tags = self._extract_headers_lxml(tree, 'h2')
            h3_tags = self._extract_headers_lxml(tree, 'h3')
            h4_tags = self._extract_headers_lxml(tree, 'h4')
            h5_tags = self._extract_headers_lxml(tree, 'h5')
            h6_tags = self._extract_headers_lxml(tree, 'h6')
        
        # Extraer links
        links = self._extract_links_lxml(tree, url)
        
        # Extraer imágenes
        images = self._extract_images_lxml(tree, url)
        
        # Extraer texto con optimización Numba
        text_content = self._extract_text_lxml(tree)
        
        # Extraer formularios
        forms = self._extract_forms_lxml(tree)
        
        # Extraer metadatos adicionales
        metadata = self._extract_metadata_lxml(tree)
        
        return ParsedData(
            title=title,
            meta_description=meta_description,
            meta_keywords=meta_keywords,
            h1_tags=h1_tags,
            h2_tags=h2_tags,
            h3_tags=h3_tags,
            h4_tags=h4_tags,
            h5_tags=h5_tags,
            h6_tags=h6_tags,
            links=links,
            images=images,
            text_content=text_content,
            forms=forms,
            metadata=metadata,
            url=url,
            parser_used='lxml'
        )
    
    def _extract_title_selectolax(self, parser: SelectolaxParser) -> str:
        """Extraer título usando Selectolax."""
        title_elem = parser.css_first('title')
        return title_elem.text() if title_elem else ""
    
    def _extract_meta_description_selectolax(self, parser: SelectolaxParser) -> str:
        """Extraer meta descripción usando Selectolax."""
        meta_desc = parser.css_first('meta[name="description"]')
        return meta_desc.attributes.get('content', '') if meta_desc else ""
    
    def _extract_meta_keywords_selectolax(self, parser: SelectolaxParser) -> str:
        """Extraer meta keywords usando Selectolax."""
        meta_keywords = parser.css_first('meta[name="keywords"]')
        return meta_keywords.attributes.get('content', '') if meta_keywords else ""
    
    def _extract_headers_selectolax(self, parser: SelectolaxParser, tag: str) -> List[str]:
        """Extraer headers usando Selectolax."""
        headers = parser.css(tag)
        return [h.text() for h in headers if h.text()]
    
    def _extract_links_selectolax(self, parser: SelectolaxParser, base_url: Optional[str]) -> List[Dict[str, str]]:
        """Extraer links usando Selectolax con optimización."""
        links = []
        for link in parser.css('a[href]'):
            href = link.attributes.get('href', '')
            text = link.text() or ""
            
            if href and not href.startswith(('#', 'javascript:', 'mailto:')):
                if base_url and not href.startswith(('http://', 'https://')):
                    href = urljoin(base_url, href)
                
                links.append({
                    'url': href,
                    'text': text,
                    'title': link.attributes.get('title', ''),
                    'rel': link.attributes.get('rel', '')
                })
        
        return links
    
    def _extract_images_selectolax(self, parser: SelectolaxParser, base_url: Optional[str]) -> List[Dict[str, str]]:
        """Extraer imágenes usando Selectolax con optimización."""
        images = []
        for img in parser.css('img[src]'):
            src = img.attributes.get('src', '')
            alt = img.attributes.get('alt', '')
            
            if src and not src.startswith('data:'):
                if base_url and not src.startswith(('http://', 'https://')):
                    src = urljoin(base_url, src)
                
                images.append({
                    'src': src,
                    'alt': alt,
                    'title': img.attributes.get('title', ''),
                    'width': img.attributes.get('width', ''),
                    'height': img.attributes.get('height', '')
                })
        
        return images
    
    def _extract_text_selectolax(self, parser: SelectolaxParser) -> str:
        """Extraer texto usando Selectolax con optimización Numba."""
        # Remover scripts y styles
        for script in parser.css('script, style'):
            script.decompose()
        
        # Obtener texto del body
        body = parser.css_first('body')
        if body:
            text = body.text()
        else:
            text = parser.text()
        
        # Optimizar con Numba si está habilitado
        if self.enable_numba_optimization:
            text = self.fast_text_clean(text)
        
        return text
    
    def _extract_forms_selectolax(self, parser: SelectolaxParser) -> List[Dict[str, Any]]:
        """Extraer formularios usando Selectolax."""
        forms = []
        for form in parser.css('form'):
            form_data = {
                'action': form.attributes.get('action', ''),
                'method': form.attributes.get('method', 'get'),
                'inputs': []
            }
            
            for input_elem in form.css('input'):
                input_data = {
                    'type': input_elem.attributes.get('type', 'text'),
                    'name': input_elem.attributes.get('name', ''),
                    'value': input_elem.attributes.get('value', ''),
                    'placeholder': input_elem.attributes.get('placeholder', '')
                }
                form_data['inputs'].append(input_data)
            
            forms.append(form_data)
        
        return forms
    
    def _extract_metadata_selectolax(self, parser: SelectolaxParser) -> Dict[str, str]:
        """Extraer metadatos adicionales usando Selectolax."""
        metadata = {}
        
        # Meta tags
        for meta in parser.css('meta'):
            name = meta.attributes.get('name', '')
            property_attr = meta.attributes.get('property', '')
            content = meta.attributes.get('content', '')
            
            if name:
                metadata[f'meta:{name}'] = content
            elif property_attr:
                metadata[f'og:{property_attr}'] = content
        
        # Canonical URL
        canonical = parser.css_first('link[rel="canonical"]')
        if canonical:
            metadata['canonical'] = canonical.attributes.get('href', '')
        
        return metadata
    
    # Métodos LXML optimizados
    def _extract_title_lxml(self, tree) -> str:
        """Extraer título usando LXML."""
        title_elem = tree.xpath('//title/text()')
        return title_elem[0] if title_elem else ""
    
    def _extract_meta_description_lxml(self, tree) -> str:
        """Extraer meta descripción usando LXML."""
        meta_desc = tree.xpath('//meta[@name="description"]/@content')
        return meta_desc[0] if meta_desc else ""
    
    def _extract_meta_keywords_lxml(self, tree) -> str:
        """Extraer meta keywords usando LXML."""
        meta_keywords = tree.xpath('//meta[@name="keywords"]/@content')
        return meta_keywords[0] if meta_keywords else ""
    
    def _extract_headers_lxml(self, tree, tag: str) -> List[str]:
        """Extraer headers usando LXML."""
        headers = tree.xpath(f'//{tag}/text()')
        return [h.strip() for h in headers if h.strip()]
    
    def _extract_links_lxml(self, tree, base_url: Optional[str]) -> List[Dict[str, str]]:
        """Extraer links usando LXML con optimización."""
        links = []
        for link in tree.xpath('//a[@href]'):
            href = link.get('href', '')
            text = ''.join(link.xpath('.//text()')).strip()
            
            if href and not href.startswith(('#', 'javascript:', 'mailto:')):
                if base_url and not href.startswith(('http://', 'https://')):
                    href = urljoin(base_url, href)
                
                links.append({
                    'url': href,
                    'text': text,
                    'title': link.get('title', ''),
                    'rel': link.get('rel', '')
                })
        
        return links
    
    def _extract_images_lxml(self, tree, base_url: Optional[str]) -> List[Dict[str, str]]:
        """Extraer imágenes usando LXML con optimización."""
        images = []
        for img in tree.xpath('//img[@src]'):
            src = img.get('src', '')
            alt = img.get('alt', '')
            
            if src and not src.startswith('data:'):
                if base_url and not src.startswith(('http://', 'https://')):
                    src = urljoin(base_url, src)
                
                images.append({
                    'src': src,
                    'alt': alt,
                    'title': img.get('title', ''),
                    'width': img.get('width', ''),
                    'height': img.get('height', '')
                })
        
        return images
    
    def _extract_text_lxml(self, tree) -> str:
        """Extraer texto usando LXML con optimización Numba."""
        # Remover scripts y styles
        for elem in tree.xpath('//script | //style'):
            elem.getparent().remove(elem)
        
        # Obtener texto del body
        body_text = tree.xpath('//body//text()')
        if body_text:
            text = ' '.join(text.strip() for text in body_text if text.strip())
        else:
            text = ' '.join(tree.xpath('//text()'))
        
        # Optimizar con Numba si está habilitado
        if self.enable_numba_optimization:
            text = self.fast_text_clean(text)
        
        return text
    
    def _extract_forms_lxml(self, tree) -> List[Dict[str, Any]]:
        """Extraer formularios usando LXML."""
        forms = []
        for form in tree.xpath('//form'):
            form_data = {
                'action': form.get('action', ''),
                'method': form.get('method', 'get'),
                'inputs': []
            }
            
            for input_elem in form.xpath('.//input'):
                input_data = {
                    'type': input_elem.get('type', 'text'),
                    'name': input_elem.get('name', ''),
                    'value': input_elem.get('value', ''),
                    'placeholder': input_elem.get('placeholder', '')
                }
                form_data['inputs'].append(input_data)
            
            forms.append(form_data)
        
        return forms
    
    def _extract_metadata_lxml(self, tree) -> Dict[str, str]:
        """Extraer metadatos adicionales usando LXML."""
        metadata = {}
        
        # Meta tags
        for meta in tree.xpath('//meta'):
            name = meta.get('name', '')
            property_attr = meta.get('property', '')
            content = meta.get('content', '')
            
            if name:
                metadata[f'meta:{name}'] = content
            elif property_attr:
                metadata[f'og:{property_attr}'] = content
        
        # Canonical URL
        canonical = tree.xpath('//link[@rel="canonical"]/@href')
        if canonical:
            metadata['canonical'] = canonical[0]
        
        return metadata
    
    def _generate_cache_key(self, html_content: str, url: Optional[str]) -> str:
        """Generar clave de cache."""
        content_hash = hashlib.md5(html_content.encode()).hexdigest()
        url_hash = hashlib.md5((url or '').encode()).hexdigest()
        return f"{content_hash}:{url_hash}"
    
    def _cache_result(self, cache_key: str, parsed_data: ParsedData):
        """Guardar resultado en cache."""
        if len(self.parsing_cache) >= self.cache_size:
            # Remover elemento más antiguo
            oldest_key = next(iter(self.parsing_cache))
            del self.parsing_cache[oldest_key]
        
        self.parsing_cache[cache_key] = parsed_data
    
    def _compress_data(self, data: Any) -> tuple:
        """Comprimir datos usando múltiples algoritmos."""
        json_data = orjson.dumps(data)
        
        if self.compression_algorithm == 'zstandard':
            compressed_data = self.compressors['zstandard']['compressor'].compress(json_data)
        elif self.compression_algorithm == 'lz4':
            compressed_data = self.compressors['lz4']['compressor'].compress(json_data)
        elif self.compression_algorithm == 'snappy':
            compressed_data = self.compressors['snappy']['compressor'].compress(json_data)
        else:
            compressed_data = json_data
        
        return compressed_data, self.compression_algorithm
    
    def _decompress_data(self, compressed_data: bytes, algorithm: str) -> Any:
        """Descomprimir datos usando múltiples algoritmos."""
        if algorithm == 'zstandard':
            json_data = self.compressors['zstandard']['decompressor'].decompress(compressed_data)
        elif algorithm == 'lz4':
            json_data = self.compressors['lz4']['decompressor'].decompress(compressed_data)
        elif algorithm == 'snappy':
            json_data = self.compressors['snappy']['decompressor'].decompress(compressed_data)
        else:
            json_data = compressed_data
        
        return orjson.loads(json_data)
    
    def _count_elements(self, parsed_data: ParsedData) -> int:
        """Contar elementos extraídos."""
        count = 0
        count += len(parsed_data.h1_tags)
        count += len(parsed_data.h2_tags)
        count += len(parsed_data.h3_tags)
        count += len(parsed_data.h4_tags)
        count += len(parsed_data.h5_tags)
        count += len(parsed_data.h6_tags)
        count += len(parsed_data.links)
        count += len(parsed_data.images)
        count += len(parsed_data.forms)
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del parser."""
        return {
            'parser_type': 'ultra_optimized_v3',
            'total_parses': self.stats['total_parses'],
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_ratio': self.stats['cache_hits'] / max(self.stats['total_parses'], 1),
            'selectolax_parses': self.stats['selectolax_parses'],
            'lxml_parses': self.stats['lxml_parses'],
            'average_parse_time': self.stats['average_parse_time'],
            'compression_savings': self.stats['compression_savings'],
            'parallel_processing_used': self.stats['parallel_processing_used'],
            'cache_size': len(self.parsing_cache),
            'max_cache_size': self.cache_size,
            'compression_enabled': self.enable_compression,
            'compression_algorithm': self.compression_algorithm,
            'compression_level': self.compression_level,
            'enable_parallel_processing': self.enable_parallel_processing,
            'enable_numba_optimization': self.enable_numba_optimization,
            'max_workers': self.max_workers
        }
    
    def clear_cache(self) -> Any:
        """Limpiar cache."""
        with self.cache_lock:
            self.parsing_cache.clear()
        logger.info("Parser cache cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Health check del parser."""
        return {
            'status': 'healthy',
            'parser_type': 'ultra_optimized_v3',
            'cache_size': len(self.parsing_cache),
            'compression_enabled': self.enable_compression,
            'compression_algorithm': self.compression_algorithm,
            'parallel_processing_enabled': self.enable_parallel_processing,
            'numba_optimization_enabled': self.enable_numba_optimization,
            'total_parses': self.stats['total_parses']
        }
    
    def __del__(self) -> Any:
        """Cleanup al destruir el objeto."""
        if hasattr(self, 'thread_pool') and self.thread_pool:
            self.thread_pool.shutdown(wait=True) 