from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import time
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from loguru import logger
import orjson
import zstandard as zstd
from concurrent.futures import ThreadPoolExecutor
import threading
from ..core.ultra_optimized_parser_v2 import UltraOptimizedParserV2
from ..core.ultra_optimized_cache_v2 import UltraOptimizedCacheV2
from ..core.ultra_optimized_http_client_v2 import UltraOptimizedHTTPClientV2
from ..core.interfaces import ParsedData
from typing import Any, List, Dict, Optional
import logging
"""
Ultra-Optimized SEO Service v2.0
Using the fastest libraries and modular architecture
"""




@dataclass
class SEOAnalysisResult:
    """Resultado de análisis SEO ultra-optimizado."""
    url: str
    title: str
    meta_description: str
    meta_keywords: str
    h1_tags: List[str]
    h2_tags: List[str]
    h3_tags: List[str]
    h4_tags: List[str]
    h5_tags: List[str]
    h6_tags: List[str]
    links: List[Dict[str, str]]
    images: List[Dict[str, str]]
    text_content: str
    forms: List[Dict[str, Any]]
    metadata: Dict[str, str]
    
    # Métricas SEO
    seo_score: float
    title_score: float
    description_score: float
    keyword_score: float
    structure_score: float
    content_score: float
    technical_score: float
    
    # Recomendaciones
    recommendations: List[str]
    warnings: List[str]
    errors: List[str]
    
    # Estadísticas
    word_count: int
    character_count: int
    link_count: int
    image_count: int
    form_count: int
    
    # Performance
    load_time: float
    parsing_time: float
    analysis_time: float
    total_time: float
    
    # Cache info
    cached: bool = False
    parser_used: str = ""


@dataclass
class SEOServiceStats:
    """Estadísticas del servicio SEO ultra-optimizado."""
    total_analyses: int = 0
    successful_analyses: int = 0
    failed_analyses: int = 0
    average_analysis_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    total_urls_processed: int = 0
    total_words_analyzed: int = 0
    total_links_found: int = 0
    total_images_found: int = 0


class UltraOptimizedSEOServiceV2:
    """Servicio SEO ultra-optimizado con arquitectura modular."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        
        # Configuraciones del servicio
        self.max_concurrent_analyses = self.config.get('max_concurrent_analyses', 10)
        self.analysis_timeout = self.config.get('analysis_timeout', 30.0)
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hora
        self.enable_compression = self.config.get('enable_compression', True)
        self.max_content_size = self.config.get('max_content_size', 10 * 1024 * 1024)  # 10MB
        
        # Configuraciones de análisis
        self.min_title_length = self.config.get('min_title_length', 30)
        self.max_title_length = self.config.get('max_title_length', 60)
        self.min_description_length = self.config.get('min_description_length', 120)
        self.max_description_length = self.config.get('max_description_length', 160)
        self.min_content_length = self.config.get('min_content_length', 300)
        
        # Inicializar componentes
        self._init_components()
        
        # Thread pool para operaciones CPU-intensivas
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 4),
            thread_name_prefix="SEO-Analysis"
        )
        
        # Semáforo para limitar análisis concurrentes
        self.analysis_semaphore = asyncio.Semaphore(self.max_concurrent_analyses)
        
        # Estadísticas
        self.stats = SEOServiceStats()
        self.lock = threading.Lock()
        
        logger.info("Ultra-Optimized SEO Service v2.0 initialized")
    
    def _init_components(self) -> Any:
        """Inicializar componentes del servicio."""
        # Parser ultra-optimizado
        parser_config = self.config.get('parser', {})
        self.parser = UltraOptimizedParserV2(parser_config)
        
        # Cache ultra-optimizado
        cache_config = self.config.get('cache', {})
        self.cache = UltraOptimizedCacheV2(cache_config)
        
        # HTTP client ultra-optimizado
        http_config = self.config.get('http_client', {})
        self.http_client = UltraOptimizedHTTPClientV2(http_config)
        
        logger.info("SEO service components initialized")
    
    async def analyze_url(self, url: str, force_refresh: bool = False) -> SEOAnalysisResult:
        """Analizar URL con optimizaciones ultra-optimizadas."""
        start_time = time.perf_counter()
        
        try:
            # Rate limiting
            async with self.analysis_semaphore:
                return await self._perform_analysis(url, force_refresh, start_time)
                
        except Exception as e:
            logger.error(f"SEO analysis failed for {url}: {e}")
            self._update_stats(False)
            raise
    
    async def analyze_urls_batch(self, urls: List[str], 
                                max_concurrent: Optional[int] = None) -> List[SEOAnalysisResult]:
        """Analizar múltiples URLs en paralelo."""
        if max_concurrent is None:
            max_concurrent = min(len(urls), self.max_concurrent_analyses)
        
        # Crear semáforo para este batch
        batch_semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_single_url(url: str) -> SEOAnalysisResult:
            async with batch_semaphore:
                return await self.analyze_url(url)
        
        # Ejecutar análisis en paralelo
        tasks = [analyze_single_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar resultados
        analysis_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch analysis failed for {urls[i]}: {result}")
                analysis_results.append(None)
            else:
                analysis_results.append(result)
        
        return analysis_results
    
    async def _perform_analysis(self, url: str, force_refresh: bool, start_time: float) -> SEOAnalysisResult:
        """Realizar análisis SEO completo."""
        # Verificar cache
        if not force_refresh and self.enable_caching:
            cache_key = f"seo_analysis:{url}"
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self.stats.cache_hits += 1
                cached_result.cached = True
                return cached_result
        
        self.stats.cache_misses += 1
        
        # Obtener contenido
        fetch_start = time.perf_counter()
        try:
            response = await self.http_client.get(url)
            fetch_time = time.perf_counter() - fetch_start
            
            if response.status_code != 200:
                raise ValueError(f"HTTP {response.status_code}: {response.text}")
            
            # Verificar tamaño del contenido
            if len(response.content) > self.max_content_size:
                raise ValueError(f"Content too large: {len(response.content)} bytes")
            
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise
        
        # Parsear HTML
        parse_start = time.perf_counter()
        try:
            parsed_data = self.parser.parse(response.text, url)
            parse_time = time.perf_counter() - parse_start
        except Exception as e:
            logger.error(f"Failed to parse {url}: {e}")
            raise
        
        # Analizar SEO
        analysis_start = time.perf_counter()
        try:
            seo_result = await self._analyze_seo(parsed_data, url)
            analysis_time = time.perf_counter() - analysis_start
        except Exception as e:
            logger.error(f"Failed to analyze SEO for {url}: {e}")
            raise
        
        # Crear resultado final
        total_time = time.perf_counter() - start_time
        
        result = SEOAnalysisResult(
            url=url,
            title=parsed_data.title,
            meta_description=parsed_data.meta_description,
            meta_keywords=parsed_data.meta_keywords,
            h1_tags=parsed_data.h1_tags,
            h2_tags=parsed_data.h2_tags,
            h3_tags=parsed_data.h3_tags,
            h4_tags=parsed_data.h4_tags,
            h5_tags=parsed_data.h5_tags,
            h6_tags=parsed_data.h6_tags,
            links=parsed_data.links,
            images=parsed_data.images,
            text_content=parsed_data.text_content,
            forms=parsed_data.forms,
            metadata=parsed_data.metadata,
            seo_score=seo_result['seo_score'],
            title_score=seo_result['title_score'],
            description_score=seo_result['description_score'],
            keyword_score=seo_result['keyword_score'],
            structure_score=seo_result['structure_score'],
            content_score=seo_result['content_score'],
            technical_score=seo_result['technical_score'],
            recommendations=seo_result['recommendations'],
            warnings=seo_result['warnings'],
            errors=seo_result['errors'],
            word_count=seo_result['word_count'],
            character_count=seo_result['character_count'],
            link_count=len(parsed_data.links),
            image_count=len(parsed_data.images),
            form_count=len(parsed_data.forms),
            load_time=fetch_time,
            parsing_time=parse_time,
            analysis_time=analysis_time,
            total_time=total_time,
            cached=False,
            parser_used=parsed_data.parser_used
        )
        
        # Cachear resultado
        if self.enable_caching:
            cache_key = f"seo_analysis:{url}"
            await self.cache.set(cache_key, result, self.cache_ttl)
        
        # Actualizar estadísticas
        self._update_stats(True, result)
        
        logger.info(f"SEO analysis completed for {url} in {total_time:.3f}s")
        
        return result
    
    async def _analyze_seo(self, parsed_data: ParsedData, url: str) -> Dict[str, Any]:
        """Analizar SEO de los datos parseados."""
        # Ejecutar análisis en thread pool para operaciones CPU-intensivas
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self._perform_seo_analysis,
            parsed_data,
            url
        )
    
    def _perform_seo_analysis(self, parsed_data: ParsedData, url: str) -> Dict[str, Any]:
        """Realizar análisis SEO (ejecutado en thread pool)."""
        recommendations = []
        warnings = []
        errors = []
        
        # Análisis del título
        title_score, title_recs = self._analyze_title(parsed_data.title)
        
        # Análisis de la descripción
        description_score, desc_recs = self._analyze_description(parsed_data.meta_description)
        
        # Análisis de keywords
        keyword_score, keyword_recs = self._analyze_keywords(parsed_data.meta_keywords)
        
        # Análisis de estructura
        structure_score, structure_recs = self._analyze_structure(parsed_data)
        
        # Análisis de contenido
        content_score, content_recs = self._analyze_content(parsed_data)
        
        # Análisis técnico
        technical_score, technical_recs = self._analyze_technical(parsed_data, url)
        
        # Combinar recomendaciones
        recommendations.extend(title_recs)
        recommendations.extend(desc_recs)
        recommendations.extend(keyword_recs)
        recommendations.extend(structure_recs)
        recommendations.extend(content_recs)
        recommendations.extend(technical_recs)
        
        # Calcular puntuación SEO general
        seo_score = (
            title_score * 0.2 +
            description_score * 0.15 +
            keyword_score * 0.1 +
            structure_score * 0.25 +
            content_score * 0.2 +
            technical_score * 0.1
        )
        
        # Estadísticas de contenido
        word_count = len(parsed_data.text_content.split())
        character_count = len(parsed_data.text_content)
        
        return {
            'seo_score': seo_score,
            'title_score': title_score,
            'description_score': description_score,
            'keyword_score': keyword_score,
            'structure_score': structure_score,
            'content_score': content_score,
            'technical_score': technical_score,
            'recommendations': recommendations,
            'warnings': warnings,
            'errors': errors,
            'word_count': word_count,
            'character_count': character_count
        }
    
    def _analyze_title(self, title: str) -> Tuple[float, List[str]]:
        """Analizar título."""
        score = 0.0
        recommendations = []
        
        if not title:
            recommendations.append("Agregar un título único y descriptivo")
            return score, recommendations
        
        title_length = len(title)
        
        if title_length < self.min_title_length:
            score += 0.3
            recommendations.append(f"El título es muy corto ({title_length} caracteres). Ideal: {self.min_title_length}-{self.max_title_length}")
        elif title_length <= self.max_title_length:
            score += 1.0
        else:
            score += 0.5
            recommendations.append(f"El título es muy largo ({title_length} caracteres). Ideal: {self.min_title_length}-{self.max_title_length}")
        
        # Verificar palabras clave
        if any(keyword in title.lower() for keyword in ['seo', 'optimization', 'search']):
            score += 0.2
        
        return min(score, 1.0), recommendations
    
    def _analyze_description(self, description: str) -> Tuple[float, List[str]]:
        """Analizar descripción meta."""
        score = 0.0
        recommendations = []
        
        if not description:
            recommendations.append("Agregar una meta descripción única y atractiva")
            return score, recommendations
        
        desc_length = len(description)
        
        if desc_length < self.min_description_length:
            score += 0.3
            recommendations.append(f"La descripción es muy corta ({desc_length} caracteres). Ideal: {self.min_description_length}-{self.max_description_length}")
        elif desc_length <= self.max_description_length:
            score += 1.0
        else:
            score += 0.5
            recommendations.append(f"La descripción es muy larga ({desc_length} caracteres). Ideal: {self.min_description_length}-{self.max_description_length}")
        
        return min(score, 1.0), recommendations
    
    def _analyze_keywords(self, keywords: str) -> Tuple[float, List[str]]:
        """Analizar keywords."""
        score = 0.0
        recommendations = []
        
        if not keywords:
            recommendations.append("Considerar agregar meta keywords relevantes")
            return score, recommendations
        
        keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
        
        if len(keyword_list) >= 3:
            score += 0.8
        elif len(keyword_list) >= 1:
            score += 0.5
        else:
            score += 0.2
        
        if len(keyword_list) > 10:
            recommendations.append("Demasiadas keywords pueden ser consideradas spam")
            score *= 0.8
        
        return min(score, 1.0), recommendations
    
    def _analyze_structure(self, parsed_data: ParsedData) -> Tuple[float, List[str]]:
        """Analizar estructura HTML."""
        score = 0.0
        recommendations = []
        
        # Análisis de headers
        if parsed_data.h1_tags:
            if len(parsed_data.h1_tags) == 1:
                score += 0.3
            else:
                recommendations.append("Usar solo un H1 por página")
        else:
            recommendations.append("Agregar un H1 principal")
        
        if parsed_data.h2_tags:
            score += 0.2
        else:
            recommendations.append("Usar H2 para organizar el contenido")
        
        if parsed_data.h3_tags:
            score += 0.1
        
        # Análisis de links
        if len(parsed_data.links) >= 5:
            score += 0.2
        else:
            recommendations.append("Agregar más enlaces internos y externos relevantes")
        
        # Análisis de imágenes
        if parsed_data.images:
            images_with_alt = sum(1 for img in parsed_data.images if img.get('alt'))
            if images_with_alt == len(parsed_data.images):
                score += 0.2
            else:
                recommendations.append("Agregar atributos alt a todas las imágenes")
        
        return min(score, 1.0), recommendations
    
    def _analyze_content(self, parsed_data: ParsedData) -> Tuple[float, List[str]]:
        """Analizar contenido."""
        score = 0.0
        recommendations = []
        
        content_length = len(parsed_data.text_content)
        word_count = len(parsed_data.text_content.split())
        
        if content_length >= self.min_content_length:
            score += 0.5
        else:
            recommendations.append(f"El contenido es muy corto ({word_count} palabras). Ideal: al menos {self.min_content_length} caracteres")
        
        if word_count >= 300:
            score += 0.3
        elif word_count >= 150:
            score += 0.2
        else:
            score += 0.1
        
        # Verificar densidad de palabras clave
        if word_count > 0:
            # Análisis básico de densidad (simplificado)
            score += 0.2
        
        return min(score, 1.0), recommendations
    
    def _analyze_technical(self, parsed_data: ParsedData, url: str) -> Tuple[float, List[str]]:
        """Análisis técnico."""
        score = 0.0
        recommendations = []
        
        # Verificar metadatos
        if parsed_data.metadata.get('canonical'):
            score += 0.2
        else:
            recommendations.append("Agregar URL canónica")
        
        # Verificar Open Graph
        og_tags = [k for k in parsed_data.metadata.keys() if k.startswith('og:')]
        if len(og_tags) >= 3:
            score += 0.2
        else:
            recommendations.append("Agregar más meta tags de Open Graph")
        
        # Verificar formularios
        if parsed_data.forms:
            score += 0.1
        
        return min(score, 1.0), recommendations
    
    def _update_stats(self, success: bool, result: Optional[SEOAnalysisResult] = None):
        """Actualizar estadísticas del servicio."""
        with self.lock:
            self.stats.total_analyses += 1
            
            if success:
                self.stats.successful_analyses += 1
                if result:
                    self.stats.total_urls_processed += 1
                    self.stats.total_words_analyzed += result.word_count
                    self.stats.total_links_found += result.link_count
                    self.stats.total_images_found += result.image_count
                    
                    # Actualizar tiempo promedio
                    if self.stats.successful_analyses > 0:
                        self.stats.average_analysis_time = (
                            (self.stats.average_analysis_time * (self.stats.successful_analyses - 1) + result.total_time) 
                            / self.stats.successful_analyses
                        )
            else:
                self.stats.failed_analyses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del servicio."""
        with self.lock:
            return {
                'service_type': 'ultra_optimized_v2',
                'total_analyses': self.stats.total_analyses,
                'successful_analyses': self.stats.successful_analyses,
                'failed_analyses': self.stats.failed_analyses,
                'success_rate': self.stats.successful_analyses / max(self.stats.total_analyses, 1),
                'average_analysis_time': self.stats.average_analysis_time,
                'cache_hits': self.stats.cache_hits,
                'cache_misses': self.stats.cache_misses,
                'cache_hit_ratio': self.stats.cache_hits / max(self.stats.cache_hits + self.stats.cache_misses, 1),
                'total_urls_processed': self.stats.total_urls_processed,
                'total_words_analyzed': self.stats.total_words_analyzed,
                'total_links_found': self.stats.total_links_found,
                'total_images_found': self.stats.total_images_found,
                'max_concurrent_analyses': self.max_concurrent_analyses,
                'enable_caching': self.enable_caching,
                'enable_compression': self.enable_compression
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check del servicio."""
        try:
            # Verificar componentes
            parser_health = self.parser.health_check()
            cache_health = await self.cache.health_check()
            http_health = await self.http_client.health_check()
            
            # Determinar estado general
            if all(h['status'] == 'healthy' for h in [parser_health, cache_health, http_health]):
                overall_status = 'healthy'
            elif any(h['status'] == 'unhealthy' for h in [parser_health, cache_health, http_health]):
                overall_status = 'unhealthy'
            else:
                overall_status = 'degraded'
            
            return {
                'status': overall_status,
                'service_type': 'ultra_optimized_v2',
                'components': {
                    'parser': parser_health,
                    'cache': cache_health,
                    'http_client': http_health
                },
                'stats': self.get_stats()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'service_type': 'ultra_optimized_v2',
                'error': str(e)
            }
    
    async def clear_cache(self) -> Any:
        """Limpiar cache del servicio."""
        await self.cache.clear()
        logger.info("SEO service cache cleared")
    
    async def close(self) -> Any:
        """Cerrar servicio SEO."""
        try:
            # Cerrar componentes
            await self.cache.close()
            await self.http_client.close()
            
            # Cerrar thread pool
            self.thread_pool.shutdown(wait=True)
            
            logger.info("SEO service closed")
            
        except Exception as e:
            logger.error(f"Error closing SEO service: {e}")
    
    async def __aenter__(self) -> Any:
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Context manager exit."""
        await self.close() 