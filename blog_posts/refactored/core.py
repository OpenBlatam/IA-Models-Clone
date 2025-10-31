from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field
from .config import NLPConfig, get_config
from .models import NLPAnalysisResult, AnalysisRequest, BasicMetrics, AnalysisStatus, QualityMetrics
from .cache_manager import CacheManager
from .model_manager import ModelManager
from .factory import AnalyzerFactory
from typing import Any, List, Dict, Optional
"""
Motor NLP ultra-optimizado refactorizado - Clase Principal.
"""



logger = logging.getLogger(__name__)

class RefactoredNLPEngine:
    """Motor NLP principal refactorizado con arquitectura modular."""
    
    def __init__(self, config: Optional[NLPConfig] = None):
        
    """__init__ function."""
self.config = config or get_config()
        self.cache_manager = CacheManager(self.config.cache)
        self.model_manager = ModelManager(self.config)
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.performance.max_workers,
            thread_name_prefix="NLPWorker"
        )
        self.analyzer_factory = AnalyzerFactory(self.config, self.executor)
        
        # Estadísticas globales
        self.stats = {
            'total_analyses': 0,
            'total_time_ms': 0.0,
            'avg_time_ms': 0.0,
            'errors': 0,
            'cache_hit_rate': 0.0
        }
        
        self._initialized = False
        self.logger = logging.getLogger(f"{__name__}.RefactoredNLPEngine")
        self._setup_logging()
    
    def _setup_logging(self) -> Any:
        """Configurar sistema de logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.performance.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def initialize(self) -> Any:
        """Inicializar todos los componentes del motor."""
        if self._initialized:
            return
        
        self.logger.info("Initializing RefactoredNLPEngine...")
        start_time = time.time()
        
        try:
            # Inicializar componentes en paralelo
            init_tasks = [
                self.model_manager.initialize(),
                self.analyzer_factory.initialize()
            ]
            
            await asyncio.gather(*init_tasks)
            
            self._initialized = True
            init_time = (time.time() - start_time) * 1000
            self.logger.info(f"RefactoredNLPEngine initialized in {init_time:.2f}ms")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP engine: {e}")
            raise
    
    async def analyze_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> NLPAnalysisResult:
        """
        Analizar un texto individual.
        
        Args:
            text: Texto a analizar
            options: Opciones de análisis opcionales
            
        Returns:
            Resultado completo del análisis NLP
        """
        if not self._initialized:
            await self.initialize()
        
        request = AnalysisRequest(text=text, options=options or {})
        return await self._process_request(request)
    
    async def analyze_batch(self, texts: List[str], options: Optional[Dict[str, Any]] = None) -> List[NLPAnalysisResult]:
        """
        Analizar lote de textos en paralelo.
        
        Args:
            texts: Lista de textos a analizar
            options: Opciones de análisis opcionales
            
        Returns:
            Lista de resultados de análisis
        """
        if not self._initialized:
            await self.initialize()
        
        if not texts:
            return []
        
        # Crear solicitudes
        requests = [AnalysisRequest(text=text, options=options or {}) for text in texts]
        
        # Procesar en paralelo con límite de concurrencia
        batch_size = self.config.performance.batch_size
        results = []
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            batch_tasks = [self._process_request(req) for req in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Manejar excepciones
            for result in batch_results:
                if isinstance(result, Exception):
                    error_result = NLPAnalysisResult()
                    error_result.add_error(f"Batch processing error: {str(result)}")
                    results.append(error_result)
                else:
                    results.append(result)
        
        return results
    
    async async def _process_request(self, request: AnalysisRequest) -> NLPAnalysisResult:
        """Procesar solicitud individual de análisis."""
        start_time = time.time()
        
        # Validar solicitud
        validation_errors = request.validate()
        if validation_errors:
            result = NLPAnalysisResult()
            for error in validation_errors:
                result.add_error(error)
            return result
        
        # Crear resultado inicial
        result = NLPAnalysisResult()
        result.status = AnalysisStatus.PROCESSING
        
        try:
            # Verificar cache
            cache_key = request.get_cache_key()
            cached_result = await self.cache_manager.get(cache_key)
            
            if cached_result:
                result = NLPAnalysisResult.from_dict(cached_result)
                result.performance.cache_hit = True
                result.status = AnalysisStatus.CACHED
                self.logger.debug(f"Cache hit for request: {cache_key[:16]}...")
            else:
                # Análisis completo
                result = await self._perform_full_analysis(request.text, result, request.options)
                
                # Guardar en cache
                await self.cache_manager.set(cache_key, result.to_dict())
                result.performance.cache_hit = False
            
            # Actualizar métricas de rendimiento
            processing_time = (time.time() - start_time) * 1000
            result.performance.processing_time_ms = processing_time
            result.mark_completed()
            
            # Actualizar estadísticas globales
            self._update_global_stats(result)
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            result.add_error(f"Analysis failed: {str(e)}")
            result.status = AnalysisStatus.ERROR
            self.stats['errors'] += 1
        
        return result
    
    async def _perform_full_analysis(self, text: str, result: NLPAnalysisResult, options: Dict[str, Any]) -> NLPAnalysisResult:
        """Realizar análisis completo usando analizadores modulares."""
        # Métricas básicas (siempre se calculan)
        result.basic = BasicMetrics.from_text(text)
        
        # Obtener analizadores habilitados
        analyzers = self.analyzer_factory.get_enabled_analyzers(options)
        
        # Ejecutar analizadores en paralelo
        analysis_tasks = []
        for analyzer in analyzers:
            task = asyncio.create_task(analyzer.analyze(text, result, options))
            analysis_tasks.append(task)
        
        # Esperar a que terminen todos los análisis
        analyzer_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Procesar resultados y manejar errores
        for i, analyzer_result in enumerate(analyzer_results):
            if isinstance(analyzer_result, Exception):
                analyzer_name = analyzers[i].get_name()
                result.add_error(f"{analyzer_name}: {str(analyzer_result)}")
            else:
                # El resultado ya se ha actualizado por referencia
                pass
        
        # Calcular score de calidad general
        result.quality = self._calculate_overall_quality(result)
        
        return result
    
    def _calculate_overall_quality(self, result: NLPAnalysisResult) -> QualityMetrics:
        """Calcular métricas de calidad general."""
        scores = []
        weights = []
        
        # Score de contenido basado en sentimientos
        content_score = max(0, min(100, result.sentiment.score))
        scores.append(content_score)
        weights.append(0.3)
        
        # Score de estructura basado en legibilidad
        structure_score = result.readability.score
        scores.append(structure_score)
        weights.append(0.4)
        
        # Score de engagement basado en palabras clave
        engagement_score = min(100, result.keywords.avg_score * 100) if result.keywords.keywords else 50
        scores.append(engagement_score)
        weights.append(0.2)
        
        # Score SEO básico
        seo_score = 50  # Placeholder - se puede implementar lógica más compleja
        if result.basic.word_count >= 300:  # Longitud mínima para SEO
            seo_score += 20
        if result.keywords.total_keywords >= 3:  # Suficientes keywords
            seo_score += 15
        if result.readability.score >= 60:  # Legibilidad adecuada
            seo_score += 15
        
        scores.append(min(100, seo_score))
        weights.append(0.1)
        
        # Calcular score general ponderado
        overall_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        return QualityMetrics(
            overall_score=overall_score,
            content_score=content_score,
            structure_score=structure_score,
            engagement_score=engagement_score,
            seo_score=min(100, seo_score)
        )
    
    def _update_global_stats(self, result: NLPAnalysisResult):
        """Actualizar estadísticas globales."""
        self.stats['total_analyses'] += 1
        self.stats['total_time_ms'] += result.performance.processing_time_ms
        self.stats['avg_time_ms'] = self.stats['total_time_ms'] / self.stats['total_analyses']
        
        # Actualizar tasa de cache hit
        cache_stats = self.cache_manager.get_stats()
        self.stats['cache_hit_rate'] = cache_stats['hit_rate']
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas completas del sistema."""
        return {
            'engine': self.stats,
            'cache': self.cache_manager.get_stats(),
            'models': self.model_manager.get_stats(),
            'analyzers': self.analyzer_factory.get_stats(),
            'config': {
                'model_tier': self.config.models.type.value,
                'cache_backend': self.config.cache.backend.value,
                'max_workers': self.config.performance.max_workers,
                'batch_size': self.config.performance.batch_size
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar estado de salud del sistema."""
        health = {
            'status': 'healthy',
            'initialized': self._initialized,
            'components': {}
        }
        
        try:
            # Verificar componentes
            health['components']['cache'] = 'ok' if self.cache_manager else 'error'
            health['components']['models'] = 'ok' if self.model_manager else 'error'
            health['components']['analyzers'] = 'ok' if self.analyzer_factory else 'error'
            
            # Test rápido de análisis
            test_result = await self.analyze_text("Test text for health check")
            health['components']['analysis'] = 'ok' if not test_result.errors else 'warning'
            
            # Verificar si hay errores
            if any(status == 'error' for status in health['components'].values()):
                health['status'] = 'unhealthy'
            elif any(status == 'warning' for status in health['components'].values()):
                health['status'] = 'degraded'
            
        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
        
        return health
    
    async def cleanup(self) -> Any:
        """Limpiar recursos y cerrar conexiones."""
        self.logger.info("Cleaning up RefactoredNLPEngine...")
        
        try:
            # Cerrar componentes
            await self.model_manager.cleanup()
            
            # Cerrar executor
            if self.executor:
                self.executor.shutdown(wait=True)
            
            self.logger.info("RefactoredNLPEngine cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

# Instancia singleton global
_global_engine: Optional[RefactoredNLPEngine] = None

async def get_nlp_engine(config: Optional[NLPConfig] = None) -> RefactoredNLPEngine:
    """
    Obtener instancia global del motor NLP.
    
    Args:
        config: Configuración opcional (solo se usa en la primera llamada)
        
    Returns:
        Instancia del motor NLP refactorizado
    """
    global _global_engine
    
    if _global_engine is None:
        _global_engine = RefactoredNLPEngine(config)
        await _global_engine.initialize()
    
    return _global_engine

# Funciones de conveniencia para mantener compatibilidad con API original
async def analyze_text_refactored(text: str, **options) -> Dict[str, Any]:
    """Función de conveniencia para analizar texto."""
    engine = await get_nlp_engine()
    result = await engine.analyze_text(text, options)
    return result.to_dict()

async def analyze_batch_refactored(texts: List[str], **options) -> List[Dict[str, Any]]:
    """Función de conveniencia para analizar lote de textos."""
    engine = await get_nlp_engine()
    results = await engine.analyze_batch(texts, options)
    return [result.to_dict() for result in results] 