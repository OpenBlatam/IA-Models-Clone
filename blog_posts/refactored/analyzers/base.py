from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor
import asyncio
from ..models import NLPAnalysisResult, AnalysisStatus
from ..config import NLPConfig
        import hashlib
from typing import Any, List, Dict, Optional
"""
Clase base para analizadores NLP.
"""



logger = logging.getLogger(__name__)

class AnalyzerInterface(ABC):
    """Interfaz para todos los analizadores NLP."""
    
    @abstractmethod
    async def analyze(self, text: str, result: NLPAnalysisResult, options: Dict[str, Any]) -> NLPAnalysisResult:
        """
        Analizar texto y actualizar resultado.
        
        Args:
            text: Texto a analizar
            result: Resultado a actualizar
            options: Opciones de análisis
            
        Returns:
            Resultado actualizado
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Verificar si el analizador está disponible."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Obtener nombre del analizador."""
        pass

class BaseAnalyzer(AnalyzerInterface):
    """Clase base para analizadores con funcionalidad común."""
    
    def __init__(self, config: NLPConfig, executor: Optional[ThreadPoolExecutor] = None):
        
    """__init__ function."""
self.config = config
        self.executor = executor
        self.stats = {
            'total_analyses': 0,
            'total_time_ms': 0.0,
            'errors': 0,
            'avg_time_ms': 0.0
        }
        self._setup_logging()
    
    def _setup_logging(self) -> Any:
        """Configurar logging específico del analizador."""
        self.logger = logging.getLogger(f"{__name__}.{self.get_name()}")
        self.logger.setLevel(getattr(logging, self.config.performance.log_level))
    
    async def analyze(self, text: str, result: NLPAnalysisResult, options: Dict[str, Any]) -> NLPAnalysisResult:
        """
        Analizar texto con manejo de errores y métricas.
        
        Args:
            text: Texto a analizar
            result: Resultado a actualizar
            options: Opciones de análisis
            
        Returns:
            Resultado actualizado
        """
        if not self.is_available():
            self.logger.warning(f"{self.get_name()} not available, skipping")
            result.add_warning(f"{self.get_name()} analyzer not available")
            return result
        
        start_time = time.time()
        
        try:
            # Realizar análisis específico
            result = await self._perform_analysis(text, result, options)
            
            # Actualizar estadísticas
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_stats(processing_time_ms)
            
            # Agregar modelo usado a las métricas
            if self.get_name() not in result.performance.models_used:
                result.performance.models_used.append(self.get_name())
            
            self.logger.debug(f"{self.get_name()} analysis completed in {processing_time_ms:.2f}ms")
            
        except Exception as e:
            self.logger.error(f"Error in {self.get_name()}: {e}")
            result.add_error(f"{self.get_name()}: {str(e)}")
            self.stats['errors'] += 1
        
        return result
    
    @abstractmethod
    async def _perform_analysis(self, text: str, result: NLPAnalysisResult, options: Dict[str, Any]) -> NLPAnalysisResult:
        """
        Realizar análisis específico del analizador.
        
        Args:
            text: Texto a analizar
            result: Resultado a actualizar
            options: Opciones de análisis
            
        Returns:
            Resultado actualizado
        """
        pass
    
    async def _run_in_executor(self, func, *args) -> Any:
        """Ejecutar función CPU-intensiva en ThreadPoolExecutor."""
        if self.executor:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, func, *args)
        else:
            # Ejecutar de forma síncrona si no hay executor
            return func(*args)
    
    def _update_stats(self, processing_time_ms: float):
        """Actualizar estadísticas del analizador."""
        self.stats['total_analyses'] += 1
        self.stats['total_time_ms'] += processing_time_ms
        self.stats['avg_time_ms'] = self.stats['total_time_ms'] / self.stats['total_analyses']
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del analizador."""
        return {
            'name': self.get_name(),
            'available': self.is_available(),
            **self.stats
        }
    
    def validate_text(self, text: str) -> List[str]:
        """
        Validar texto antes del análisis.
        
        Args:
            text: Texto a validar
            
        Returns:
            Lista de errores de validación
        """
        errors = []
        
        if not text or not text.strip():
            errors.append("Text is empty")
        
        if len(text) > 100000:  # 100KB máximo
            errors.append("Text too long (max 100KB)")
        
        return errors
    
    def _normalize_score(self, score: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
        """
        Normalizar score a rango 0-100.
        
        Args:
            score: Score a normalizar
            min_val: Valor mínimo del rango original
            max_val: Valor máximo del rango original
            
        Returns:
            Score normalizado (0-100)
        """
        if max_val == min_val:
            return 50.0  # Valor neutral
        
        normalized = ((score - min_val) / (max_val - min_val)) * 100
        return max(0.0, min(100.0, normalized))
    
    def _get_quality_level(self, score: float) -> str:
        """
        Determinar nivel de calidad basado en score.
        
        Args:
            score: Score 0-100
            
        Returns:
            Nivel de calidad
        """
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "average"
        elif score >= 40:
            return "poor"
        else:
            return "very_poor"

class AsyncAnalyzerMixin:
    """Mixin para análisis asíncrono avanzado."""
    
    async def analyze_batch(self, texts: List[str], options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analizar lote de textos en paralelo.
        
        Args:
            texts: Lista de textos
            options: Opciones de análisis
            
        Returns:
            Lista de resultados
        """
        tasks = []
        for text in texts:
            result = NLPAnalysisResult()
            task = asyncio.create_task(self.analyze(text, result, options))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrar excepciones
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch analysis error: {result}")
                processed_results.append({"error": str(result)})
            else:
                processed_results.append(result.to_dict())
        
        return processed_results

class CachedAnalyzerMixin:
    """Mixin para análisis con cache."""
    
    def __init__(self, *args, **kwargs) -> Any:
        super().__init__(*args, **kwargs)
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    async def analyze_with_cache(self, text: str, result: NLPAnalysisResult, options: Dict[str, Any]) -> NLPAnalysisResult:
        """Analizar con cache interno."""
        cache_key = self._get_cache_key(text, options)
        
        # Intentar cache
        if cache_key in self._cache:
            self._cache_hits += 1
            cached_data = self._cache[cache_key]
            self._apply_cached_result(result, cached_data)
            return result
        
        # Cache miss - analizar
        self._cache_misses += 1
        result = await self.analyze(text, result, options)
        
        # Guardar en cache
        self._cache[cache_key] = self._extract_cacheable_data(result)
        
        return result
    
    def _get_cache_key(self, text: str, options: Dict[str, Any]) -> str:
        """Generar key de cache."""
        content = f"{text}:{str(sorted(options.items()))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _extract_cacheable_data(self, result: NLPAnalysisResult) -> Dict[str, Any]:
        """Extraer datos cacheables del resultado."""
        return result.to_dict()
    
    def _apply_cached_result(self, result: NLPAnalysisResult, cached_data: Dict[str, Any]):
        """Aplicar datos cacheados al resultado."""
        # Implementación específica en cada analizador
        pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de cache."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._cache)
        } 