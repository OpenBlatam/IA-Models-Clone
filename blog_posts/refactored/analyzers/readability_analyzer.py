from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import logging
import re
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from .base import BaseAnalyzer, CachedAnalyzerMixin
from ..models import NLPAnalysisResult, ReadabilityMetrics
from ..config import NLPConfig
    import textstat
from typing import Any, List, Dict, Optional
import asyncio
"""
Analizador de legibilidad ultra-optimizado.
"""



# Importaciones condicionales
try:
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

logger = logging.getLogger(__name__)

class ReadabilityAnalyzer(BaseAnalyzer, CachedAnalyzerMixin):
    """Analizador de legibilidad con múltiples métricas."""
    
    def __init__(self, config: NLPConfig, executor: Optional[ThreadPoolExecutor] = None):
        
    """__init__ function."""
super().__init__(config, executor)
        self.target_grade = config.analysis.readability_target_grade
        self._initialize()
    
    def _initialize(self) -> Any:
        """Inicializar analizador."""
        if TEXTSTAT_AVAILABLE:
            self.logger.info("Textstat library available for advanced readability analysis")
        else:
            self.logger.warning("Textstat not available, using basic readability analysis")
    
    def get_name(self) -> str:
        """Obtener nombre del analizador."""
        return "readability"
    
    def is_available(self) -> bool:
        """Verificar si el analizador está disponible."""
        return True  # Siempre disponible con fallback básico
    
    async def _perform_analysis(self, text: str, result: NLPAnalysisResult, options: Dict[str, Any]) -> NLPAnalysisResult:
        """Realizar análisis de legibilidad."""
        # Validar texto
        validation_errors = self.validate_text(text)
        if validation_errors:
            for error in validation_errors:
                result.add_error(f"Readability validation: {error}")
            return result
        
        # Análisis de legibilidad
        if TEXTSTAT_AVAILABLE:
            result.readability = await self._analyze_with_textstat(text)
        else:
            result.readability = await self._basic_readability_analysis(text)
        
        # Generar recomendaciones
        self._generate_readability_recommendations(result)
        
        return result
    
    async def _analyze_with_textstat(self, text: str) -> ReadabilityMetrics:
        """Análisis avanzado con textstat."""
        def analyze():
            
    """analyze function."""
try:
                # Múltiples métricas de legibilidad
                flesch_ease = textstat.flesch_reading_ease(text)
                flesch_grade = textstat.flesch_kincaid_grade(text)
                gunning_fog = textstat.gunning_fog(text)
                coleman_liau = textstat.coleman_liau_index(text)
                automated_readability = textstat.automated_readability_index(text)
                
                # Calcular score promedio normalizado
                scores = [flesch_ease]  # Flesch ya está en 0-100
                
                # Normalizar otras métricas a 0-100
                if flesch_grade > 0:
                    grade_score = max(0, 100 - (flesch_grade - 6) * 10)  # Grado 6 = 100, incrementa penalty
                    scores.append(grade_score)
                
                if gunning_fog > 0:
                    fog_score = max(0, 100 - (gunning_fog - 8) * 8)  # Grado 8 = 100
                    scores.append(fog_score)
                
                avg_score = sum(scores) / len(scores) if scores else 50.0
                
                return ReadabilityMetrics(
                    flesch_reading_ease=flesch_ease,
                    flesch_kincaid_grade=flesch_grade,
                    gunning_fog=gunning_fog,
                    coleman_liau=coleman_liau,
                    automated_readability=automated_readability,
                    score=avg_score,
                    level=self._score_to_level(avg_score)
                )
            except Exception as e:
                self.logger.warning(f"Textstat analysis failed: {e}")
                return self._fallback_analysis(text)
        
        return await self._run_in_executor(analyze)
    
    async def _basic_readability_analysis(self, text: str) -> ReadabilityMetrics:
        """Análisis básico sin dependencias externas."""
        def analyze():
            
    """analyze function."""
return self._fallback_analysis(text)
        
        return await self._run_in_executor(analyze)
    
    def _fallback_analysis(self, text: str) -> ReadabilityMetrics:
        """Análisis de fallback cuando textstat no está disponible."""
        # Métricas básicas
        words = text.split()
        sentences = re.findall(r'[.!?]+', text)
        
        word_count = len(words)
        sentence_count = len(sentences) if sentences else 1
        
        # Longitud promedio de palabra
        avg_word_length = sum(len(word) for word in words) / word_count if words else 0
        
        # Longitud promedio de oración
        avg_sentence_length = word_count / sentence_count
        
        # Score básico basado en longitudes
        # Palabras más cortas y oraciones más cortas = mejor legibilidad
        word_score = max(0, 100 - (avg_word_length - 4) * 15)  # 4 letras promedio = 100
        sentence_score = max(0, 100 - (avg_sentence_length - 15) * 3)  # 15 palabras = 100
        
        basic_score = (word_score + sentence_score) / 2
        
        # Aproximar Flesch Reading Ease
        estimated_flesch = basic_score
        
        return ReadabilityMetrics(
            flesch_reading_ease=estimated_flesch,
            flesch_kincaid_grade=max(1, 18 - (estimated_flesch / 6)),  # Aproximación
            gunning_fog=avg_sentence_length * 0.4,  # Aproximación simple
            coleman_liau=0.0,
            automated_readability=0.0,
            score=basic_score,
            level=self._score_to_level(basic_score)
        )
    
    def _score_to_level(self, score: float) -> str:
        """Convertir score a nivel de legibilidad."""
        if score >= 85:
            return "very_easy"
        elif score >= 70:
            return "easy"
        elif score >= 55:
            return "average"
        elif score >= 35:
            return "difficult"
        else:
            return "very_difficult"
    
    def _generate_readability_recommendations(self, result: NLPAnalysisResult):
        """Generar recomendaciones de legibilidad."""
        readability = result.readability
        
        if readability.score < 60:
            result.add_recommendation("Improve readability by using shorter sentences and simpler words")
        
        if readability.flesch_kincaid_grade > self.target_grade + 2:
            result.add_recommendation(f"Content is at grade {readability.flesch_kincaid_grade:.1f} level, consider simplifying for grade {self.target_grade}")
        
        if readability.gunning_fog > 15:
            result.add_recommendation("Reduce sentence complexity to improve comprehension")
        
        if result.basic.avg_words_per_sentence > 20:
            result.add_recommendation("Break up long sentences for better flow")
        
        if readability.score >= 85:
            result.add_recommendation("Excellent readability! Content is very accessible")
    
    def _apply_cached_result(self, result: NLPAnalysisResult, cached_data: Dict[str, Any]):
        """Aplicar resultado cacheado."""
        if 'readability' in cached_data:
            read_data = cached_data['readability']
            result.readability = ReadabilityMetrics(
                flesch_reading_ease=read_data.get('flesch_reading_ease', 0.0),
                flesch_kincaid_grade=read_data.get('flesch_kincaid_grade', 0.0),
                gunning_fog=read_data.get('gunning_fog', 0.0),
                coleman_liau=read_data.get('coleman_liau', 0.0),
                automated_readability=read_data.get('automated_readability', 0.0),
                score=read_data.get('score', 50.0),
                level=read_data.get('level', 'average')
            ) 