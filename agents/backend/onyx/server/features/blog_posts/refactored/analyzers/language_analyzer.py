from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from .base import BaseAnalyzer, CachedAnalyzerMixin
from ..models import NLPAnalysisResult, LanguageMetrics
from ..config import NLPConfig
    from langdetect import detect_langs, detect
    from textblob import TextBlob
    import spacy
    from spacy.lang.en import English
    from spacy.lang.es import Spanish
from typing import Any, List, Dict, Optional
import asyncio
"""
Analizador de detección de idioma ultra-optimizado.
"""



# Importaciones condicionales
try:
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

class LanguageAnalyzer(BaseAnalyzer, CachedAnalyzerMixin):
    """Analizador de detección de idioma con múltiples técnicas."""
    
    def __init__(self, config: NLPConfig, executor: Optional[ThreadPoolExecutor] = None):
        
    """__init__ function."""
super().__init__(config, executor)
        self.language_patterns = self._load_language_patterns()
        self._initialize()
    
    def _initialize(self) -> Any:
        """Inicializar analizador."""
        available_methods = []
        if LANGDETECT_AVAILABLE:
            available_methods.append("langdetect")
        if TEXTBLOB_AVAILABLE:
            available_methods.append("textblob")
        if SPACY_AVAILABLE:
            available_methods.append("spacy")
        
        self.logger.info(f"Language detection methods available: {available_methods}")
    
    def get_name(self) -> str:
        """Obtener nombre del analizador."""
        return "language"
    
    def is_available(self) -> bool:
        """Verificar si el analizador está disponible."""
        return True  # Siempre disponible con fallback básico
    
    async def _perform_analysis(self, text: str, result: NLPAnalysisResult, options: Dict[str, Any]) -> NLPAnalysisResult:
        """Realizar análisis de detección de idioma."""
        # Validar texto
        validation_errors = self.validate_text(text)
        if validation_errors:
            for error in validation_errors:
                result.add_error(f"Language validation: {error}")
            return result
        
        # Múltiples técnicas de detección
        detection_results = []
        
        # langdetect (muy rápido y preciso)
        if LANGDETECT_AVAILABLE:
            langdetect_result = await self._detect_with_langdetect(text)
            if langdetect_result:
                detection_results.append(langdetect_result)
        
        # TextBlob (backup)
        if TEXTBLOB_AVAILABLE:
            textblob_result = await self._detect_with_textblob(text)
            if textblob_result:
                detection_results.append(textblob_result)
        
        # Detección básica por patrones
        pattern_result = await self._detect_with_patterns(text)
        if pattern_result:
            detection_results.append(pattern_result)
        
        # Combinar resultados
        if detection_results:
            result.language = self._combine_language_results(detection_results)
        else:
            # Fallback absoluto
            result.language = LanguageMetrics(
                detected_language="unknown",
                confidence=0.0,
                all_languages=[("unknown", 0.0)]
            )
            result.add_warning("Could not detect language")
        
        return result
    
    async def _detect_with_langdetect(self, text: str) -> Optional[LanguageMetrics]:
        """Detección con langdetect."""
        try:
            def detect_lang():
                
    """detect_lang function."""
# Detectar todos los idiomas posibles
                langs = detect_langs(text)
                
                # Idioma principal
                main_lang = langs[0] if langs else None
                if not main_lang:
                    return None
                
                # Convertir a formato estándar
                all_languages = [(lang.lang, lang.prob) for lang in langs[:5]]
                
                return LanguageMetrics(
                    detected_language=main_lang.lang,
                    confidence=main_lang.prob,
                    all_languages=all_languages
                )
            
            return await self._run_in_executor(detect_lang)
        except Exception as e:
            self.logger.warning(f"Langdetect failed: {e}")
            return None
    
    async def _detect_with_textblob(self, text: str) -> Optional[LanguageMetrics]:
        """Detección con TextBlob."""
        try:
            def detect_lang():
                
    """detect_lang function."""
blob = TextBlob(text)
                detected_lang = blob.detect_language()
                
                # TextBlob no proporciona confidence, estimamos basado en longitud
                confidence = min(0.8, len(text) / 1000) if len(text) > 50 else 0.5
                
                return LanguageMetrics(
                    detected_language=detected_lang,
                    confidence=confidence,
                    all_languages=[(detected_lang, confidence)]
                )
            
            return await self._run_in_executor(detect_lang)
        except Exception as e:
            self.logger.warning(f"TextBlob detection failed: {e}")
            return None
    
    async def _detect_with_patterns(self, text: str) -> Optional[LanguageMetrics]:
        """Detección básica usando patrones de idioma."""
        def detect_lang():
            
    """detect_lang function."""
text_lower = text.lower()
            language_scores = {}
            
            # Analizar patrones para cada idioma
            for lang, patterns in self.language_patterns.items():
                score = 0
                total_patterns = len(patterns['words']) + len(patterns['chars'])
                
                # Buscar palabras características
                for word in patterns['words']:
                    if word in text_lower:
                        score += 1
                
                # Buscar caracteres característicos
                for char in patterns['chars']:
                    if char in text_lower:
                        score += text_lower.count(char) * 0.1
                
                # Normalizar score
                if total_patterns > 0:
                    language_scores[lang] = score / total_patterns
            
            if not language_scores:
                return None
            
            # Encontrar idioma con mayor score
            best_lang = max(language_scores, key=language_scores.get)
            best_score = language_scores[best_lang]
            
            # Crear lista ordenada de idiomas
            sorted_langs = sorted(language_scores.items(), key=lambda x: x[1], reverse=True)
            all_languages = [(lang, score) for lang, score in sorted_langs[:3]]
            
            return LanguageMetrics(
                detected_language=best_lang,
                confidence=min(best_score, 0.7),  # Máximo 70% confidence para método básico
                all_languages=all_languages
            )
        
        return await self._run_in_executor(detect_lang)
    
    def _combine_language_results(self, results: List[LanguageMetrics]) -> LanguageMetrics:
        """Combinar resultados de múltiples detectores."""
        if len(results) == 1:
            return results[0]
        
        # Votar por idioma más común y promediar confidences
        language_votes = {}
        language_confidences = {}
        
        for result in results:
            lang = result.detected_language
            if lang not in language_votes:
                language_votes[lang] = 0
                language_confidences[lang] = []
            
            language_votes[lang] += 1
            language_confidences[lang].append(result.confidence)
        
        # Encontrar idioma con más votos
        best_lang = max(language_votes, key=language_votes.get)
        
        # Calcular confidence promedio para el idioma ganador
        avg_confidence = sum(language_confidences[best_lang]) / len(language_confidences[best_lang])
        
        # Combinar todas las detecciones para all_languages
        all_detections = {}
        for result in results:
            for lang, conf in result.all_languages:
                if lang not in all_detections:
                    all_detections[lang] = []
                all_detections[lang].append(conf)
        
        # Promediar confidences para cada idioma
        combined_languages = []
        for lang, confs in all_detections.items():
            avg_conf = sum(confs) / len(confs)
            combined_languages.append((lang, avg_conf))
        
        # Ordenar por confidence
        combined_languages.sort(key=lambda x: x[1], reverse=True)
        
        return LanguageMetrics(
            detected_language=best_lang,
            confidence=avg_confidence,
            all_languages=combined_languages[:5]
        )
    
    def _load_language_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Cargar patrones característicos de idiomas."""
        return {
            'en': {
                'words': ['the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but'],
                'chars': ['th', 'he', 'in', 'er', 'an']
            },
            'es': {
                'words': ['que', 'de', 'no', 'la', 'el', 'en', 'y', 'un', 'es', 'se', 'te', 'lo'],
                'chars': ['ñ', 'ó', 'á', 'í', 'é', 'ú']
            },
            'fr': {
                'words': ['le', 'de', 'et', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que'],
                'chars': ['ç', 'à', 'é', 'è', 'ê', 'ë', 'î', 'ï', 'ô', 'ù', 'û', 'ü', 'ÿ']
            },
            'pt': {
                'words': ['que', 'de', 'não', 'um', 'o', 'ser', 'e', 'em', 'ter', 'com'],
                'chars': ['ã', 'õ', 'ç', 'á', 'é', 'í', 'ó', 'ú', 'â', 'ê', 'î', 'ô', 'û']
            },
            'it': {
                'words': ['che', 'di', 'non', 'un', 'il', 'essere', 'e', 'in', 'avere', 'con'],
                'chars': ['à', 'è', 'é', 'ì', 'í', 'î', 'ò', 'ó', 'ù', 'ú']
            },
            'de': {
                'words': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'],
                'chars': ['ä', 'ö', 'ü', 'ß']
            }
        }
    
    def _apply_cached_result(self, result: NLPAnalysisResult, cached_data: Dict[str, Any]):
        """Aplicar resultado cacheado."""
        if 'language' in cached_data:
            lang_data = cached_data['language']
            result.language = LanguageMetrics(
                detected_language=lang_data.get('detected_language', 'unknown'),
                confidence=lang_data.get('confidence', 0.0),
                all_languages=lang_data.get('all_languages', [])
            ) 