from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import string
from .base import BaseAnalyzer, CachedAnalyzerMixin
from ..models import NLPAnalysisResult, KeywordMetrics
from ..config import NLPConfig
    import yake
    from sklearn.feature_extraction.text import TfidfVectorizer
    import spacy
from typing import Any, List, Dict, Optional
import asyncio
"""
Analizador de palabras clave ultra-optimizado.
"""



# Importaciones condicionales
try:
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False

try:
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

class KeywordAnalyzer(BaseAnalyzer, CachedAnalyzerMixin):
    """Analizador de palabras clave con múltiples técnicas."""
    
    def __init__(self, config: NLPConfig, executor: Optional[ThreadPoolExecutor] = None):
        
    """__init__ function."""
super().__init__(config, executor)
        self.max_keywords = config.analysis.keyword_max_count
        self.max_ngram = config.analysis.keyword_max_ngram
        self.yake_extractor = None
        self.spacy_model = None
        self._initialize_models()
    
    def _initialize_models(self) -> Any:
        """Inicializar modelos de extracción de palabras clave."""
        # Inicializar YAKE si está disponible
        if YAKE_AVAILABLE and self.config.analysis.enable_keywords:
            try:
                self.yake_extractor = yake.KeywordExtractor(
                    lan="en",
                    n=self.max_ngram,
                    dedupLim=0.9,
                    top=self.max_keywords * 2  # Extraer más para filtrar después
                )
                self.logger.info("YAKE keyword extractor initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize YAKE: {e}")
        
        # Inicializar spaCy si está disponible
        if SPACY_AVAILABLE and self.config.models.type.value in ['standard', 'advanced']:
            try:
                self.spacy_model = spacy.load(self.config.models.spacy_model)
                self.logger.info("spaCy model for keywords initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize spaCy model: {e}")
    
    def get_name(self) -> str:
        """Obtener nombre del analizador."""
        return "keywords"
    
    def is_available(self) -> bool:
        """Verificar si el analizador está disponible."""
        return True  # Siempre disponible con fallback básico
    
    async def _perform_analysis(self, text: str, result: NLPAnalysisResult, options: Dict[str, Any]) -> NLPAnalysisResult:
        """Realizar análisis de palabras clave."""
        # Validar texto
        validation_errors = self.validate_text(text)
        if validation_errors:
            for error in validation_errors:
                result.add_error(f"Keywords validation: {error}")
            return result
        
        # Múltiples técnicas de extracción
        keyword_results = []
        
        # YAKE (unsupervised keyword extraction)
        if self.yake_extractor:
            yake_keywords = await self._extract_with_yake(text)
            if yake_keywords:
                keyword_results.extend(yake_keywords)
        
        # spaCy NER y POS tagging
        if self.spacy_model:
            spacy_keywords = await self._extract_with_spacy(text)
            if spacy_keywords:
                keyword_results.extend(spacy_keywords)
        
        # TF-IDF básico
        if SKLEARN_AVAILABLE:
            tfidf_keywords = await self._extract_with_tfidf(text)
            if tfidf_keywords:
                keyword_results.extend(tfidf_keywords)
        
        # Extracción básica de fallback
        if not keyword_results:
            keyword_results = await self._basic_keyword_extraction(text)
            result.add_warning("Using basic keyword extraction")
        
        # Combinar y normalizar resultados
        final_keywords = self._combine_keyword_results(keyword_results)
        
        # Calcular densidad
        keyword_density = self._calculate_keyword_density(text, final_keywords)
        
        # Crear métricas
        result.keywords = KeywordMetrics(
            keywords=final_keywords,
            density=keyword_density,
            total_keywords=len(final_keywords),
            avg_score=sum(score for _, score in final_keywords) / len(final_keywords) if final_keywords else 0.0
        )
        
        return result
    
    async def _extract_with_yake(self, text: str) -> List[Tuple[str, float]]:
        """Extracción con YAKE."""
        try:
            def extract():
                
    """extract function."""
keywords = self.yake_extractor.extract_keywords(text)
                # YAKE devuelve (score, keyword) - score más bajo = mejor
                # Convertir a (keyword, confidence) donde confidence más alto = mejor
                normalized_keywords = []
                for score, keyword in keywords:
                    # Normalizar score a 0-1 (invertir porque score bajo = bueno en YAKE)
                    confidence = 1.0 / (1.0 + score)
                    normalized_keywords.append((keyword, confidence))
                return normalized_keywords
            
            return await self._run_in_executor(extract)
        except Exception as e:
            self.logger.warning(f"YAKE extraction failed: {e}")
            return []
    
    async def _extract_with_spacy(self, text: str) -> List[Tuple[str, float]]:
        """Extracción con spaCy usando NER y POS."""
        try:
            def extract():
                
    """extract function."""
doc = self.spacy_model(text)
                keywords = []
                
                # Entidades nombradas
                for ent in doc.ents:
                    if len(ent.text) > 2 and ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']:
                        keywords.append((ent.text.lower(), 0.8))
                
                # Sustantivos y adjetivos importantes
                important_tokens = []
                for token in doc:
                    if (token.pos_ in ['NOUN', 'ADJ'] and 
                        not token.is_stop and 
                        not token.is_punct and 
                        len(token.text) > 2):
                        important_tokens.append(token.lemma_.lower())
                
                # Contar frecuencias
                token_counts = Counter(important_tokens)
                total_tokens = len(important_tokens)
                
                # Convertir a keywords con scores basados en frecuencia
                for token, count in token_counts.most_common(self.max_keywords):
                    score = count / total_tokens if total_tokens > 0 else 0
                    keywords.append((token, score))
                
                return keywords
            
            return await self._run_in_executor(extract)
        except Exception as e:
            self.logger.warning(f"spaCy extraction failed: {e}")
            return []
    
    async def _extract_with_tfidf(self, text: str) -> List[Tuple[str, float]]:
        """Extracción con TF-IDF."""
        try:
            def extract():
                
    """extract function."""
# Preparar texto para TF-IDF
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                if len(sentences) < 2:
                    return []
                
                # Vectorizador TF-IDF
                vectorizer = TfidfVectorizer(
                    max_features=self.max_keywords * 3,
                    stop_words='english',
                    ngram_range=(1, self.max_ngram),
                    min_df=1,
                    max_df=0.95
                )
                
                tfidf_matrix = vectorizer.fit_transform(sentences)
                feature_names = vectorizer.get_feature_names_out()
                
                # Calcular scores promedio para cada término
                mean_scores = tfidf_matrix.mean(axis=0).A1
                
                # Crear lista de (keyword, score)
                keyword_scores = list(zip(feature_names, mean_scores))
                keyword_scores.sort(key=lambda x: x[1], reverse=True)
                
                return keyword_scores[:self.max_keywords]
            
            return await self._run_in_executor(extract)
        except Exception as e:
            self.logger.warning(f"TF-IDF extraction failed: {e}")
            return []
    
    async def _basic_keyword_extraction(self, text: str) -> List[Tuple[str, float]]:
        """Extracción básica de palabras clave."""
        def extract():
            
    """extract function."""
# Limpiar texto
            text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
            words = text_clean.split()
            
            # Filtrar palabras comunes y muy cortas
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'that', 'this', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 
                'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
                'its', 'our', 'their', 'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es',
                'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para'
            }
            
            filtered_words = [
                word for word in words 
                if len(word) > 2 and word not in stop_words
            ]
            
            # Contar frecuencias
            word_counts = Counter(filtered_words)
            total_words = len(filtered_words)
            
            # Calcular scores
            keywords = []
            for word, count in word_counts.most_common(self.max_keywords):
                score = count / total_words if total_words > 0 else 0
                keywords.append((word, score))
            
            return keywords
        
        return await self._run_in_executor(extract)
    
    def _combine_keyword_results(self, keyword_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Combinar resultados de múltiples extractores."""
        if not keyword_results:
            return []
        
        # Agrupar por keyword y promediar scores
        keyword_scores = {}
        keyword_counts = {}
        
        for keyword, score in keyword_results:
            keyword_clean = keyword.lower().strip()
            if keyword_clean not in keyword_scores:
                keyword_scores[keyword_clean] = 0.0
                keyword_counts[keyword_clean] = 0
            
            keyword_scores[keyword_clean] += score
            keyword_counts[keyword_clean] += 1
        
        # Calcular promedios
        averaged_keywords = []
        for keyword, total_score in keyword_scores.items():
            avg_score = total_score / keyword_counts[keyword]
            averaged_keywords.append((keyword, avg_score))
        
        # Ordenar por score y tomar los mejores
        averaged_keywords.sort(key=lambda x: x[1], reverse=True)
        return averaged_keywords[:self.max_keywords]
    
    def _calculate_keyword_density(self, text: str, keywords: List[Tuple[str, float]]) -> Dict[str, float]:
        """Calcular densidad de palabras clave en el texto."""
        text_lower = text.lower()
        word_count = len(text.split())
        
        density = {}
        for keyword, _ in keywords:
            # Contar ocurrencias de la keyword
            count = text_lower.count(keyword)
            density[keyword] = (count / word_count * 100) if word_count > 0 else 0.0
        
        return density
    
    def _apply_cached_result(self, result: NLPAnalysisResult, cached_data: Dict[str, Any]):
        """Aplicar resultado cacheado."""
        if 'keywords' in cached_data:
            kw_data = cached_data['keywords']
            result.keywords = KeywordMetrics(
                keywords=kw_data.get('keywords', []),
                density=kw_data.get('density', {}),
                total_keywords=kw_data.get('total_keywords', 0),
                avg_score=kw_data.get('avg_score', 0.0)
            ) 