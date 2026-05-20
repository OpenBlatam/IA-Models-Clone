from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import logging
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from .base import BaseAnalyzer, CachedAnalyzerMixin
from ..models import NLPAnalysisResult, SentimentMetrics
from ..config import NLPConfig
    from textblob import TextBlob
    from transformers import pipeline
    import torch
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Any, List, Dict, Optional
import asyncio
"""
Analizador de sentimientos ultra-optimizado.
"""



# Importaciones condicionales
try:
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

class SentimentAnalyzer(BaseAnalyzer, CachedAnalyzerMixin):
    """Analizador de sentimientos con múltiples técnicas."""
    
    def __init__(self, config: NLPConfig, executor: Optional[ThreadPoolExecutor] = None):
        
    """__init__ function."""
super().__init__(config, executor)
        self.transformer_model = None
        self.vader_analyzer = None
        self._initialize_models()
    
    def _initialize_models(self) -> Any:
        """Inicializar modelos de sentimientos."""
        # Inicializar VADER si está disponible
        if NLTK_AVAILABLE and self.config.analysis.enable_sentiment:
            try:
                nltk.download('vader_lexicon', quiet=True)
                self.vader_analyzer = SentimentIntensityAnalyzer()
                self.logger.info("VADER sentiment analyzer initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize VADER: {e}")
        
        # Inicializar transformer si está disponible y configurado
        if (TRANSFORMERS_AVAILABLE and 
            self.config.models.type.value in ['standard', 'advanced'] and
            self.config.analysis.enable_sentiment):
            try:
                self.transformer_model = pipeline(
                    "sentiment-analysis",
                    model=self.config.models.sentiment_model,
                    return_all_scores=True
                )
                self.logger.info("Transformer sentiment model initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize transformer model: {e}")
    
    def get_name(self) -> str:
        """Obtener nombre del analizador."""
        return "sentiment"
    
    def is_available(self) -> bool:
        """Verificar si el analizador está disponible."""
        return (TEXTBLOB_AVAILABLE or 
                (NLTK_AVAILABLE and self.vader_analyzer is not None) or
                (TRANSFORMERS_AVAILABLE and self.transformer_model is not None))
    
    async def _perform_analysis(self, text: str, result: NLPAnalysisResult, options: Dict[str, Any]) -> NLPAnalysisResult:
        """Realizar análisis de sentimientos."""
        # Validar texto
        validation_errors = self.validate_text(text)
        if validation_errors:
            for error in validation_errors:
                result.add_error(f"Sentiment validation: {error}")
            return result
        
        # Múltiples técnicas de análisis
        sentiment_results = []
        
        # TextBlob (rápido y ligero)
        if TEXTBLOB_AVAILABLE:
            textblob_result = await self._analyze_with_textblob(text)
            if textblob_result:
                sentiment_results.append(textblob_result)
        
        # VADER (especializado en sentimientos)
        if self.vader_analyzer:
            vader_result = await self._analyze_with_vader(text)
            if vader_result:
                sentiment_results.append(vader_result)
        
        # Transformer (más preciso pero más lento)
        if self.transformer_model and len(text) < 512:  # Límite de tokens
            transformer_result = await self._analyze_with_transformer(text)
            if transformer_result:
                sentiment_results.append(transformer_result)
        
        # Combinar resultados
        if sentiment_results:
            result.sentiment = self._combine_sentiment_results(sentiment_results)
        else:
            # Fallback a análisis básico
            result.sentiment = self._basic_sentiment_analysis(text)
            result.add_warning("Using basic sentiment analysis")
        
        return result
    
    async def _analyze_with_textblob(self, text: str) -> Optional[SentimentMetrics]:
        """Análisis con TextBlob."""
        try:
            def analyze():
                
    """analyze function."""
blob = TextBlob(text)
                return SentimentMetrics(
                    polarity=blob.sentiment.polarity,
                    subjectivity=blob.sentiment.subjectivity,
                    confidence=0.7,  # TextBlob no proporciona confidence
                    label=self._polarity_to_label(blob.sentiment.polarity),
                    score=(blob.sentiment.polarity + 1) * 50
                )
            
            return await self._run_in_executor(analyze)
        except Exception as e:
            self.logger.warning(f"TextBlob analysis failed: {e}")
            return None
    
    async def _analyze_with_vader(self, text: str) -> Optional[SentimentMetrics]:
        """Análisis con VADER."""
        try:
            def analyze():
                
    """analyze function."""
scores = self.vader_analyzer.polarity_scores(text)
                compound = scores['compound']
                
                return SentimentMetrics(
                    polarity=compound,
                    subjectivity=0.5,  # VADER no calcula subjetividad
                    confidence=abs(compound),
                    label=self._polarity_to_label(compound),
                    score=(compound + 1) * 50
                )
            
            return await self._run_in_executor(analyze)
        except Exception as e:
            self.logger.warning(f"VADER analysis failed: {e}")
            return None
    
    async def _analyze_with_transformer(self, text: str) -> Optional[SentimentMetrics]:
        """Análisis con modelo Transformer."""
        try:
            def analyze():
                
    """analyze function."""
# Límite de texto para transformers
                text_truncated = text[:500]
                results = self.transformer_model(text_truncated)
                
                if results and len(results[0]) > 0:
                    # Buscar el resultado más confiable
                    best_result = max(results[0], key=lambda x: x['score'])
                    
                    # Convertir label a polaridad
                    label = best_result['label'].lower()
                    if 'positive' in label:
                        polarity = best_result['score'] 
                    elif 'negative' in label:
                        polarity = -best_result['score']
                    else:
                        polarity = 0.0
                    
                    return SentimentMetrics(
                        polarity=polarity,
                        subjectivity=0.8,  # Asumimos alta subjetividad
                        confidence=best_result['score'],
                        label=label,
                        score=(polarity + 1) * 50
                    )
                
                return None
            
            return await self._run_in_executor(analyze)
        except Exception as e:
            self.logger.warning(f"Transformer analysis failed: {e}")
            return None
    
    def _combine_sentiment_results(self, results: list) -> SentimentMetrics:
        """Combinar múltiples resultados de sentimientos."""
        if len(results) == 1:
            return results[0]
        
        # Promedio ponderado por confidence
        total_weight = sum(r.confidence for r in results)
        
        if total_weight == 0:
            # Si no hay confidence, usar promedio simple
            avg_polarity = sum(r.polarity for r in results) / len(results)
            avg_subjectivity = sum(r.subjectivity for r in results) / len(results)
            avg_score = sum(r.score for r in results) / len(results)
        else:
            # Promedio ponderado
            avg_polarity = sum(r.polarity * r.confidence for r in results) / total_weight
            avg_subjectivity = sum(r.subjectivity * r.confidence for r in results) / total_weight
            avg_score = sum(r.score * r.confidence for r in results) / total_weight
        
        # Confidence combinada (máximo)
        combined_confidence = max(r.confidence for r in results)
        
        return SentimentMetrics(
            polarity=avg_polarity,
            subjectivity=avg_subjectivity,
            confidence=combined_confidence,
            label=self._polarity_to_label(avg_polarity),
            score=avg_score
        )
    
    def _basic_sentiment_analysis(self, text: str) -> SentimentMetrics:
        """Análisis básico de sentimientos usando palabras clave."""
        positive_words = [
            'excelente', 'bueno', 'genial', 'fantástico', 'maravilloso',
            'increíble', 'perfecto', 'amazing', 'excellent', 'good', 
            'great', 'wonderful', 'perfect', 'awesome', 'love'
        ]
        
        negative_words = [
            'malo', 'terrible', 'horrible', 'pésimo', 'awful',
            'bad', 'terrible', 'horrible', 'hate', 'worst',
            'disgusting', 'disappointing', 'fail', 'wrong'
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            polarity = 0.0
        else:
            polarity = (positive_count - negative_count) / total_sentiment_words
        
        return SentimentMetrics(
            polarity=polarity,
            subjectivity=0.5,
            confidence=0.3,  # Baja confidence para método básico
            label=self._polarity_to_label(polarity),
            score=(polarity + 1) * 50
        )
    
    def _polarity_to_label(self, polarity: float) -> str:
        """Convertir polaridad a etiqueta."""
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"
    
    def _apply_cached_result(self, result: NLPAnalysisResult, cached_data: Dict[str, Any]):
        """Aplicar resultado cacheado."""
        if 'sentiment' in cached_data:
            sent_data = cached_data['sentiment']
            result.sentiment = SentimentMetrics(
                polarity=sent_data.get('polarity', 0.0),
                subjectivity=sent_data.get('subjectivity', 0.0),
                confidence=sent_data.get('confidence', 0.0),
                label=sent_data.get('label', 'neutral'),
                score=sent_data.get('score', 50.0)
            ) 