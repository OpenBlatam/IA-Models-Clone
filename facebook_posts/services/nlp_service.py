from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import re
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from typing import Any, List, Dict, Optional
"""
🧠 Facebook Posts - Advanced NLP Service
========================================

Sistema NLP avanzado para análisis y optimización de Facebook posts.
Integrado con Clean Architecture y optimizado para performance.
"""


# NLP Libraries (simulated imports - in production would be real)
# import spacy
# import nltk
# from transformers import pipeline, AutoTokenizer, AutoModel
# from textblob import TextBlob
# import numpy as np


class NLPModelType(str, Enum):
    """Tipos de modelos NLP disponibles."""
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    EMOTION_DETECTION = "emotion_detection"
    TOPIC_MODELING = "topic_modeling"
    READABILITY_ANALYSIS = "readability_analysis"
    ENGAGEMENT_PREDICTION = "engagement_prediction"
    CONTENT_OPTIMIZATION = "content_optimization"
    HASHTAG_GENERATION = "hashtag_generation"
    LANGUAGE_DETECTION = "language_detection"


@dataclass
class NLPAnalysisResult:
    """Resultado de análisis NLP."""
    model_type: NLPModelType
    confidence: float
    processing_time_ms: float
    results: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> Any:
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class SentimentAnalysis:
    """Análisis de sentimiento."""
    polarity: float  # -1 (negative) to 1 (positive)
    subjectivity: float  # 0 (objective) to 1 (subjective)
    emotion: str  # dominant emotion
    confidence: float
    emotional_scores: Dict[str, float] = field(default_factory=dict)
    
    def get_sentiment_label(self) -> str:
        """Obtener etiqueta de sentimiento."""
        if self.polarity > 0.1:
            return "positive"
        elif self.polarity < -0.1:
            return "negative"
        else:
            return "neutral"


@dataclass
class EngagementPrediction:
    """Predicción de engagement."""
    engagement_score: float  # 0-1
    virality_potential: float  # 0-1
    click_probability: float  # 0-1
    share_probability: float  # 0-1
    comment_probability: float  # 0-1
    factors: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ContentOptimization:
    """Optimización de contenido."""
    original_score: float
    optimized_score: float
    improvements: Dict[str, Any]
    optimized_text: str
    optimization_steps: List[str]
    confidence: float


class FacebookPostsNLPService:
    """
    Servicio NLP avanzado para Facebook posts.
    
    Características:
    - Análisis de sentimientos multi-dimensional
    - Predicción de engagement con ML
    - Optimización automática de contenido
    - Generación inteligente de hashtags
    - Análisis de legibilidad y complejidad
    - Detección de emociones y tonos
    """
    
    def __init__(self, model_cache_size: int = 100):
        
    """__init__ function."""
self.logger = logging.getLogger(__name__)
        self.model_cache_size = model_cache_size
        
        # Initialize NLP models and pipelines
        self._initialize_nlp_models()
        
        # Performance tracking
        self.performance_metrics = {
            'analyses_performed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Analysis cache
        self._analysis_cache: Dict[str, NLPAnalysisResult] = {}
        
        self.logger.info("FacebookPostsNLPService initialized successfully")
    
    def _initialize_nlp_models(self) -> Any:
        """Inicializar modelos NLP."""
        self.logger.info("Initializing NLP models...")
        
        # En producción, aquí se cargarían los modelos reales
        # self.sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
        # self.emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
        # self.nlp_spacy = spacy.load("en_core_web_sm")
        
        # Para el demo, usamos configuraciones simuladas
        self.models_config = {
            'sentiment': {'model': 'roberta-sentiment', 'loaded': True},
            'emotion': {'model': 'distilroberta-emotion', 'loaded': True},
            'readability': {'model': 'custom-readability', 'loaded': True},
            'engagement': {'model': 'facebook-engagement-predictor', 'loaded': True}
        }
        
        # Patterns for text analysis
        self._initialize_text_patterns()
        
        self.logger.info("NLP models initialized successfully")
    
    def _initialize_text_patterns(self) -> Any:
        """Inicializar patrones de texto."""
        self.patterns = {
            'questions': [r'\?', r'what\s+do\s+you\s+think', r'tell\s+us', r'share\s+your'],
            'call_to_action': [r'click', r'visit', r'follow', r'subscribe', r'share', r'comment'],
            'urgency': [r'now', r'today', r'limited\s+time', r'hurry', r'act\s+fast'],
            'emotions': [r'amazing', r'awesome', r'incredible', r'fantastic', r'love', r'hate'],
            'social_proof': [r'thousands', r'millions', r'everyone', r'people\s+are'],
            'numbers': r'\d+',
            'hashtags': r'#\w+',
            'mentions': r'@\w+',
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        }
    
    async def analyze_facebook_post(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, NLPAnalysisResult]:
        """
        Análisis NLP completo de Facebook post.
        
        Args:
            text: Contenido del post
            metadata: Metadatos adicionales
            
        Returns:
            Diccionario con todos los análisis NLP
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("Starting comprehensive NLP analysis")
            
            # Check cache first
            cache_key = self._generate_cache_key(text, metadata)
            cached_results = self._get_cached_analysis(cache_key)
            if cached_results:
                self.performance_metrics['cache_hits'] += 1
                return cached_results
            
            self.performance_metrics['cache_misses'] += 1
            
            # Perform all NLP analyses in parallel
            analyses = await asyncio.gather(
                self._analyze_sentiment(text),
                self._analyze_emotions(text),
                self._analyze_readability(text),
                self._predict_engagement(text, metadata or {}),
                self._analyze_topics(text),
                self._analyze_language_features(text),
                return_exceptions=True
            )
            
            # Compile results
            results = {}
            analysis_types = [
                'sentiment', 'emotions', 'readability', 
                'engagement', 'topics', 'language_features'
            ]
            
            for i, analysis in enumerate(analyses):
                if not isinstance(analysis, Exception):
                    results[analysis_types[i]] = analysis
                else:
                    self.logger.warning(f"Analysis {analysis_types[i]} failed: {analysis}")
            
            # Cache results
            self._cache_analysis(cache_key, results)
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_metrics(processing_time)
            
            self.logger.info(f"NLP analysis completed in {processing_time:.2f}ms")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in NLP analysis: {e}")
            return self._get_fallback_analysis(text)
    
    async def _analyze_sentiment(self, text: str) -> NLPAnalysisResult:
        """Análisis avanzado de sentimientos."""
        start_time = datetime.now()
        
        try:
            # Simulated sentiment analysis (in production, would use real models)
            polarity = await self._calculate_polarity(text)
            subjectivity = await self._calculate_subjectivity(text)
            emotions = await self._detect_emotions(text)
            
            # Get dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            confidence = max(emotions.values())
            
            sentiment = SentimentAnalysis(
                polarity=polarity,
                subjectivity=subjectivity,
                emotion=dominant_emotion,
                confidence=confidence,
                emotional_scores=emotions
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return NLPAnalysisResult(
                model_type=NLPModelType.SENTIMENT_ANALYSIS,
                confidence=confidence,
                processing_time_ms=processing_time,
                results={
                    'sentiment': sentiment,
                    'polarity': polarity,
                    'subjectivity': subjectivity,
                    'emotions': emotions,
                    'sentiment_label': sentiment.get_sentiment_label()
                },
                metadata={'model': 'roberta-sentiment-v2'}
            )
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return self._get_fallback_sentiment_analysis()
    
    async def _analyze_emotions(self, text: str) -> NLPAnalysisResult:
        """Análisis detallado de emociones."""
        start_time = datetime.now()
        
        try:
            # Emotion detection simulation
            emotions = await self._detect_advanced_emotions(text)
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            
            # Emotional intensity
            intensity = await self._calculate_emotional_intensity(text)
            
            # Emotional stability (consistency across text)
            stability = await self._calculate_emotional_stability(text)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return NLPAnalysisResult(
                model_type=NLPModelType.EMOTION_DETECTION,
                confidence=dominant_emotion[1],
                processing_time_ms=processing_time,
                results={
                    'emotions': emotions,
                    'dominant_emotion': dominant_emotion[0],
                    'emotional_intensity': intensity,
                    'emotional_stability': stability,
                    'emotion_distribution': self._calculate_emotion_distribution(emotions)
                },
                metadata={'model': 'distilroberta-emotion-v2'}
            )
            
        except Exception as e:
            self.logger.error(f"Emotion analysis failed: {e}")
            return self._get_fallback_emotion_analysis()
    
    async def _analyze_readability(self, text: str) -> NLPAnalysisResult:
        """Análisis de legibilidad y complejidad."""
        start_time = datetime.now()
        
        try:
            # Readability metrics
            flesch_score = await self._calculate_flesch_score(text)
            complexity_score = await self._calculate_complexity_score(text)
            reading_time = await self._calculate_reading_time(text)
            
            # Advanced metrics
            sentence_variety = await self._analyze_sentence_variety(text)
            vocabulary_richness = await self._calculate_vocabulary_richness(text)
            clarity_score = await self._calculate_clarity_score(text)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return NLPAnalysisResult(
                model_type=NLPModelType.READABILITY_ANALYSIS,
                confidence=0.9,
                processing_time_ms=processing_time,
                results={
                    'flesch_score': flesch_score,
                    'complexity_score': complexity_score,
                    'reading_time_seconds': reading_time,
                    'sentence_variety': sentence_variety,
                    'vocabulary_richness': vocabulary_richness,
                    'clarity_score': clarity_score,
                    'readability_grade': self._get_readability_grade(flesch_score)
                },
                metadata={'model': 'custom-readability-analyzer'}
            )
            
        except Exception as e:
            self.logger.error(f"Readability analysis failed: {e}")
            return self._get_fallback_readability_analysis()
    
    async def _predict_engagement(self, text: str, metadata: Dict[str, Any]) -> NLPAnalysisResult:
        """Predicción avanzada de engagement."""
        start_time = datetime.now()
        
        try:
            # Feature extraction for engagement prediction
            features = await self._extract_engagement_features(text, metadata)
            
            # Engagement prediction
            engagement_score = await self._calculate_engagement_score(features)
            virality_potential = await self._calculate_virality_potential(features)
            
            # Specific action predictions
            click_prob = await self._predict_click_probability(features)
            share_prob = await self._predict_share_probability(features)
            comment_prob = await self._predict_comment_probability(features)
            
            # Optimization recommendations
            recommendations = await self._generate_engagement_recommendations(features)
            
            prediction = EngagementPrediction(
                engagement_score=engagement_score,
                virality_potential=virality_potential,
                click_probability=click_prob,
                share_probability=share_prob,
                comment_probability=comment_prob,
                factors=features,
                recommendations=recommendations
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return NLPAnalysisResult(
                model_type=NLPModelType.ENGAGEMENT_PREDICTION,
                confidence=0.85,
                processing_time_ms=processing_time,
                results={
                    'prediction': prediction,
                    'engagement_score': engagement_score,
                    'virality_potential': virality_potential,
                    'action_probabilities': {
                        'click': click_prob,
                        'share': share_prob,
                        'comment': comment_prob
                    },
                    'key_factors': self._get_top_engagement_factors(features),
                    'recommendations': recommendations
                },
                metadata={'model': 'facebook-engagement-predictor-v3'}
            )
            
        except Exception as e:
            self.logger.error(f"Engagement prediction failed: {e}")
            return self._get_fallback_engagement_prediction()
    
    async def _analyze_topics(self, text: str) -> NLPAnalysisResult:
        """Análisis de temas y entidades."""
        start_time = datetime.now()
        
        try:
            # Topic extraction
            topics = await self._extract_topics(text)
            
            # Named entity recognition
            entities = await self._extract_entities(text)
            
            # Keywords extraction
            keywords = await self._extract_keywords(text)
            
            # Category classification
            categories = await self._classify_content_categories(text)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return NLPAnalysisResult(
                model_type=NLPModelType.TOPIC_MODELING,
                confidence=0.8,
                processing_time_ms=processing_time,
                results={
                    'topics': topics,
                    'entities': entities,
                    'keywords': keywords,
                    'categories': categories,
                    'topic_coherence': await self._calculate_topic_coherence(topics),
                    'content_focus': await self._analyze_content_focus(text)
                },
                metadata={'model': 'bert-topic-modeling'}
            )
            
        except Exception as e:
            self.logger.error(f"Topic analysis failed: {e}")
            return self._get_fallback_topic_analysis()
    
    async def _analyze_language_features(self, text: str) -> NLPAnalysisResult:
        """Análisis de características lingüísticas."""
        start_time = datetime.now()
        
        try:
            # Language detection
            language = await self._detect_language(text)
            
            # Linguistic features
            pos_tags = await self._analyze_pos_tags(text)
            syntax_complexity = await self._analyze_syntax_complexity(text)
            
            # Style analysis
            formality_score = await self._calculate_formality_score(text)
            creativity_score = await self._calculate_creativity_score(text)
            
            # Text statistics
            text_stats = await self._calculate_text_statistics(text)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return NLPAnalysisResult(
                model_type=NLPModelType.LANGUAGE_DETECTION,
                confidence=0.9,
                processing_time_ms=processing_time,
                results={
                    'language': language,
                    'pos_distribution': pos_tags,
                    'syntax_complexity': syntax_complexity,
                    'formality_score': formality_score,
                    'creativity_score': creativity_score,
                    'text_statistics': text_stats,
                    'linguistic_patterns': await self._detect_linguistic_patterns(text)
                },
                metadata={'model': 'spacy-multilingual'}
            )
            
        except Exception as e:
            self.logger.error(f"Language analysis failed: {e}")
            return self._get_fallback_language_analysis()
    
    async def optimize_content_for_engagement(
        self,
        text: str,
        target_metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContentOptimization:
        """
        Optimizar contenido para mayor engagement.
        
        Args:
            text: Contenido original
            target_metrics: Métricas objetivo
            metadata: Metadatos adicionales
            
        Returns:
            Optimización del contenido
        """
        try:
            self.logger.info("Starting content optimization")
            
            # Analyze current content
            current_analysis = await self.analyze_facebook_post(text, metadata)
            original_score = self._calculate_overall_score(current_analysis)
            
            # Generate optimization suggestions
            optimizations = await self._generate_optimization_strategies(
                text, current_analysis, target_metrics
            )
            
            # Apply optimizations
            optimized_text = await self._apply_optimizations(text, optimizations)
            
            # Analyze optimized content
            optimized_analysis = await self.analyze_facebook_post(optimized_text, metadata)
            optimized_score = self._calculate_overall_score(optimized_analysis)
            
            return ContentOptimization(
                original_score=original_score,
                optimized_score=optimized_score,
                improvements=optimizations,
                optimized_text=optimized_text,
                optimization_steps=optimizations.get('steps', []),
                confidence=min(optimized_score / original_score, 1.0) if original_score > 0 else 0.8
            )
            
        except Exception as e:
            self.logger.error(f"Content optimization failed: {e}")
            return self._get_fallback_optimization(text)
    
    async def generate_hashtags(
        self,
        text: str,
        max_hashtags: int = 10,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generar hashtags inteligentes basados en el contenido.
        
        Args:
            text: Contenido del post
            max_hashtags: Máximo número de hashtags
            metadata: Metadatos adicionales
            
        Returns:
            Lista de hashtags optimizados
        """
        try:
            self.logger.info("Generating intelligent hashtags")
            
            # Extract topics and keywords
            topics_result = await self._analyze_topics(text)
            topics = topics_result.results.get('topics', [])
            keywords = topics_result.results.get('keywords', [])
            
            # Analyze sentiment for hashtag tone
            sentiment_result = await self._analyze_sentiment(text)
            sentiment = sentiment_result.results.get('sentiment_label', 'neutral')
            
            # Generate hashtag candidates
            hashtag_candidates = await self._generate_hashtag_candidates(
                topics, keywords, sentiment, metadata or {}
            )
            
            # Score and rank hashtags
            scored_hashtags = await self._score_hashtags(hashtag_candidates, text)
            
            # Return top hashtags
            return scored_hashtags[:max_hashtags]
            
        except Exception as e:
            self.logger.error(f"Hashtag generation failed: {e}")
            return self._get_fallback_hashtags(text)
    
    # ===== PRIVATE ANALYSIS METHODS =====
    
    async def _calculate_polarity(self, text: str) -> float:
        """Calcular polaridad del sentimiento."""
        # Simplified sentiment calculation
        positive_words = ['good', 'great', 'awesome', 'amazing', 'love', 'best', 'perfect', 'excellent']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / len(words)
    
    async def _calculate_subjectivity(self, text: str) -> float:
        """Calcular subjetividad del texto."""
        subjective_patterns = ['i think', 'i believe', 'i feel', 'in my opinion', 'personally']
        objective_patterns = ['studies show', 'research indicates', 'data suggests', 'according to']
        
        text_lower = text.lower()
        subjective_count = sum(1 for pattern in subjective_patterns if pattern in text_lower)
        objective_count = sum(1 for pattern in objective_patterns if pattern in text_lower)
        
        # Base subjectivity on presence of subjective indicators
        return min(0.5 + (subjective_count * 0.2) - (objective_count * 0.1), 1.0)
    
    async def _detect_emotions(self, text: str) -> Dict[str, float]:
        """Detectar emociones en el texto."""
        # Simplified emotion detection
        emotion_keywords = {
            'joy': ['happy', 'excited', 'joy', 'delighted', 'thrilled', 'elated'],
            'anger': ['angry', 'furious', 'mad', 'annoyed', 'irritated'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous'],
            'sadness': ['sad', 'depressed', 'upset', 'disappointed', 'hurt'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished'],
            'trust': ['trust', 'confident', 'secure', 'reliable', 'believe']
        }
        
        words = text.lower().split()
        emotions = {}
        
        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for word in words if word in keywords)
            emotions[emotion] = min(count / len(words) * 10, 1.0)  # Normalize
        
        # Ensure at least one emotion has some value
        if all(score == 0 for score in emotions.values()):
            emotions['neutral'] = 0.8
        
        return emotions
    
    async def _extract_engagement_features(self, text: str, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Extraer características para predicción de engagement."""
        features = {}
        
        # Text length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.findall(r'[.!?]+', text))
        
        # Engagement indicators
        features['has_question'] = 1.0 if '?' in text else 0.0
        features['has_exclamation'] = 1.0 if '!' in text else 0.0
        features['emoji_count'] = len(re.findall(r'[😀-🿿]', text))
        
        # Call to action indicators
        cta_patterns = ['click', 'visit', 'follow', 'share', 'comment', 'like', 'subscribe']
        features['has_cta'] = 1.0 if any(pattern in text.lower() for pattern in cta_patterns) else 0.0
        
        # Social proof indicators
        social_proof_patterns = ['people', 'everyone', 'thousands', 'millions', 'users']
        features['has_social_proof'] = 1.0 if any(pattern in text.lower() for pattern in social_proof_patterns) else 0.0
        
        # Urgency indicators
        urgency_patterns = ['now', 'today', 'limited', 'hurry', 'fast', 'deadline']
        features['has_urgency'] = 1.0 if any(pattern in text.lower() for pattern in urgency_patterns) else 0.0
        
        # Hashtag and mention features
        features['hashtag_count'] = len(re.findall(r'#\w+', text))
        features['mention_count'] = len(re.findall(r'@\w+', text))
        
        # URL features
        features['has_url'] = 1.0 if re.search(r'http[s]?://', text) else 0.0
        
        return features
    
    async def _calculate_engagement_score(self, features: Dict[str, float]) -> float:
        """Calcular score de engagement basado en características."""
        # Weighted scoring model
        weights = {
            'has_question': 0.15,
            'has_cta': 0.20,
            'emoji_count': 0.10,
            'has_social_proof': 0.15,
            'has_urgency': 0.10,
            'hashtag_count': 0.10,
            'word_count_optimal': 0.20  # Optimal word count range
        }
        
        score = 0.0
        
        # Apply weights to features
        for feature, weight in weights.items():
            if feature == 'word_count_optimal':
                # Optimal word count is 80-150 words
                word_count = features.get('word_count', 0)
                if 80 <= word_count <= 150:
                    score += weight
                elif 50 <= word_count < 80 or 150 < word_count <= 200:
                    score += weight * 0.7
            elif feature == 'emoji_count':
                # 1-3 emojis is optimal
                emoji_count = features.get('emoji_count', 0)
                if 1 <= emoji_count <= 3:
                    score += weight
                elif emoji_count > 0:
                    score += weight * 0.5
            else:
                score += features.get(feature, 0) * weight
        
        return min(score, 1.0)
    
    # ===== UTILITY METHODS =====
    
    def _generate_cache_key(self, text: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Generar clave de cache para análisis."""
        content = text + str(sorted((metadata or {}).items()))
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_analysis(self, cache_key: str) -> Optional[Dict[str, NLPAnalysisResult]]:
        """Obtener análisis del cache."""
        return self._analysis_cache.get(cache_key)
    
    def _cache_analysis(self, cache_key: str, results: Dict[str, NLPAnalysisResult]):
        """Cachear resultados de análisis."""
        if len(self._analysis_cache) >= self.model_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._analysis_cache))
            del self._analysis_cache[oldest_key]
        
        self._analysis_cache[cache_key] = results
    
    def _update_performance_metrics(self, processing_time: float):
        """Actualizar métricas de performance."""
        self.performance_metrics['analyses_performed'] += 1
        self.performance_metrics['total_processing_time'] += processing_time
        
        count = self.performance_metrics['analyses_performed']
        total_time = self.performance_metrics['total_processing_time']
        self.performance_metrics['average_processing_time'] = total_time / count
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de performance."""
        cache_total = self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']
        cache_hit_rate = (self.performance_metrics['cache_hits'] / cache_total) if cache_total > 0 else 0
        
        return {
            **self.performance_metrics,
            'cache_hit_rate': cache_hit_rate,
            'models_loaded': len(self.models_config),
            'cache_size': len(self._analysis_cache)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del servicio NLP."""
        try:
            # Test basic functionality
            test_text = "This is a test for NLP service health check."
            test_result = await self._analyze_sentiment(test_text)
            
            return {
                'status': 'healthy',
                'service': 'FacebookPostsNLPService',
                'models_loaded': len(self.models_config),
                'test_analysis': test_result.confidence > 0,
                'performance_metrics': self.get_performance_metrics()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'service': 'FacebookPostsNLPService',
                'error': str(e)
            }
    
    # ===== FALLBACK METHODS =====
    
    def _get_fallback_analysis(self, text: str) -> Dict[str, NLPAnalysisResult]:
        """Análisis de fallback cuando falla el análisis principal."""
        return {
            'sentiment': self._get_fallback_sentiment_analysis(),
            'engagement': self._get_fallback_engagement_prediction(),
            'readability': self._get_fallback_readability_analysis()
        }
    
    def _get_fallback_sentiment_analysis(self) -> NLPAnalysisResult:
        """Análisis de sentimiento de fallback."""
        return NLPAnalysisResult(
            model_type=NLPModelType.SENTIMENT_ANALYSIS,
            confidence=0.5,
            processing_time_ms=1.0,
            results={
                'sentiment_label': 'neutral',
                'polarity': 0.0,
                'subjectivity': 0.5,
                'emotions': {'neutral': 0.8}
            },
            metadata={'fallback': True}
        )
    
    # ... (more fallback methods would be implemented here)


# ===== HELPER FUNCTIONS =====

async def create_nlp_service() -> FacebookPostsNLPService:
    """Factory function para crear servicio NLP."""
    return FacebookPostsNLPService()


def get_nlp_analysis_summary(results: Dict[str, NLPAnalysisResult]) -> Dict[str, Any]:
    """Obtener resumen de análisis NLP."""
    summary = {
        'overall_confidence': 0.0,
        'total_processing_time': 0.0,
        'models_used': [],
        'key_insights': []
    }
    
    if not results:
        return summary
    
    # Calculate averages
    confidences = [r.confidence for r in results.values()]
    processing_times = [r.processing_time_ms for r in results.values()]
    
    summary['overall_confidence'] = sum(confidences) / len(confidences)
    summary['total_processing_time'] = sum(processing_times)
    summary['models_used'] = [r.model_type.value for r in results.values()]
    
    # Extract key insights
    for analysis_type, result in results.items():
        if analysis_type == 'sentiment':
            sentiment_label = result.results.get('sentiment_label', 'neutral')
            summary['key_insights'].append(f"Sentiment: {sentiment_label}")
        elif analysis_type == 'engagement':
            engagement_score = result.results.get('engagement_score', 0)
            summary['key_insights'].append(f"Engagement potential: {engagement_score:.2f}")
    
    return summary 