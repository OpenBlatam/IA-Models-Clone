from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import os
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
⚙️ NLP Configuration Models
===========================

Configuración modular para el sistema NLP.
Settings, patterns, thresholds, y configuraciones por módulo.
"""



class ModelQuality(str, Enum):
    """Calidad de modelos NLP."""
    FAST = "fast"          # Modelos rápidos, menor precisión
    BALANCED = "balanced"  # Balance velocidad/precisión
    ACCURATE = "accurate"  # Alta precisión, más lento


class CacheStrategy(str, Enum):
    """Estrategias de cache."""
    NONE = "none"
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"


@dataclass
class PerformanceConfig:
    """Configuración de performance."""
    max_concurrent_analyses: int = 10
    timeout_seconds: float = 30.0
    enable_parallel_processing: bool = True
    cache_strategy: CacheStrategy = CacheStrategy.MEMORY
    cache_size_mb: int = 100
    cache_ttl_seconds: int = 3600
    
    def __post_init__(self) -> Any:
        if self.max_concurrent_analyses <= 0:
            raise ValueError("max_concurrent_analyses must be positive")


@dataclass
class SentimentConfig:
    """Configuración para análisis de sentimientos."""
    model_quality: ModelQuality = ModelQuality.BALANCED
    confidence_threshold: float = 0.7
    intensity_multiplier: float = 1.5
    
    # Lexicon weights
    positive_words: Dict[str, float] = field(default_factory=lambda: {
        'amazing': 0.9, 'awesome': 0.8, 'excellent': 0.9, 'fantastic': 0.8,
        'great': 0.7, 'good': 0.6, 'love': 0.8, 'wonderful': 0.8,
        'perfect': 0.9, 'best': 0.8, 'brilliant': 0.8, 'outstanding': 0.9,
        'incredible': 0.9, 'superb': 0.8, 'marvelous': 0.8, 'fabulous': 0.8
    })
    
    negative_words: Dict[str, float] = field(default_factory=lambda: {
        'terrible': -0.9, 'awful': -0.8, 'horrible': -0.9, 'bad': -0.6,
        'worst': -0.9, 'hate': -0.8, 'disappointing': -0.7, 'sad': -0.6,
        'angry': -0.7, 'frustrated': -0.6, 'annoying': -0.6, 'poor': -0.5,
        'disgusting': -0.9, 'pathetic': -0.7, 'useless': -0.7, 'ridiculous': -0.6
    })
    
    # Threshold for sentiment classification
    positive_threshold: float = 0.1
    negative_threshold: float = -0.1


@dataclass
class EmotionConfig:
    """Configuración para análisis de emociones."""
    model_quality: ModelQuality = ModelQuality.BALANCED
    emotion_threshold: float = 0.1
    diversity_calculation: str = "entropy"  # entropy, variance, range
    
    # Emotion keyword patterns
    emotion_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        'joy': ['happy', 'excited', 'amazing', 'awesome', 'love', 'great', 'wonderful', 'fantastic'],
        'anger': ['angry', 'mad', 'frustrated', 'hate', 'terrible', 'furious', 'outraged'],
        'fear': ['scared', 'worried', 'afraid', 'nervous', 'anxious', 'frightened', 'terrified'],
        'sadness': ['sad', 'disappointed', 'upset', 'depressed', 'heartbroken', 'miserable'],
        'surprise': ['wow', 'amazing', 'incredible', 'unbelievable', 'shocking', 'astonishing'],
        'trust': ['reliable', 'honest', 'trustworthy', 'dependable', 'loyal', 'faithful'],
        'disgust': ['disgusting', 'revolting', 'sick', 'gross', 'awful', 'horrible']
    })


@dataclass
class EngagementConfig:
    """Configuración para predicción de engagement."""
    model_quality: ModelQuality = ModelQuality.BALANCED
    
    # Feature weights for engagement calculation
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        'questions': 0.25,
        'call_to_action': 0.20,
        'emojis': 0.15,
        'hashtags': 0.10,
        'mentions': 0.05,
        'urls': 0.05,
        'urgency': 0.15,
        'social_proof': 0.10,
        'word_count_optimal': 0.15,
        'emotional_intensity': 0.10
    })
    
    # Optimal ranges
    optimal_word_count: tuple[int, int] = (50, 150)
    optimal_emoji_count: tuple[int, int] = (1, 3)
    optimal_hashtag_count: tuple[int, int] = (3, 7)
    
    # Engagement patterns
    question_patterns: List[str] = field(default_factory=lambda: [
        r'\?', r'what do you think', r'tell us', r'share your', r'what\'s your',
        r'how do you', r'do you think', r'would you', r'have you'
    ])
    
    cta_patterns: List[str] = field(default_factory=lambda: [
        r'click', r'visit', r'follow', r'share', r'comment', r'like',
        r'subscribe', r'join', r'sign up', r'download', r'get started'
    ])
    
    urgency_patterns: List[str] = field(default_factory=lambda: [
        r'now', r'today', r'limited time', r'hurry', r'act fast',
        r'don\'t miss', r'expires soon', r'last chance'
    ])
    
    social_proof_patterns: List[str] = field(default_factory=lambda: [
        r'everyone', r'thousands', r'millions', r'most people',
        r'bestselling', r'popular', r'trending', r'viral'
    ])


@dataclass
class ReadabilityConfig:
    """Configuración para análisis de legibilidad."""
    
    # Flesch Reading Ease thresholds
    flesch_thresholds: Dict[str, tuple[float, float]] = field(default_factory=lambda: {
        'very_easy': (90, 100),
        'easy': (80, 90),
        'fairly_easy': (70, 80),
        'standard': (60, 70),
        'fairly_difficult': (50, 60),
        'difficult': (30, 50),
        'very_difficult': (0, 30)
    })
    
    # Syllable counting configuration
    vowels: str = "aeiouy"
    silent_e_penalty: bool = True
    
    # Reading speed (words per minute)
    average_reading_speed: int = 200
    
    # Sentence complexity factors
    max_sentence_length: int = 20  # Words
    complex_word_syllables: int = 3


@dataclass
class TopicsConfig:
    """Configuración para extracción de temas."""
    max_topics: int = 5
    max_keywords: int = 10
    min_keyword_length: int = 3
    
    # Topic categories with keywords
    topic_categories: Dict[str, List[str]] = field(default_factory=lambda: {
        'business': ['business', 'company', 'startup', 'entrepreneur', 'corporate', 'strategy', 'growth'],
        'technology': ['technology', 'tech', 'AI', 'software', 'digital', 'innovation', 'automation'],
        'marketing': ['marketing', 'advertising', 'promotion', 'campaign', 'brand', 'social media'],
        'lifestyle': ['lifestyle', 'life', 'personal', 'wellness', 'health', 'fitness', 'travel'],
        'education': ['education', 'learning', 'study', 'course', 'training', 'skill', 'knowledge'],
        'entertainment': ['entertainment', 'fun', 'game', 'movie', 'music', 'show', 'celebrity'],
        'finance': ['money', 'finance', 'investment', 'economy', 'budget', 'savings', 'profit'],
        'food': ['food', 'recipe', 'cooking', 'restaurant', 'cuisine', 'meal', 'diet'],
        'sports': ['sports', 'football', 'soccer', 'basketball', 'tennis', 'fitness', 'athlete'],
        'fashion': ['fashion', 'style', 'clothing', 'outfit', 'trend', 'design', 'beauty']
    })
    
    # Stop words for keyword extraction
    stop_words: set = field(default_factory=lambda: {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those'
    })


@dataclass
class OptimizationConfig:
    """Configuración para optimización de contenido."""
    target_engagement_score: float = 0.8
    target_sentiment_score: float = 0.3
    max_optimization_iterations: int = 3
    
    # Optimization strategies
    enable_emoji_addition: bool = True
    enable_question_addition: bool = True
    enable_cta_addition: bool = True
    enable_hashtag_optimization: bool = True
    
    # Default optimizations
    default_emoji: str = "✨"
    default_question: str = "What do you think?"
    default_question_emoji: str = "💭"
    
    # Hashtag generation
    max_generated_hashtags: int = 5
    hashtag_relevance_threshold: float = 0.3


@dataclass
class NLPSystemConfig:
    """Configuración principal del sistema NLP."""
    
    # System settings
    system_name: str = "Facebook Posts NLP System"
    version: str = "2.0.0"
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Module configurations
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    engagement: EngagementConfig = field(default_factory=EngagementConfig)
    readability: ReadabilityConfig = field(default_factory=ReadabilityConfig)
    topics: TopicsConfig = field(default_factory=TopicsConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    # Environment overrides
    def load_from_env(self) -> Any:
        """Load configuration from environment variables."""
        
        # Performance settings
        if os.getenv('NLP_MAX_CONCURRENT'):
            self.performance.max_concurrent_analyses = int(os.getenv('NLP_MAX_CONCURRENT'))
        
        if os.getenv('NLP_CACHE_STRATEGY'):
            self.performance.cache_strategy = CacheStrategy(os.getenv('NLP_CACHE_STRATEGY'))
        
        if os.getenv('NLP_DEBUG'):
            self.debug_mode = os.getenv('NLP_DEBUG').lower() == 'true'
        
        if os.getenv('NLP_LOG_LEVEL'):
            self.log_level = os.getenv('NLP_LOG_LEVEL')
        
        # Model quality settings
        model_quality = os.getenv('NLP_MODEL_QUALITY', 'balanced')
        if model_quality in [e.value for e in ModelQuality]:
            quality_enum = ModelQuality(model_quality)
            self.sentiment.model_quality = quality_enum
            self.emotion.model_quality = quality_enum
            self.engagement.model_quality = quality_enum
    
    def get_model_config(self, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Get configuration for specific analysis type."""
        configs = {
            'sentiment': self.sentiment,
            'emotion': self.emotion,
            'engagement': self.engagement,
            'readability': self.readability,
            'topics': self.topics,
            'optimization': self.optimization
        }
        return configs.get(analysis_type)
    
    def validate(self) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []
        
        # Validate performance config
        if self.performance.max_concurrent_analyses <= 0:
            errors.append("max_concurrent_analyses must be positive")
        
        if not 0 <= self.sentiment.confidence_threshold <= 1:
            errors.append("sentiment confidence_threshold must be between 0 and 1")
        
        if not 0 <= self.engagement.target_engagement_score <= 1:
            errors.append("target_engagement_score must be between 0 and 1")
        
        # Validate feature weights sum
        total_weight = sum(self.engagement.feature_weights.values())
        if not 0.9 <= total_weight <= 1.1:  # Allow small floating point errors
            errors.append(f"engagement feature weights sum to {total_weight}, should be ~1.0")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'system_name': self.system_name,
            'version': self.version,
            'debug_mode': self.debug_mode,
            'log_level': self.log_level,
            'performance': {
                'max_concurrent_analyses': self.performance.max_concurrent_analyses,
                'timeout_seconds': self.performance.timeout_seconds,
                'cache_strategy': self.performance.cache_strategy.value,
                'cache_size_mb': self.performance.cache_size_mb
            },
            'sentiment': {
                'model_quality': self.sentiment.model_quality.value,
                'confidence_threshold': self.sentiment.confidence_threshold,
                'positive_threshold': self.sentiment.positive_threshold,
                'negative_threshold': self.sentiment.negative_threshold
            },
            'engagement': {
                'optimal_word_count': self.engagement.optimal_word_count,
                'optimal_emoji_count': self.engagement.optimal_emoji_count,
                'optimal_hashtag_count': self.engagement.optimal_hashtag_count
            },
            'topics': {
                'max_topics': self.topics.max_topics,
                'max_keywords': self.topics.max_keywords,
                'topic_categories': list(self.topics.topic_categories.keys())
            }
        }


# Default global configuration instance
default_config = NLPSystemConfig()

# Load environment variables on import
default_config.load_from_env() 