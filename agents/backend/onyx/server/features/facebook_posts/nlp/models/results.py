from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
📊 NLP Results Models
====================

Modelos de datos para resultados de análisis NLP.
Estructuras de datos reutilizables y type-safe.
"""



class AnalysisType(str, Enum):
    """Tipos de análisis NLP."""
    SENTIMENT = "sentiment"
    EMOTION = "emotion"
    ENGAGEMENT = "engagement"
    READABILITY = "readability"
    TOPICS = "topics"
    OPTIMIZATION = "optimization"


class ContentType(str, Enum):
    """Tipos de contenido detectados."""
    PROMOTIONAL = "promotional"
    EDUCATIONAL = "educational"
    QUESTION = "question"
    NEWS = "news"
    PERSONAL = "personal"
    GENERAL = "general"


class SentimentLabel(str, Enum):
    """Etiquetas de sentimiento."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class AnalysisMetadata:
    """Metadatos de análisis."""
    model_name: str
    confidence: float
    processing_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    
    def __post_init__(self) -> Any:
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class SentimentResult:
    """Resultado de análisis de sentimientos."""
    polarity: float          # -1 (negative) to 1 (positive)
    subjectivity: float      # 0 (objective) to 1 (subjective)
    label: SentimentLabel
    confidence: float
    intensity: float         # Emotional intensity
    metadata: AnalysisMetadata
    
    def is_positive(self) -> bool:
        """Check if sentiment is positive."""
        return self.label == SentimentLabel.POSITIVE
    
    def is_negative(self) -> bool:
        """Check if sentiment is negative."""
        return self.label == SentimentLabel.NEGATIVE
    
    def get_strength(self) -> str:
        """Get sentiment strength description."""
        if abs(self.polarity) > 0.7:
            return "strong"
        elif abs(self.polarity) > 0.3:
            return "moderate"
        else:
            return "weak"


@dataclass
class EmotionResult:
    """Resultado de análisis de emociones."""
    emotions: Dict[str, float]    # emotion_name -> score
    dominant_emotion: str
    emotional_diversity: float   # How many emotions are present
    emotional_stability: float   # Consistency across text
    metadata: AnalysisMetadata
    
    def get_top_emotions(self, n: int = 3) -> List[tuple[str, float]]:
        """Get top N emotions sorted by score."""
        return sorted(self.emotions.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def has_mixed_emotions(self) -> bool:
        """Check if text has mixed emotions."""
        return self.emotional_diversity > 0.6


@dataclass
class EngagementResult:
    """Resultado de predicción de engagement."""
    overall_score: float          # 0-1 overall engagement potential
    virality_potential: float     # 0-1 likelihood to go viral
    interaction_scores: Dict[str, float]  # click, share, comment probabilities
    key_factors: Dict[str, float] # factors contributing to score
    content_type: ContentType
    metadata: AnalysisMetadata
    
    def get_engagement_level(self) -> str:
        """Get engagement level description."""
        if self.overall_score > 0.8:
            return "high"
        elif self.overall_score > 0.5:
            return "medium"
        else:
            return "low"
    
    def get_top_factors(self, n: int = 5) -> List[tuple[str, float]]:
        """Get top factors contributing to engagement."""
        return sorted(self.key_factors.items(), key=lambda x: x[1], reverse=True)[:n]


@dataclass
class ReadabilityResult:
    """Resultado de análisis de legibilidad."""
    flesch_score: float           # Flesch Reading Ease
    complexity_score: float       # Text complexity (0-1)
    reading_time_seconds: float   # Estimated reading time
    grade_level: str             # Reading grade level
    sentence_stats: Dict[str, float]  # avg_length, variety, etc.
    vocabulary_richness: float    # Lexical diversity
    metadata: AnalysisMetadata
    
    def is_easy_to_read(self) -> bool:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        """Check if text is easy to read."""
        return self.flesch_score > 70
    
    def get_readability_level(self) -> str:
        """Get readability level description."""
        if self.flesch_score > 90:
            return "very_easy"
        elif self.flesch_score > 80:
            return "easy"
        elif self.flesch_score > 70:
            return "fairly_easy"
        elif self.flesch_score > 60:
            return "standard"
        elif self.flesch_score > 50:
            return "fairly_difficult"
        elif self.flesch_score > 30:
            return "difficult"
        else:
            return "very_difficult"


@dataclass
class TopicsResult:
    """Resultado de extracción de temas."""
    topics: List[str]             # Detected topics
    keywords: List[str]           # Key terms
    entities: Dict[str, List[str]] # Named entities by type
    categories: List[str]         # Content categories
    topic_coherence: float        # How coherent topics are
    content_focus: float          # How focused the content is
    metadata: AnalysisMetadata
    
    def get_primary_topic(self) -> Optional[str]:
        """Get the primary topic if available."""
        return self.topics[0] if self.topics else None
    
    def has_clear_focus(self) -> bool:
        """Check if content has clear topical focus."""
        return self.content_focus > 0.7


@dataclass
class OptimizationResult:
    """Resultado de optimización de contenido."""
    original_text: str
    optimized_text: str
    improvements: List[str]       # Applied improvements
    score_improvement: float      # How much score improved
    optimization_type: str        # Type of optimization applied
    confidence: float             # Confidence in optimization
    metadata: AnalysisMetadata
    
    def get_improvement_percentage(self) -> float:
        """Get improvement as percentage."""
        return self.score_improvement * 100
    
    def was_successful(self) -> bool:
        """Check if optimization was successful."""
        return self.score_improvement > 0.1


@dataclass
class ComprehensiveNLPResult:
    """Resultado completo de análisis NLP."""
    text: str
    sentiment: Optional[SentimentResult] = None
    emotion: Optional[EmotionResult] = None
    engagement: Optional[EngagementResult] = None
    readability: Optional[ReadabilityResult] = None
    topics: Optional[TopicsResult] = None
    optimization: Optional[OptimizationResult] = None
    
    # Overall metrics
    overall_score: float = 0.0
    processing_time_ms: float = 0.0
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all analyses."""
        summary = {
            'text_length': len(self.text),
            'processing_time_ms': self.processing_time_ms,
            'overall_score': self.overall_score,
            'analyses_performed': []
        }
        
        if self.sentiment:
            summary['sentiment_score'] = self.sentiment.polarity
            summary['sentiment_label'] = self.sentiment.label.value
            summary['analyses_performed'].append('sentiment')
        
        if self.emotion:
            summary['dominant_emotion'] = self.emotion.dominant_emotion
            summary['emotional_diversity'] = self.emotion.emotional_diversity
            summary['analyses_performed'].append('emotion')
        
        if self.engagement:
            summary['engagement_score'] = self.engagement.overall_score
            summary['engagement_level'] = self.engagement.get_engagement_level()
            summary['analyses_performed'].append('engagement')
        
        if self.readability:
            summary['readability_score'] = self.readability.flesch_score
            summary['readability_level'] = self.readability.get_readability_level()
            summary['analyses_performed'].append('readability')
        
        if self.topics:
            summary['primary_topic'] = self.topics.get_primary_topic()
            summary['topics_count'] = len(self.topics.topics)
            summary['analyses_performed'].append('topics')
        
        if self.optimization:
            summary['optimization_improvement'] = self.optimization.get_improvement_percentage()
            summary['analyses_performed'].append('optimization')
        
        return summary
    
    def get_recommendations(self) -> List[str]:
        """Get all recommendations from analyses."""
        recommendations = []
        
        if self.engagement and self.engagement.overall_score < 0.6:
            recommendations.append("Consider adding more engaging elements like questions or calls-to-action")
        
        if self.sentiment and self.sentiment.is_negative():
            recommendations.append("Consider using more positive language to improve sentiment")
        
        if self.readability and not self.readability.is_easy_to_read():
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            recommendations.append("Consider simplifying language for better readability")
        
        if self.topics and not self.topics.has_clear_focus():
            recommendations.append("Consider focusing on fewer topics for clearer messaging")
        
        return recommendations
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export all results to dictionary format."""
        result = {
            'text': self.text,
            'overall_score': self.overall_score,
            'processing_time_ms': self.processing_time_ms,
            'timestamp': self.analysis_timestamp.isoformat(),
            'summary': self.get_summary(),
            'recommendations': self.get_recommendations()
        }
        
        if self.sentiment:
            result['sentiment'] = {
                'polarity': self.sentiment.polarity,
                'subjectivity': self.sentiment.subjectivity,
                'label': self.sentiment.label.value,
                'confidence': self.sentiment.confidence,
                'intensity': self.sentiment.intensity
            }
        
        if self.emotion:
            result['emotion'] = {
                'emotions': self.emotion.emotions,
                'dominant_emotion': self.emotion.dominant_emotion,
                'emotional_diversity': self.emotion.emotional_diversity
            }
        
        if self.engagement:
            result['engagement'] = {
                'overall_score': self.engagement.overall_score,
                'virality_potential': self.engagement.virality_potential,
                'interaction_scores': self.engagement.interaction_scores,
                'content_type': self.engagement.content_type.value
            }
        
        if self.readability:
            result['readability'] = {
                'flesch_score': self.readability.flesch_score,
                'complexity_score': self.readability.complexity_score,
                'reading_time_seconds': self.readability.reading_time_seconds,
                'grade_level': self.readability.grade_level
            }
        
        if self.topics:
            result['topics'] = {
                'topics': self.topics.topics,
                'keywords': self.topics.keywords,
                'categories': self.topics.categories,
                'primary_topic': self.topics.get_primary_topic()
            }
        
        if self.optimization:
            result['optimization'] = {
                'original_text': self.optimization.original_text,
                'optimized_text': self.optimization.optimized_text,
                'improvements': self.optimization.improvements,
                'score_improvement': self.optimization.score_improvement
            }
        
        return result 