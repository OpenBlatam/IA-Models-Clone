"""
Advanced Emotion and Sentiment Analysis System
==============================================

Comprehensive emotion and sentiment analysis with multiple models,
emotion detection, and advanced psychological profiling.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from pathlib import Path
import re
import statistics
from collections import Counter

logger = logging.getLogger(__name__)

class EmotionType(Enum):
    """Emotion types"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    LOVE = "love"
    HATE = "hate"
    EXCITEMENT = "excitement"
    ANXIETY = "anxiety"
    CONFUSION = "confusion"
    SATISFACTION = "satisfaction"
    FRUSTRATION = "frustration"
    HOPE = "hope"
    DESPAIR = "despair"
    PRIDE = "pride"
    SHAME = "shame"
    GRATITUDE = "gratitude"

class SentimentPolarity(Enum):
    """Sentiment polarity"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"

class SentimentIntensity(Enum):
    """Sentiment intensity"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class EmotionScore:
    """Emotion score"""
    emotion: EmotionType
    score: float
    confidence: float
    intensity: SentimentIntensity

@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    id: str
    text: str
    overall_sentiment: SentimentPolarity
    sentiment_score: float
    sentiment_intensity: SentimentIntensity
    emotions: List[EmotionScore]
    dominant_emotion: EmotionType
    emotional_complexity: float
    psychological_indicators: Dict[str, Any]
    language_patterns: Dict[str, Any]
    analyzed_at: datetime
    processing_time: float
    model_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmotionProfile:
    """Emotion profile for a user or document"""
    id: str
    name: str
    total_analyses: int
    emotion_distribution: Dict[EmotionType, float]
    sentiment_trend: List[float]
    dominant_emotions: List[EmotionType]
    emotional_stability: float
    created_at: datetime
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedEmotionAnalyzer:
    """
    Advanced emotion and sentiment analysis system
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize emotion analyzer
        
        Args:
            models_dir: Directory for emotion models
        """
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Emotion profiles
        self.emotion_profiles: Dict[str, EmotionProfile] = {}
        
        # Emotion dictionaries
        self.emotion_lexicons = self._initialize_emotion_lexicons()
        self.sentiment_lexicons = self._initialize_sentiment_lexicons()
        
        # Psychological indicators
        self.psychological_indicators = self._initialize_psychological_indicators()
        
        # Language patterns
        self.language_patterns = self._initialize_language_patterns()
        
        # Analysis history
        self.analysis_history: List[SentimentResult] = []
    
    def _initialize_emotion_lexicons(self) -> Dict[EmotionType, List[str]]:
        """Initialize emotion lexicons"""
        return {
            EmotionType.JOY: [
                "happy", "joyful", "cheerful", "delighted", "ecstatic", "elated",
                "excited", "thrilled", "blissful", "content", "pleased", "satisfied",
                "amazing", "wonderful", "fantastic", "brilliant", "excellent", "great"
            ],
            EmotionType.SADNESS: [
                "sad", "depressed", "melancholy", "gloomy", "sorrowful", "mournful",
                "heartbroken", "devastated", "disappointed", "discouraged", "hopeless",
                "miserable", "unhappy", "down", "blue", "tearful", "weepy"
            ],
            EmotionType.ANGER: [
                "angry", "mad", "furious", "rage", "irritated", "annoyed", "frustrated",
                "outraged", "livid", "enraged", "hostile", "aggressive", "bitter",
                "resentful", "indignant", "wrathful", "incensed"
            ],
            EmotionType.FEAR: [
                "afraid", "scared", "terrified", "frightened", "anxious", "worried",
                "nervous", "concerned", "apprehensive", "alarmed", "panicked", "horrified",
                "dread", "uneasy", "restless", "tense"
            ],
            EmotionType.SURPRISE: [
                "surprised", "shocked", "amazed", "astonished", "stunned", "bewildered",
                "confused", "perplexed", "puzzled", "startled", "taken aback", "flabbergasted",
                "dumbfounded", "speechless", "incredible", "unbelievable"
            ],
            EmotionType.DISGUST: [
                "disgusted", "revolted", "repulsed", "sickened", "nauseated", "appalled",
                "horrified", "offended", "repugnant", "abhorrent", "loathsome", "detestable",
                "vile", "gross", "yucky", "nasty"
            ],
            EmotionType.TRUST: [
                "trust", "confident", "secure", "reliable", "faithful", "loyal",
                "dependable", "honest", "sincere", "genuine", "authentic", "credible",
                "believable", "trustworthy", "safe", "protected"
            ],
            EmotionType.ANTICIPATION: [
                "excited", "eager", "enthusiastic", "hopeful", "optimistic", "expectant",
                "anxious", "nervous", "impatient", "restless", "curious", "interested",
                "motivated", "inspired", "determined", "focused"
            ],
            EmotionType.LOVE: [
                "love", "adore", "cherish", "treasure", "affection", "fondness",
                "passion", "romance", "devotion", "attachment", "bond", "connection",
                "intimacy", "warmth", "tenderness", "care"
            ],
            EmotionType.HATE: [
                "hate", "despise", "loathe", "detest", "abhor", "resent", "dislike",
                "contempt", "scorn", "disdain", "aversion", "antipathy", "hostility",
                "animosity", "enmity", "malice", "spite"
            ]
        }
    
    def _initialize_sentiment_lexicons(self) -> Dict[str, float]:
        """Initialize sentiment lexicons"""
        return {
            # Positive words
            "excellent": 0.9, "amazing": 0.8, "wonderful": 0.8, "fantastic": 0.8,
            "great": 0.7, "good": 0.6, "nice": 0.5, "okay": 0.3, "fine": 0.4,
            "perfect": 0.9, "brilliant": 0.8, "outstanding": 0.8, "superb": 0.8,
            "marvelous": 0.8, "terrific": 0.7, "awesome": 0.7, "incredible": 0.8,
            
            # Negative words
            "terrible": -0.9, "awful": -0.8, "horrible": -0.8, "disgusting": -0.8,
            "bad": -0.6, "poor": -0.5, "worst": -0.9, "pathetic": -0.7,
            "disappointing": -0.6, "frustrating": -0.5, "annoying": -0.4,
            "hate": -0.8, "despise": -0.8, "loathe": -0.8, "detest": -0.8,
            
            # Intensifiers
            "very": 1.5, "extremely": 2.0, "incredibly": 2.0, "absolutely": 1.8,
            "totally": 1.5, "completely": 1.5, "utterly": 1.8, "entirely": 1.3,
            "somewhat": 0.7, "slightly": 0.5, "barely": 0.3, "hardly": 0.2,
            
            # Negators
            "not": -1.0, "never": -1.0, "no": -1.0, "none": -1.0,
            "nothing": -1.0, "nobody": -1.0, "nowhere": -1.0, "neither": -1.0
        }
    
    def _initialize_psychological_indicators(self) -> Dict[str, List[str]]:
        """Initialize psychological indicators"""
        return {
            "depression_indicators": [
                "hopeless", "worthless", "empty", "numb", "tired", "exhausted",
                "overwhelmed", "stuck", "trapped", "alone", "isolated", "lost"
            ],
            "anxiety_indicators": [
                "worried", "nervous", "anxious", "panic", "fear", "scared",
                "uneasy", "restless", "tense", "stressed", "overwhelmed", "racing"
            ],
            "anger_indicators": [
                "furious", "rage", "angry", "mad", "irritated", "frustrated",
                "hostile", "aggressive", "bitter", "resentful", "outraged"
            ],
            "positive_indicators": [
                "grateful", "blessed", "fortunate", "lucky", "happy", "joyful",
                "content", "satisfied", "fulfilled", "proud", "accomplished"
            ],
            "stress_indicators": [
                "stressed", "overwhelmed", "pressure", "deadline", "busy",
                "rushed", "hectic", "chaotic", "demanding", "exhausting"
            ]
        }
    
    def _initialize_language_patterns(self) -> Dict[str, List[str]]:
        """Initialize language patterns"""
        return {
            "certainty_patterns": [
                r"\b(always|never|all|every|none|nothing|everything)\b",
                r"\b(definitely|certainly|absolutely|surely|undoubtedly)\b",
                r"\b(must|have to|need to|should|ought to)\b"
            ],
            "uncertainty_patterns": [
                r"\b(maybe|perhaps|possibly|might|could|probably)\b",
                r"\b(sort of|kind of|somewhat|rather|quite)\b",
                r"\b(I think|I believe|I guess|I suppose|I assume)\b"
            ],
            "emotional_intensity": [
                r"\b(so|very|extremely|incredibly|absolutely|totally)\b",
                r"\b(amazing|incredible|fantastic|terrible|awful|horrible)\b",
                r"!{2,}", r"\?{2,}"  # Multiple punctuation
            ],
            "question_patterns": [
                r"\b(why|what|how|when|where|who)\b",
                r"\?", r"\b(question|wonder|curious|confused)\b"
            ],
            "negation_patterns": [
                r"\b(not|no|never|nothing|nobody|nowhere|neither|nor)\b",
                r"\b(can't|couldn't|won't|wouldn't|shouldn't|don't|doesn't|didn't)\b"
            ]
        }
    
    async def analyze_emotion(self, text: str, user_id: Optional[str] = None) -> SentimentResult:
        """
        Analyze emotion and sentiment in text
        
        Args:
            text: Text to analyze
            user_id: Optional user ID for profile tracking
            
        Returns:
            Sentiment analysis result
        """
        start_time = datetime.now()
        result_id = str(uuid.uuid4())
        
        try:
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Analyze emotions
            emotions = self._analyze_emotions(cleaned_text)
            
            # Analyze sentiment
            sentiment_score, sentiment_polarity = self._analyze_sentiment(cleaned_text)
            
            # Determine sentiment intensity
            sentiment_intensity = self._determine_sentiment_intensity(sentiment_score)
            
            # Find dominant emotion
            dominant_emotion = max(emotions, key=lambda e: e.score).emotion if emotions else EmotionType.NEUTRAL
            
            # Calculate emotional complexity
            emotional_complexity = self._calculate_emotional_complexity(emotions)
            
            # Analyze psychological indicators
            psychological_indicators = self._analyze_psychological_indicators(cleaned_text)
            
            # Analyze language patterns
            language_patterns = self._analyze_language_patterns(cleaned_text)
            
            # Create result
            result = SentimentResult(
                id=result_id,
                text=text,
                overall_sentiment=sentiment_polarity,
                sentiment_score=sentiment_score,
                sentiment_intensity=sentiment_intensity,
                emotions=emotions,
                dominant_emotion=dominant_emotion,
                emotional_complexity=emotional_complexity,
                psychological_indicators=psychological_indicators,
                language_patterns=language_patterns,
                analyzed_at=datetime.now(),
                processing_time=(datetime.now() - start_time).total_seconds(),
                model_used="advanced_emotion_analyzer_v1.0",
                metadata={
                    "text_length": len(text),
                    "word_count": len(text.split()),
                    "user_id": user_id
                }
            )
            
            # Store in history
            self.analysis_history.append(result)
            
            # Update user profile if provided
            if user_id:
                self._update_emotion_profile(user_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing emotion: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-]', '', text)
        
        return text.strip()
    
    def _analyze_emotions(self, text: str) -> List[EmotionScore]:
        """Analyze emotions in text"""
        emotions = []
        words = text.split()
        
        for emotion_type, lexicon in self.emotion_lexicons.items():
            score = 0
            matches = 0
            
            for word in words:
                if word in lexicon:
                    score += 1
                    matches += 1
            
            if matches > 0:
                # Normalize score
                normalized_score = score / len(words)
                confidence = min(1.0, matches / 10)  # Confidence based on number of matches
                intensity = self._determine_emotion_intensity(normalized_score)
                
                emotions.append(EmotionScore(
                    emotion=emotion_type,
                    score=normalized_score,
                    confidence=confidence,
                    intensity=intensity
                ))
        
        # Sort by score descending
        emotions.sort(key=lambda e: e.score, reverse=True)
        
        return emotions
    
    def _analyze_sentiment(self, text: str) -> Tuple[float, SentimentPolarity]:
        """Analyze sentiment in text"""
        words = text.split()
        sentiment_score = 0
        word_count = 0
        
        for word in words:
            if word in self.sentiment_lexicons:
                sentiment_score += self.sentiment_lexicons[word]
                word_count += 1
        
        # Normalize score
        if word_count > 0:
            sentiment_score = sentiment_score / word_count
        else:
            sentiment_score = 0
        
        # Determine polarity
        if sentiment_score >= 0.6:
            polarity = SentimentPolarity.VERY_POSITIVE
        elif sentiment_score >= 0.2:
            polarity = SentimentPolarity.POSITIVE
        elif sentiment_score <= -0.6:
            polarity = SentimentPolarity.VERY_NEGATIVE
        elif sentiment_score <= -0.2:
            polarity = SentimentPolarity.NEGATIVE
        else:
            polarity = SentimentPolarity.NEUTRAL
        
        return sentiment_score, polarity
    
    def _determine_sentiment_intensity(self, score: float) -> SentimentIntensity:
        """Determine sentiment intensity"""
        abs_score = abs(score)
        
        if abs_score >= 0.8:
            return SentimentIntensity.VERY_HIGH
        elif abs_score >= 0.6:
            return SentimentIntensity.HIGH
        elif abs_score >= 0.4:
            return SentimentIntensity.MODERATE
        elif abs_score >= 0.2:
            return SentimentIntensity.LOW
        else:
            return SentimentIntensity.VERY_LOW
    
    def _determine_emotion_intensity(self, score: float) -> SentimentIntensity:
        """Determine emotion intensity"""
        if score >= 0.1:
            return SentimentIntensity.HIGH
        elif score >= 0.05:
            return SentimentIntensity.MODERATE
        elif score >= 0.02:
            return SentimentIntensity.LOW
        else:
            return SentimentIntensity.VERY_LOW
    
    def _calculate_emotional_complexity(self, emotions: List[EmotionScore]) -> float:
        """Calculate emotional complexity"""
        if not emotions:
            return 0.0
        
        # Count emotions with significant scores
        significant_emotions = [e for e in emotions if e.score > 0.01]
        
        # Calculate diversity
        diversity = len(significant_emotions) / len(EmotionType)
        
        # Calculate intensity variance
        if len(significant_emotions) > 1:
            scores = [e.score for e in significant_emotions]
            variance = statistics.variance(scores)
        else:
            variance = 0
        
        # Combine diversity and variance
        complexity = (diversity * 0.7) + (min(variance, 1.0) * 0.3)
        
        return min(1.0, complexity)
    
    def _analyze_psychological_indicators(self, text: str) -> Dict[str, Any]:
        """Analyze psychological indicators"""
        indicators = {}
        words = text.split()
        
        for indicator_type, lexicon in self.psychological_indicators.items():
            matches = [word for word in words if word in lexicon]
            score = len(matches) / len(words) if words else 0
            
            indicators[indicator_type] = {
                "score": score,
                "matches": matches,
                "intensity": self._determine_emotion_intensity(score)
            }
        
        return indicators
    
    def _analyze_language_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze language patterns"""
        patterns = {}
        
        for pattern_type, regex_list in self.language_patterns.items():
            matches = []
            for regex in regex_list:
                found_matches = re.findall(regex, text, re.IGNORECASE)
                matches.extend(found_matches)
            
            patterns[pattern_type] = {
                "count": len(matches),
                "matches": matches,
                "density": len(matches) / len(text.split()) if text.split() else 0
            }
        
        return patterns
    
    def _update_emotion_profile(self, user_id: str, result: SentimentResult):
        """Update emotion profile for user"""
        if user_id not in self.emotion_profiles:
            self.emotion_profiles[user_id] = EmotionProfile(
                id=user_id,
                name=f"User {user_id}",
                total_analyses=0,
                emotion_distribution={},
                sentiment_trend=[],
                dominant_emotions=[],
                emotional_stability=0.0,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
        
        profile = self.emotion_profiles[user_id]
        profile.total_analyses += 1
        profile.sentiment_trend.append(result.sentiment_score)
        profile.last_updated = datetime.now()
        
        # Update emotion distribution
        for emotion_score in result.emotions:
            if emotion_score.emotion in profile.emotion_distribution:
                profile.emotion_distribution[emotion_score.emotion] = (
                    profile.emotion_distribution[emotion_score.emotion] + emotion_score.score
                ) / 2
            else:
                profile.emotion_distribution[emotion_score.emotion] = emotion_score.score
        
        # Update dominant emotions
        dominant_emotions = [e.emotion for e in result.emotions[:3]]
        profile.dominant_emotions = dominant_emotions
        
        # Calculate emotional stability
        if len(profile.sentiment_trend) > 1:
            variance = statistics.variance(profile.sentiment_trend)
            profile.emotional_stability = max(0, 1 - variance)
    
    def get_emotion_profile(self, user_id: str) -> Optional[EmotionProfile]:
        """Get emotion profile for user"""
        return self.emotion_profiles.get(user_id)
    
    def get_sentiment_trend(self, user_id: str, days: int = 30) -> List[float]:
        """Get sentiment trend for user over specified days"""
        if user_id not in self.emotion_profiles:
            return []
        
        profile = self.emotion_profiles[user_id]
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter recent analyses
        recent_analyses = [
            result for result in self.analysis_history
            if result.metadata.get("user_id") == user_id and result.analyzed_at >= cutoff_date
        ]
        
        return [result.sentiment_score for result in recent_analyses]
    
    def get_emotion_insights(self, user_id: str) -> Dict[str, Any]:
        """Get emotion insights for user"""
        if user_id not in self.emotion_profiles:
            return {"error": "User profile not found"}
        
        profile = self.emotion_profiles[user_id]
        
        # Get recent analyses
        recent_analyses = [
            result for result in self.analysis_history
            if result.metadata.get("user_id") == user_id
        ]
        
        if not recent_analyses:
            return {"error": "No analyses found"}
        
        # Calculate insights
        avg_sentiment = statistics.mean([r.sentiment_score for r in recent_analyses])
        most_common_emotion = max(profile.emotion_distribution.items(), key=lambda x: x[1])[0] if profile.emotion_distribution else None
        
        # Emotional patterns
        positive_count = len([r for r in recent_analyses if r.sentiment_score > 0.1])
        negative_count = len([r for r in recent_analyses if r.sentiment_score < -0.1])
        neutral_count = len(recent_analyses) - positive_count - negative_count
        
        return {
            "user_id": user_id,
            "total_analyses": profile.total_analyses,
            "average_sentiment": avg_sentiment,
            "emotional_stability": profile.emotional_stability,
            "most_common_emotion": most_common_emotion.value if most_common_emotion else None,
            "sentiment_distribution": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count
            },
            "emotion_distribution": {
                emotion.value: score for emotion, score in profile.emotion_distribution.items()
            },
            "dominant_emotions": [e.value for e in profile.dominant_emotions],
            "trend_direction": "improving" if len(profile.sentiment_trend) > 1 and profile.sentiment_trend[-1] > profile.sentiment_trend[0] else "declining"
        }
    
    def get_analyzer_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        total_analyses = len(self.analysis_history)
        
        if total_analyses == 0:
            return {
                "total_analyses": 0,
                "total_users": 0,
                "average_sentiment": 0,
                "most_common_emotion": None,
                "emotion_distribution": {}
            }
        
        # Calculate statistics
        sentiment_scores = [r.sentiment_score for r in self.analysis_history]
        avg_sentiment = statistics.mean(sentiment_scores)
        
        # Count emotions
        emotion_counts = Counter()
        for result in self.analysis_history:
            if result.emotions:
                emotion_counts[result.emotions[0].emotion] += 1
        
        most_common_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else None
        
        return {
            "total_analyses": total_analyses,
            "total_users": len(self.emotion_profiles),
            "average_sentiment": avg_sentiment,
            "most_common_emotion": most_common_emotion.value if most_common_emotion else None,
            "emotion_distribution": {
                emotion.value: count for emotion, count in emotion_counts.items()
            },
            "sentiment_distribution": {
                "very_positive": len([s for s in sentiment_scores if s >= 0.6]),
                "positive": len([s for s in sentiment_scores if 0.2 <= s < 0.6]),
                "neutral": len([s for s in sentiment_scores if -0.2 < s < 0.2]),
                "negative": len([s for s in sentiment_scores if -0.6 < s <= -0.2]),
                "very_negative": len([s for s in sentiment_scores if s <= -0.6])
            }
        }

# Example usage
if __name__ == "__main__":
    # Initialize emotion analyzer
    analyzer = AdvancedEmotionAnalyzer()
    
    # Analyze sample text
    sample_text = "I'm so excited about this new project! It's going to be amazing and I can't wait to get started. This is exactly what I've been hoping for."
    
    result = asyncio.run(analyzer.analyze_emotion(sample_text, "user_001"))
    
    print("Emotion Analysis Results:")
    print(f"Overall Sentiment: {result.overall_sentiment.value}")
    print(f"Sentiment Score: {result.sentiment_score:.3f}")
    print(f"Dominant Emotion: {result.dominant_emotion.value}")
    print(f"Emotional Complexity: {result.emotional_complexity:.3f}")
    
    print("\nTop Emotions:")
    for emotion in result.emotions[:3]:
        print(f"  {emotion.emotion.value}: {emotion.score:.3f} (confidence: {emotion.confidence:.3f})")
    
    print("\nPsychological Indicators:")
    for indicator, data in result.psychological_indicators.items():
        if data["score"] > 0:
            print(f"  {indicator}: {data['score']:.3f}")
    
    # Get user insights
    insights = analyzer.get_emotion_insights("user_001")
    print(f"\nUser Insights:")
    print(f"Average Sentiment: {insights['average_sentiment']:.3f}")
    print(f"Emotional Stability: {insights['emotional_stability']:.3f}")
    
    # Get analyzer statistics
    stats = analyzer.get_analyzer_statistics()
    print(f"\nAnalyzer Statistics:")
    print(f"Total Analyses: {stats['total_analyses']}")
    print(f"Average Sentiment: {stats['average_sentiment']:.3f}")
    
    print("\nAdvanced Emotion Analyzer initialized successfully")
