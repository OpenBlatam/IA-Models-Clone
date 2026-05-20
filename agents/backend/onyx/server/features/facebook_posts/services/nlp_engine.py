from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import re
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
        from collections import Counter
from typing import Any, List, Dict, Optional
"""
🧠 Facebook Posts - NLP Engine
==============================

Sistema NLP optimizado para análisis de Facebook posts.
"""



@dataclass
class NLPResult:
    """Resultado de análisis NLP."""
    sentiment_score: float  # -1 to 1
    engagement_score: float  # 0 to 1
    readability_score: float  # 0 to 1
    emotion_scores: Dict[str, float]
    topics: List[str]
    keywords: List[str]
    recommendations: List[str]
    confidence: float
    processing_time_ms: float


class FacebookNLPEngine:
    """Motor NLP para Facebook posts."""
    
    def __init__(self) -> Any:
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        
        # NLP patterns
        self.emotion_patterns = {
            'joy': ['happy', 'excited', 'amazing', 'awesome', 'love', 'great'],
            'anger': ['angry', 'mad', 'frustrated', 'hate', 'terrible'],
            'fear': ['scared', 'worried', 'afraid', 'nervous', 'anxious'],
            'sadness': ['sad', 'disappointed', 'upset', 'depressed'],
            'surprise': ['wow', 'amazing', 'incredible', 'unbelievable'],
            'trust': ['reliable', 'honest', 'trustworthy', 'dependable']
        }
        
        self.engagement_indicators = {
            'questions': [r'\?', r'what do you think', r'tell us'],
            'cta': [r'click', r'visit', r'follow', r'share', r'comment'],
            'urgency': [r'now', r'today', r'limited time', r'hurry'],
            'social_proof': [r'everyone', r'thousands', r'millions']
        }
        
        self.logger.info("FacebookNLPEngine initialized")
    
    async def analyze_post(self, text: str, metadata: Optional[Dict] = None) -> NLPResult:
        """Análisis NLP completo del post."""
        start_time = datetime.now()
        
        try:
            # Parallel analysis
            sentiment_task = self._analyze_sentiment(text)
            engagement_task = self._analyze_engagement(text)
            readability_task = self._analyze_readability(text)
            emotion_task = self._analyze_emotions(text)
            topic_task = self._extract_topics(text)
            
            # Wait for all analyses
            sentiment_score = await sentiment_task
            engagement_score = await engagement_task
            readability_score = await readability_task
            emotion_scores = await emotion_task
            topics = await topic_task
            
            # Extract keywords
            keywords = self._extract_keywords(text)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                text, sentiment_score, engagement_score, emotion_scores
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return NLPResult(
                sentiment_score=sentiment_score,
                engagement_score=engagement_score,
                readability_score=readability_score,
                emotion_scores=emotion_scores,
                topics=topics,
                keywords=keywords,
                recommendations=recommendations,
                confidence=0.85,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"NLP analysis failed: {e}")
            return self._get_fallback_result()
    
    async def _analyze_sentiment(self, text: str) -> float:
        """Análisis de sentimiento."""
        positive_words = ['good', 'great', 'awesome', 'amazing', 'love', 'excellent']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if len(words) == 0:
            return 0.0
        
        return (positive_count - negative_count) / len(words)
    
    async def _analyze_engagement(self, text: str) -> float:
        """Análisis de potencial de engagement."""
        score = 0.0
        
        # Check for engagement indicators
        for category, patterns in self.engagement_indicators.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    if category == 'questions':
                        score += 0.3
                    elif category == 'cta':
                        score += 0.25
                    elif category == 'urgency':
                        score += 0.2
                    elif category == 'social_proof':
                        score += 0.15
        
        # Word count factor
        word_count = len(text.split())
        if 50 <= word_count <= 150:
            score += 0.2
        elif 20 <= word_count < 50 or 150 < word_count <= 200:
            score += 0.1
        
        # Emoji factor
        emoji_count = len(re.findall(r'[😀-🿿]', text))
        if 1 <= emoji_count <= 3:
            score += 0.15
        
        return min(score, 1.0)
    
    async def _analyze_readability(self, text: str) -> float:
        """Análisis de legibilidad."""
        if not text:
            return 0.0
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        
        if len(sentences) == 0:
            return 0.5
        
        # Average words per sentence
        avg_words_per_sentence = len(words) / len(sentences)
        
        # Average syllables per word (simplified)
        avg_syllables = sum(self._count_syllables(word) for word in words) / len(words)
        
        # Flesch Reading Ease approximation
        flesch_score = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables
        
        # Normalize to 0-1 scale
        return max(0, min(1, flesch_score / 100))
    
    async def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Análisis de emociones."""
        emotions = {}
        words = text.lower().split()
        
        for emotion, keywords in self.emotion_patterns.items():
            count = sum(1 for word in words if word in keywords)
            emotions[emotion] = min(count / max(len(words), 1) * 5, 1.0)
        
        # Normalize emotions
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
        else:
            emotions['neutral'] = 1.0
        
        return emotions
    
    async def _extract_topics(self, text: str) -> List[str]:
        """Extracción de temas."""
        # Simple topic extraction based on key phrases
        topic_keywords = {
            'business': ['business', 'company', 'startup', 'entrepreneur', 'corporate'],
            'technology': ['technology', 'tech', 'AI', 'software', 'digital'],
            'marketing': ['marketing', 'advertising', 'promotion', 'campaign', 'brand'],
            'lifestyle': ['lifestyle', 'life', 'personal', 'wellness', 'health'],
            'education': ['education', 'learning', 'study', 'course', 'training'],
            'entertainment': ['entertainment', 'fun', 'game', 'movie', 'music']
        }
        
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics[:3]  # Max 3 topics
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extracción de palabras clave."""
        # Remove common words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Get word frequency
        word_freq = Counter(keywords)
        
        return [word for word, count in word_freq.most_common(10)]
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas en una palabra (aproximación)."""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _generate_recommendations(
        self, 
        text: str, 
        sentiment: float, 
        engagement: float, 
        emotions: Dict[str, float]
    ) -> List[str]:
        """Generar recomendaciones de optimización."""
        recommendations = []
        
        # Sentiment recommendations
        if sentiment < 0:
            recommendations.append("Consider adding more positive language")
        elif sentiment > 0.5:
            recommendations.append("Great positive tone! Maintain this energy")
        
        # Engagement recommendations
        if engagement < 0.5:
            if '?' not in text:
                recommendations.append("Add a question to encourage interaction")
            if not any(word in text.lower() for word in ['share', 'comment', 'tell']):
                recommendations.append("Include a call-to-action")
        
        # Length recommendations
        word_count = len(text.split())
        if word_count < 20:
            recommendations.append("Consider expanding the content for better engagement")
        elif word_count > 200:
            recommendations.append("Consider shortening for better readability")
        
        # Emoji recommendations
        emoji_count = len(re.findall(r'[😀-🿿]', text))
        if emoji_count == 0:
            recommendations.append("Add relevant emojis to increase visual appeal")
        
        return recommendations[:5]  # Max 5 recommendations
    
    def _get_fallback_result(self) -> NLPResult:
        """Resultado de fallback en caso de error."""
        return NLPResult(
            sentiment_score=0.0,
            engagement_score=0.5,
            readability_score=0.7,
            emotion_scores={'neutral': 1.0},
            topics=['general'],
            keywords=['content'],
            recommendations=["Content created successfully"],
            confidence=0.5,
            processing_time_ms=1.0
        )
    
    async def optimize_text(self, text: str, target_engagement: float = 0.8) -> str:
        """Optimizar texto para mayor engagement."""
        optimized = text
        
        # Add emoji if none present
        if not re.search(r'[😀-🿿]', optimized):
            optimized = "✨ " + optimized
        
        # Add question if none present and engagement is low
        analysis = await self.analyze_post(optimized)
        if analysis.engagement_score < target_engagement and '?' not in optimized:
            optimized += " What do you think? 💭"
        
        return optimized
    
    async def generate_hashtags(self, text: str, max_count: int = 5) -> List[str]:
        """Generar hashtags inteligentes."""
        topics = await self._extract_topics(text)
        keywords = self._extract_keywords(text)
        
        # Combine topics and keywords
        hashtag_candidates = []
        
        # Add topics as hashtags
        for topic in topics:
            hashtag_candidates.append(topic.lower())
        
        # Add top keywords as hashtags
        for keyword in keywords[:3]:
            if len(keyword) > 3:
                hashtag_candidates.append(keyword.lower())
        
        # Add general engagement hashtags
        general_hashtags = ['trending', 'viral', 'socialmedia', 'content']
        hashtag_candidates.extend(general_hashtags[:2])
        
        # Remove duplicates and limit count
        unique_hashtags = list(set(hashtag_candidates))
        return unique_hashtags[:max_count]
    
    def get_analytics(self) -> Dict[str, Any]:
        """Obtener analytics del motor NLP."""
        return {
            'service': 'FacebookNLPEngine',
            'cache_size': len(self.cache),
            'patterns_loaded': len(self.emotion_patterns),
            'status': 'active'
        } 