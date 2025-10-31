from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import re
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from collections import Counter
from typing import Any, List, Dict, Optional
import logging
"""
🎭 Emotion Analysis Module
==========================

Módulo especializado para análisis de emociones.
Detección multi-emocional y análisis de diversidad emocional.
"""



class EmotionAnalyzer:
    """Analizador de emociones especializado."""
    
    def __init__(self) -> Any:
        self.emotion_lexicon = {
            'joy': {
                'words': ['happy', 'excited', 'amazing', 'awesome', 'love', 'great', 'wonderful', 'fantastic', 'delighted', 'thrilled'],
                'patterns': [r':\)', r'😊', r'😍', r'🎉', r'❤️', r'💖']
            },
            'anger': {
                'words': ['angry', 'mad', 'frustrated', 'hate', 'terrible', 'furious', 'outraged', 'annoyed'],
                'patterns': [r'😠', r'😡', r'🤬', r'💢']
            },
            'fear': {
                'words': ['scared', 'worried', 'afraid', 'nervous', 'anxious', 'frightened', 'terrified', 'concerned'],
                'patterns': [r'😨', r'😰', r'😱']
            },
            'sadness': {
                'words': ['sad', 'disappointed', 'upset', 'depressed', 'heartbroken', 'miserable', 'devastated'],
                'patterns': [r'😢', r'😭', r'💔', r'☹️']
            },
            'surprise': {
                'words': ['wow', 'amazing', 'incredible', 'unbelievable', 'shocking', 'astonishing', 'stunned'],
                'patterns': [r'😲', r'😮', r'🤯', r'😱']
            },
            'trust': {
                'words': ['reliable', 'honest', 'trustworthy', 'dependable', 'loyal', 'faithful', 'confident'],
                'patterns': [r'🤝', r'👍', r'💪']
            },
            'disgust': {
                'words': ['disgusting', 'revolting', 'sick', 'gross', 'awful', 'horrible', 'repulsive'],
                'patterns': [r'🤢', r'🤮', r'😵']
            }
        }
        
        self.intensity_modifiers = {
            'very': 1.5, 'extremely': 2.0, 'incredibly': 1.8, 'absolutely': 1.7,
            'totally': 1.6, 'completely': 1.5, 'really': 1.3, 'quite': 1.2,
            'somewhat': 0.8, 'slightly': 0.6, 'barely': 0.4
        }
    
    async def analyze(self, text: str) -> Dict[str, any]:
        """Analizar emociones del texto."""
        start_time = datetime.now()
        
        # Detect emotions
        emotion_scores = await self._detect_emotions(text)
        
        # Calculate emotional metrics
        dominant_emotion = await self._get_dominant_emotion(emotion_scores)
        emotional_diversity = await self._calculate_emotional_diversity(emotion_scores)
        emotional_stability = await self._calculate_emotional_stability(text)
        emotional_intensity = await self._calculate_emotional_intensity(text)
        
        # Analyze emotional progression
        emotional_arc = await self._analyze_emotional_arc(text)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            'emotions': emotion_scores,
            'dominant_emotion': dominant_emotion,
            'emotional_diversity': emotional_diversity,
            'emotional_stability': emotional_stability,
            'emotional_intensity': emotional_intensity,
            'emotional_arc': emotional_arc,
            'processing_time_ms': processing_time
        }
    
    async def _detect_emotions(self, text: str) -> Dict[str, float]:
        """Detectar emociones en el texto."""
        words = re.findall(r'\b\w+\b', text.lower())
        text_lower = text.lower()
        
        emotion_scores = {}
        
        for emotion, data in self.emotion_lexicon.items():
            score = 0.0
            
            # Word-based detection
            word_matches = sum(1 for word in words if word in data['words'])
            if word_matches > 0:
                score += word_matches / len(words) * 5  # Scale factor
            
            # Pattern-based detection (emojis, emoticons)
            pattern_matches = sum(len(re.findall(pattern, text)) for pattern in data['patterns'])
            if pattern_matches > 0:
                score += pattern_matches * 0.3  # Emoji weight
            
            # Apply intensity modifiers
            score = self._apply_intensity_modifiers(text_lower, score, words)
            
            emotion_scores[emotion] = min(score, 1.0)  # Cap at 1.0
        
        # Normalize scores
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
        else:
            # No emotions detected, assign neutral
            emotion_scores = {emotion: 0.0 for emotion in self.emotion_lexicon.keys()}
            emotion_scores['neutral'] = 1.0
        
        return emotion_scores
    
    def _apply_intensity_modifiers(self, text_lower: str, base_score: float, words: List[str]) -> float:
        """Aplicar modificadores de intensidad."""
        intensity_multiplier = 1.0
        
        for word in words:
            if word in self.intensity_modifiers:
                intensity_multiplier = max(intensity_multiplier, self.intensity_modifiers[word])
        
        # Check for caps (indicates intensity)
        caps_words = sum(1 for word in text_lower.split() if word.isupper() and len(word) > 1)
        if caps_words > 0:
            intensity_multiplier *= (1 + caps_words * 0.2)
        
        # Check for exclamations
        exclamation_count = text_lower.count('!')
        if exclamation_count > 0:
            intensity_multiplier *= (1 + exclamation_count * 0.15)
        
        return base_score * intensity_multiplier
    
    async def _get_dominant_emotion(self, emotion_scores: Dict[str, float]) -> str:
        """Obtener emoción dominante."""
        if not emotion_scores:
            return 'neutral'
        
        return max(emotion_scores.items(), key=lambda x: x[1])[0]
    
    async def _calculate_emotional_diversity(self, emotion_scores: Dict[str, float]) -> float:
        """Calcular diversidad emocional."""
        # Shannon entropy for emotional diversity
        total = sum(emotion_scores.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for score in emotion_scores.values():
            if score > 0:
                p = score / total
                entropy -= p * (p.bit_length() - 1) if p > 0 else 0
        
        # Normalize entropy to 0-1 scale
        max_entropy = (len(emotion_scores).bit_length() - 1) if len(emotion_scores) > 1 else 1
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    async def _calculate_emotional_stability(self, text: str) -> float:
        """Calcular estabilidad emocional a lo largo del texto."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 1.0  # Single sentence is considered stable
        
        sentence_emotions = []
        for sentence in sentences:
            emotion_scores = await self._detect_emotions(sentence)
            dominant = await self._get_dominant_emotion(emotion_scores)
            sentence_emotions.append(dominant)
        
        # Calculate consistency
        emotion_counts = Counter(sentence_emotions)
        most_common_count = emotion_counts.most_common(1)[0][1]
        stability = most_common_count / len(sentence_emotions)
        
        return stability
    
    async def _calculate_emotional_intensity(self, text: str) -> float:
        """Calcular intensidad emocional general."""
        intensity_factors = 0.0
        words = text.split()
        
        # Caps factor
        caps_count = sum(1 for word in words if word.isupper() and len(word) > 1)
        intensity_factors += caps_count * 0.2
        
        # Punctuation factor
        exclamation_count = text.count('!')
        question_count = text.count('?')
        intensity_factors += (exclamation_count + question_count) * 0.15
        
        # Intensifier words
        text_lower = text.lower()
        intensifier_count = sum(1 for word in text_lower.split() if word in self.intensity_modifiers)
        intensity_factors += intensifier_count * 0.3
        
        # Emotional emoji factor
        emotional_emojis = len(re.findall(r'[😀-🿿]', text))
        intensity_factors += emotional_emojis * 0.1
        
        # Normalize by text length
        return min(intensity_factors / max(len(words), 1) * 5, 1.0)
    
    async def _analyze_emotional_arc(self, text: str) -> Dict[str, any]:
        """Analizar arco emocional del texto."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return {
                'progression': 'stable',
                'trend': 'neutral',
                'emotional_journey': []
            }
        
        emotional_journey = []
        for sentence in sentences:
            emotion_scores = await self._detect_emotions(sentence)
            dominant = await self._get_dominant_emotion(emotion_scores)
            emotional_journey.append(dominant)
        
        # Analyze progression
        progression = 'stable'
        if len(set(emotional_journey)) > len(emotional_journey) / 2:
            progression = 'varied'
        elif emotional_journey[0] != emotional_journey[-1]:
            progression = 'changing'
        
        # Analyze trend
        trend = 'neutral'
        positive_emotions = ['joy', 'trust', 'surprise']
        negative_emotions = ['anger', 'fear', 'sadness', 'disgust']
        
        start_sentiment = 1 if emotional_journey[0] in positive_emotions else (-1 if emotional_journey[0] in negative_emotions else 0)
        end_sentiment = 1 if emotional_journey[-1] in positive_emotions else (-1 if emotional_journey[-1] in negative_emotions else 0)
        
        if end_sentiment > start_sentiment:
            trend = 'improving'
        elif end_sentiment < start_sentiment:
            trend = 'declining'
        
        return {
            'progression': progression,
            'trend': trend,
            'emotional_journey': emotional_journey
        } 