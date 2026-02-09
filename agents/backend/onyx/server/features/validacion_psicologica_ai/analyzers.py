"""
Analizadores avanzados para Validación Psicológica AI
======================================================
Análisis psicológico usando NLP y técnicas avanzadas
"""

from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime, timedelta
import structlog
import asyncio
import re
from collections import Counter

from .models import (
    PsychologicalProfile,
    SocialMediaPlatform,
    ValidationReport,
)
from .config import config

logger = structlog.get_logger()


class SentimentAnalyzer:
    """Analizador de sentimientos"""
    
    # Palabras clave para análisis de sentimientos
    POSITIVE_WORDS = {
        "happy", "joy", "excited", "love", "amazing", "wonderful", "great",
        "fantastic", "excellent", "beautiful", "perfect", "awesome", "good",
        "best", "favorite", "enjoy", "smile", "laugh", "celebrate", "success"
    }
    
    NEGATIVE_WORDS = {
        "sad", "angry", "hate", "terrible", "awful", "bad", "worst",
        "disappointed", "frustrated", "depressed", "anxious", "worried",
        "stress", "pain", "suffer", "cry", "fail", "problem", "difficult"
    }
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analizar sentimiento de un texto
        
        Args:
            text: Texto a analizar
            
        Returns:
            Análisis de sentimiento
        """
        if not text:
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0
            }
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in words if word in self.POSITIVE_WORDS)
        negative_count = sum(1 for word in words if word in self.NEGATIVE_WORDS)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            sentiment = "neutral"
            score = 0.0
        elif positive_count > negative_count:
            sentiment = "positive"
            score = positive_count / max(total_sentiment_words, 1)
        elif negative_count > positive_count:
            sentiment = "negative"
            score = -negative_count / max(total_sentiment_words, 1)
        else:
            sentiment = "neutral"
            score = 0.0
        
        confidence = min(total_sentiment_words / max(len(words), 1), 1.0)
        
        return {
            "sentiment": sentiment,
            "score": score,
            "confidence": confidence,
            "positive_words": positive_count,
            "negative_words": negative_count
        }
    
    def analyze_batch(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analizar múltiples textos
        
        Args:
            texts: Lista de textos
            
        Returns:
            Análisis agregado
        """
        if not texts:
            return {
                "overall_sentiment": "neutral",
                "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
                "average_score": 0.0
            }
        
        results = [self.analyze(text) for text in texts]
        
        sentiment_counts = Counter(r["sentiment"] for r in results)
        total = len(results)
        
        sentiment_distribution = {
            "positive": sentiment_counts.get("positive", 0) / total,
            "neutral": sentiment_counts.get("neutral", 0) / total,
            "negative": sentiment_counts.get("negative", 0) / total
        }
        
        overall_sentiment = max(sentiment_distribution, key=sentiment_distribution.get)
        average_score = sum(r["score"] for r in results) / total
        
        return {
            "overall_sentiment": overall_sentiment,
            "sentiment_distribution": sentiment_distribution,
            "average_score": average_score,
            "total_analyzed": total
        }


class PersonalityAnalyzer:
    """Analizador de personalidad (Big Five)"""
    
    # Indicadores de rasgos de personalidad
    TRAIT_INDICATORS = {
        "openness": {
            "positive": ["creative", "curious", "imaginative", "artistic", "adventurous"],
            "negative": ["conventional", "practical", "traditional"]
        },
        "conscientiousness": {
            "positive": ["organized", "disciplined", "reliable", "hardworking", "punctual"],
            "negative": ["careless", "disorganized", "lazy"]
        },
        "extraversion": {
            "positive": ["outgoing", "social", "energetic", "talkative", "friendly"],
            "negative": ["shy", "quiet", "reserved", "introverted"]
        },
        "agreeableness": {
            "positive": ["kind", "cooperative", "trusting", "helpful", "empathetic"],
            "negative": ["competitive", "skeptical", "critical"]
        },
        "neuroticism": {
            "positive": ["calm", "stable", "relaxed", "confident"],
            "negative": ["anxious", "worried", "stressed", "emotional", "nervous"]
        }
    }
    
    def analyze(self, texts: List[str]) -> Dict[str, float]:
        """
        Analizar rasgos de personalidad
        
        Args:
            texts: Lista de textos del usuario
            
        Returns:
            Scores de rasgos de personalidad (0-1)
        """
        if not texts:
            return {
                "openness": 0.5,
                "conscientiousness": 0.5,
                "extraversion": 0.5,
                "agreeableness": 0.5,
                "neuroticism": 0.5
            }
        
        all_text = " ".join(texts).lower()
        words = re.findall(r'\b\w+\b', all_text)
        word_set = set(words)
        
        trait_scores = {}
        
        for trait, indicators in self.TRAIT_INDICATORS.items():
            positive_matches = sum(1 for word in indicators["positive"] if word in word_set)
            negative_matches = sum(1 for word in indicators["negative"] if word in word_set)
            
            total_matches = positive_matches + negative_matches
            
            if total_matches == 0:
                score = 0.5  # Neutral
            else:
                # Calcular score basado en proporción de indicadores positivos
                score = 0.5 + (positive_matches - negative_matches) / (total_matches * 2)
                score = max(0.0, min(1.0, score))  # Clamp entre 0 y 1
            
            trait_scores[trait] = score
        
        return trait_scores


class BehavioralPatternAnalyzer:
    """Analizador de patrones de comportamiento"""
    
    def analyze(
        self,
        posts: List[Dict[str, Any]],
        interactions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Analizar patrones de comportamiento
        
        Args:
            posts: Lista de posts
            interactions: Datos de interacciones
            
        Returns:
            Lista de patrones identificados
        """
        patterns = []
        
        if not posts:
            return patterns
        
        # Análisis de frecuencia de posting
        if len(posts) > 0:
            post_dates = [
                datetime.fromisoformat(p.get("created_at", datetime.utcnow().isoformat()))
                for p in posts
            ]
            post_dates.sort()
            
            if len(post_dates) > 1:
                time_diffs = [
                    (post_dates[i+1] - post_dates[i]).total_seconds() / 3600
                    for i in range(len(post_dates) - 1)
                ]
                avg_hours = sum(time_diffs) / len(time_diffs)
                
                if avg_hours < 24:
                    frequency = "daily"
                elif avg_hours < 168:  # 1 week
                    frequency = "weekly"
                else:
                    frequency = "occasional"
                
                patterns.append({
                    "pattern": "Posting frequency",
                    "frequency": frequency,
                    "average_hours_between_posts": avg_hours,
                    "confidence": 0.8
                })
        
        # Análisis de engagement
        if interactions:
            total_likes = interactions.get("total_likes", 0)
            total_comments = interactions.get("total_comments", 0)
            total_shares = interactions.get("total_shares", 0)
            
            total_engagement = total_likes + total_comments + total_shares
            
            if total_engagement > 0:
                engagement_rate = total_engagement / len(posts) if posts else 0
                
                if engagement_rate > 50:
                    level = "high"
                elif engagement_rate > 20:
                    level = "medium"
                else:
                    level = "low"
                
                patterns.append({
                    "pattern": "Engagement level",
                    "level": level,
                    "engagement_rate": engagement_rate,
                    "confidence": 0.7
                })
        
        return patterns


class AdvancedPsychologicalAnalyzer:
    """Analizador psicológico avanzado que combina múltiples técnicas"""
    
    def __init__(self):
        """Inicializar analizador"""
        self.sentiment_analyzer = SentimentAnalyzer()
        self.personality_analyzer = PersonalityAnalyzer()
        self.behavioral_analyzer = BehavioralPatternAnalyzer()
        logger.info("AdvancedPsychologicalAnalyzer initialized")
    
    async def analyze_social_media_data(
        self,
        social_media_data: Dict[SocialMediaPlatform, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analizar datos de redes sociales de forma avanzada
        
        Args:
            social_media_data: Datos de todas las plataformas
            
        Returns:
            Análisis completo
        """
        logger.info("Starting advanced analysis", platforms=len(social_media_data))
        
        all_texts = []
        all_posts = []
        all_interactions = {}
        
        # Agregar datos de todas las plataformas
        for platform, data in social_media_data.items():
            posts = data.get("posts", [])
            for post in posts:
                text = post.get("text", "") or post.get("content", "")
                if text:
                    all_texts.append(text)
                    all_posts.append(post)
            
            interactions = data.get("interactions", {})
            for key, value in interactions.items():
                all_interactions[key] = all_interactions.get(key, 0) + value
        
        # Análisis de sentimientos
        sentiment_analysis = self.sentiment_analyzer.analyze_batch(all_texts)
        
        # Análisis de personalidad
        personality_traits = self.personality_analyzer.analyze(all_texts)
        
        # Análisis de patrones de comportamiento
        behavioral_patterns = self.behavioral_analyzer.analyze(all_posts, all_interactions)
        
        # Calcular score de confianza
        confidence_score = self._calculate_confidence_score(
            len(all_texts),
            len(social_media_data),
            sentiment_analysis
        )
        
        return {
            "sentiment_analysis": sentiment_analysis,
            "personality_traits": personality_traits,
            "behavioral_patterns": behavioral_patterns,
            "confidence_score": confidence_score,
            "data_points": {
                "total_texts": len(all_texts),
                "total_posts": len(all_posts),
                "total_platforms": len(social_media_data)
            }
        }
    
    def _calculate_confidence_score(
        self,
        text_count: int,
        platform_count: int,
        sentiment_analysis: Dict[str, Any]
    ) -> float:
        """
        Calcular score de confianza del análisis
        
        Args:
            text_count: Cantidad de textos analizados
            platform_count: Cantidad de plataformas
            sentiment_analysis: Análisis de sentimientos
            
        Returns:
            Score de confianza (0-1)
        """
        # Factor de cantidad de datos
        data_factor = min(text_count / 100, 1.0)  # Máximo en 100 textos
        
        # Factor de múltiples plataformas
        platform_factor = min(platform_count / 5, 1.0)  # Máximo en 5 plataformas
        
        # Factor de calidad del análisis de sentimientos
        sentiment_confidence = sentiment_analysis.get("total_analyzed", 0) / max(text_count, 1)
        
        # Calcular score combinado
        confidence = (
            data_factor * 0.4 +
            platform_factor * 0.3 +
            sentiment_confidence * 0.3
        )
        
        return max(0.0, min(1.0, confidence))




