"""
Tests para Analizadores
=======================
"""

import pytest
from agents.backend.onyx.server.features.validacion_psicologica_ai.analyzers import (
    SentimentAnalyzer,
    PersonalityAnalyzer,
    BehavioralPatternAnalyzer,
    AdvancedPsychologicalAnalyzer
)
from agents.backend.onyx.server.features.validacion_psicologica_ai.models import SocialMediaPlatform


class TestSentimentAnalyzer:
    """Tests para analizador de sentimientos"""
    
    def test_positive_sentiment(self):
        """Test análisis de sentimiento positivo"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("I'm so happy and excited about this amazing opportunity!")
        
        assert result["sentiment"] == "positive"
        assert result["score"] > 0
    
    def test_negative_sentiment(self):
        """Test análisis de sentimiento negativo"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("I'm feeling sad and disappointed about this terrible situation.")
        
        assert result["sentiment"] == "negative"
        assert result["score"] < 0
    
    def test_neutral_sentiment(self):
        """Test análisis de sentimiento neutral"""
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("The weather is nice today.")
        
        assert result["sentiment"] in ["neutral", "positive"]
    
    def test_batch_analysis(self):
        """Test análisis por lotes"""
        analyzer = SentimentAnalyzer()
        texts = [
            "I'm happy!",
            "This is terrible.",
            "The weather is nice."
        ]
        result = analyzer.analyze_batch(texts)
        
        assert "overall_sentiment" in result
        assert "sentiment_distribution" in result
        assert "average_score" in result


class TestPersonalityAnalyzer:
    """Tests para analizador de personalidad"""
    
    def test_personality_analysis(self):
        """Test análisis de personalidad"""
        analyzer = PersonalityAnalyzer()
        texts = [
            "I love creative projects and artistic endeavors",
            "I'm very organized and disciplined in my work",
            "I enjoy social gatherings and meeting new people"
        ]
        result = analyzer.analyze(texts)
        
        assert "openness" in result
        assert "conscientiousness" in result
        assert "extraversion" in result
        assert "agreeableness" in result
        assert "neuroticism" in result
        
        # Verificar que los scores están en rango válido
        for trait, score in result.items():
            assert 0.0 <= score <= 1.0
    
    def test_empty_texts(self):
        """Test con textos vacíos"""
        analyzer = PersonalityAnalyzer()
        result = analyzer.analyze([])
        
        assert all(0.4 <= score <= 0.6 for score in result.values())


class TestBehavioralPatternAnalyzer:
    """Tests para analizador de patrones de comportamiento"""
    
    def test_behavioral_patterns(self):
        """Test análisis de patrones"""
        analyzer = BehavioralPatternAnalyzer()
        posts = [
            {"created_at": "2024-01-01T10:00:00", "likes": 10},
            {"created_at": "2024-01-02T10:00:00", "likes": 15},
            {"created_at": "2024-01-03T10:00:00", "likes": 12}
        ]
        interactions = {
            "total_likes": 100,
            "total_comments": 50,
            "total_shares": 25
        }
        
        result = analyzer.analyze(posts, interactions)
        
        assert isinstance(result, list)
        assert len(result) > 0


class TestAdvancedPsychologicalAnalyzer:
    """Tests para analizador avanzado"""
    
    @pytest.mark.asyncio
    async def test_advanced_analysis(self):
        """Test análisis avanzado completo"""
        analyzer = AdvancedPsychologicalAnalyzer()
        social_media_data = {
            SocialMediaPlatform.TWITTER: {
                "posts": [
                    {
                        "text": "I'm so happy today!",
                        "created_at": "2024-01-01T10:00:00",
                        "likes": 10,
                        "comments": 2
                    }
                ],
                "interactions": {
                    "total_likes": 10,
                    "total_comments": 2,
                    "total_shares": 1
                }
            }
        }
        
        result = await analyzer.analyze_social_media_data(social_media_data)
        
        assert "sentiment_analysis" in result
        assert "personality_traits" in result
        assert "behavioral_patterns" in result
        assert "confidence_score" in result
        assert 0.0 <= result["confidence_score"] <= 1.0




