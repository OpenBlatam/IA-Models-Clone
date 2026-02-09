"""
Tests para Utilidades
=====================
"""

import pytest
from agents.backend.onyx.server.features.validacion_psicologica_ai.utils import (
    TextProcessor,
    CacheManager,
    MetricsCollector,
    ValidationComparator
)


class TestTextProcessor:
    """Tests para procesador de texto"""
    
    def test_clean_text(self):
        """Test limpieza de texto"""
        text = "Check out this link: https://example.com #hashtag @mention"
        cleaned = TextProcessor.clean_text(text)
        
        assert "https://" not in cleaned
        assert len(cleaned) > 0
    
    def test_extract_keywords(self):
        """Test extracción de palabras clave"""
        text = "I love programming and coding. Programming is fun and coding is great."
        keywords = TextProcessor.extract_keywords(text, top_n=5)
        
        assert len(keywords) <= 5
        assert "programming" in keywords or "coding" in keywords
    
    def test_readability_score(self):
        """Test cálculo de legibilidad"""
        text = "This is a simple sentence. It is easy to read."
        score = TextProcessor.calculate_readability_score(text)
        
        assert 0.0 <= score <= 1.0


class TestCacheManager:
    """Tests para gestor de caché"""
    
    def test_cache_set_get(self):
        """Test guardar y obtener del caché"""
        cache = CacheManager(ttl=3600)
        cache.set("test_key", "test_value")
        
        value = cache.get("test_key")
        assert value == "test_value"
    
    def test_cache_expiration(self):
        """Test expiración del caché"""
        cache = CacheManager(ttl=1)  # 1 segundo
        cache.set("test_key", "test_value")
        
        import time
        time.sleep(2)
        
        value = cache.get("test_key")
        assert value is None
    
    def test_cache_stats(self):
        """Test estadísticas del caché"""
        cache = CacheManager()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        stats = cache.get_stats()
        assert stats["size"] == 2


class TestMetricsCollector:
    """Tests para colector de métricas"""
    
    def test_metrics_increment(self):
        """Test incremento de métricas"""
        metrics = MetricsCollector()
        metrics.increment("test_metric")
        
        assert metrics.get("test_metric") == 1
    
    def test_metrics_get_all(self):
        """Test obtener todas las métricas"""
        metrics = MetricsCollector()
        metrics.increment("validations_created")
        metrics.increment("validations_completed")
        
        all_metrics = metrics.get_all()
        assert "validations_created" in all_metrics
        assert "validations_completed" in all_metrics
        assert "uptime_seconds" in all_metrics


class TestValidationComparator:
    """Tests para comparador de validaciones"""
    
    def test_compare_validations(self):
        """Test comparación de validaciones"""
        validation1 = {
            "profile": {
                "personality_traits": {
                    "openness": 0.7,
                    "extraversion": 0.6
                },
                "emotional_state": {
                    "overall_sentiment": "positive"
                },
                "confidence_score": 0.75
            },
            "created_at": "2024-01-01T00:00:00"
        }
        
        validation2 = {
            "profile": {
                "personality_traits": {
                    "openness": 0.8,
                    "extraversion": 0.7
                },
                "emotional_state": {
                    "overall_sentiment": "positive"
                },
                "confidence_score": 0.80
            },
            "created_at": "2024-01-15T00:00:00"
        }
        
        result = ValidationComparator.compare_validations(validation1, validation2)
        
        assert "trait_changes" in result
        assert "emotional_changes" in result
        assert "confidence_change" in result




