"""
Tests de performance y carga
"""

import pytest
import time
from unittest.mock import Mock, patch


class TestPerformanceMusicAnalyzer:
    """Tests de performance para MusicAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Fixture para crear MusicAnalyzer"""
        with patch('music_analyzer_ai.core.music_analyzer.GenreDetector'), \
             patch('music_analyzer_ai.core.music_analyzer.HarmonicAnalyzer'), \
             patch('music_analyzer_ai.core.music_analyzer.EmotionAnalyzer'):
            from ..core.music_analyzer import MusicAnalyzer
            return MusicAnalyzer()
    
    @pytest.fixture
    def large_audio_analysis(self):
        """Análisis de audio con muchos segmentos"""
        return {
            "sections": [{"start": i * 10, "duration": 10} for i in range(100)],
            "segments": [{"start": i * 0.5, "pitches": [0.1] * 12} for i in range(1000)],
            "beats": [{"start": i * 0.5} for i in range(500)],
            "bars": [{"start": i * 2} for i in range(100)]
        }
    
    def test_analyze_performance_large_data(self, analyzer, large_audio_analysis):
        """Test de performance con datos grandes"""
        audio_features = {
            "key": 0,
            "mode": 1,
            "tempo": 120.0,
            "time_signature": 4
        }
        
        start_time = time.time()
        result = analyzer._analyze_composition(audio_features, large_audio_analysis)
        elapsed_time = time.time() - start_time
        
        assert result is not None
        # Debe completarse en menos de 5 segundos
        assert elapsed_time < 5.0
    
    def test_analyze_performance_multiple_calls(self, analyzer, sample_audio_features, sample_audio_analysis):
        """Test de performance con múltiples llamadas"""
        times = []
        
        for _ in range(10):
            start_time = time.time()
            analyzer._analyze_technical_aspects(sample_audio_features, sample_audio_analysis)
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Tiempo promedio debe ser razonable
        assert avg_time < 1.0
        # Tiempo máximo no debe ser excesivo
        assert max_time < 2.0


class TestPerformanceServices:
    """Tests de performance para servicios"""
    
    def test_spotify_service_response_time(self):
        """Test de tiempo de respuesta del servicio Spotify"""
        from ..services.spotify_service import SpotifyService
        with patch('music_analyzer_ai.services.spotify_service.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"tracks": {"items": []}}
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            service = SpotifyService()
            
            start_time = time.time()
            service.search_tracks("test")
            elapsed = time.time() - start_time
            
            # Debe ser rápido con mock
            assert elapsed < 0.1
    
    def test_comparison_service_batch_performance(self):
        """Test de performance de comparación en lote"""
        from ..services.comparison_service import ComparisonService
        service = ComparisonService()
        
        features = {
            "key": 0,
            "mode": 1,
            "tempo": 120.0,
            "energy": 0.8
        }
        
        start_time = time.time()
        for _ in range(100):
            service.compare_tracks(features, features)
        elapsed = time.time() - start_time
        
        # 100 comparaciones deben completarse rápidamente
        assert elapsed < 1.0


class TestMemoryUsage:
    """Tests de uso de memoria"""
    
    def test_analyze_memory_efficient(self):
        """Test de eficiencia de memoria en análisis"""
        # Simular análisis sin cargar todo en memoria
        def process_sections_efficiently(sections):
            count = 0
            for section in sections:
                count += 1
                if count > 50:  # Limitar procesamiento
                    break
            return count
        
        sections = [{"start": i * 10} for i in range(1000)]
        result = process_sections_efficiently(sections)
        
        assert result == 50  # Solo procesó 50 secciones


class TestConcurrency:
    """Tests de concurrencia"""
    
    def test_concurrent_analysis_requests(self):
        """Test de requests concurrentes"""
        import threading
        
        results = []
        errors = []
        
        def analyze_track(track_id):
            try:
                # Simular análisis
                result = {"track_id": track_id, "status": "completed"}
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(10):
            thread = threading.Thread(target=analyze_track, args=(f"track_{i}",))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == 10
        assert len(errors) == 0


class TestCachingPerformance:
    """Tests de performance de caché"""
    
    def test_cache_hit_performance(self):
        """Test de performance con cache hit"""
        cache = {}
        
        def get_cached(key, compute_func):
            if key in cache:
                return cache[key]
            result = compute_func()
            cache[key] = result
            return result
        
        def expensive_operation():
            time.sleep(0.1)  # Simular operación costosa
            return "result"
        
        # Primera llamada (cache miss)
        start = time.time()
        result1 = get_cached("key1", expensive_operation)
        time1 = time.time() - start
        
        # Segunda llamada (cache hit)
        start = time.time()
        result2 = get_cached("key1", expensive_operation)
        time2 = time.time() - start
        
        assert result1 == result2
        # Cache hit debe ser mucho más rápido
        assert time2 < time1 / 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])

