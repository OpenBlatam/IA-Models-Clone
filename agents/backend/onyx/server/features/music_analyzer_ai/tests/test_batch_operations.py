"""
Tests de operaciones en lote mejoradas
"""

import pytest
from unittest.mock import Mock, patch
import concurrent.futures


class TestBatchAnalysis:
    """Tests de análisis en lote"""
    
    @pytest.fixture
    def analyzer(self):
        """Fixture para MusicAnalyzer"""
        with patch('music_analyzer_ai.core.music_analyzer.GenreDetector'), \
             patch('music_analyzer_ai.core.music_analyzer.HarmonicAnalyzer'), \
             patch('music_analyzer_ai.core.music_analyzer.EmotionAnalyzer'):
            from ..core.music_analyzer import MusicAnalyzer
            return MusicAnalyzer()
    
    def test_batch_analyze_tracks(self, analyzer):
        """Test de análisis en lote de tracks"""
        track_ids = ["1", "2", "3", "4", "5"]
        results = []
        
        analyzer.genre_detector.detect_genre = Mock(return_value={})
        analyzer.emotion_analyzer.analyze_emotions = Mock(return_value={})
        analyzer.harmonic_analyzer.analyze_harmonic_progression = Mock(return_value={})
        
        for track_id in track_ids:
            spotify_data = {
                "track_info": {"name": f"Track {track_id}"},
                "audio_features": {"key": 0, "mode": 1, "tempo": 120.0},
                "audio_analysis": {}
            }
            result = analyzer.analyze_track(spotify_data)
            results.append(result)
        
        assert len(results) == len(track_ids)
        assert all(r is not None for r in results)
    
    def test_batch_analyze_with_errors(self, analyzer):
        """Test de análisis en lote con errores"""
        track_ids = ["1", "2", "invalid", "4"]
        results = []
        errors = []
        
        analyzer.genre_detector.detect_genre = Mock(return_value={})
        analyzer.emotion_analyzer.analyze_emotions = Mock(return_value={})
        analyzer.harmonic_analyzer.analyze_harmonic_progression = Mock(return_value={})
        
        for track_id in track_ids:
            try:
                if track_id == "invalid":
                    raise ValueError("Invalid track")
                
                spotify_data = {
                    "track_info": {"name": f"Track {track_id}"},
                    "audio_features": {"key": 0, "mode": 1, "tempo": 120.0},
                    "audio_analysis": {}
                }
                result = analyzer.analyze_track(spotify_data)
                results.append(result)
            except Exception as e:
                errors.append({"track_id": track_id, "error": str(e)})
        
        assert len(results) == 3
        assert len(errors) == 1


class TestBatchComparison:
    """Tests de comparación en lote"""
    
    def test_batch_compare_multiple_pairs(self):
        """Test de comparación en lote de múltiples pares"""
        from ..services.comparison_service import ComparisonService
        service = ComparisonService()
        
        tracks = [
            {"key": 0, "mode": 1, "tempo": 120.0, "energy": 0.8},
            {"key": 0, "mode": 1, "tempo": 125.0, "energy": 0.7},
            {"key": 1, "mode": 0, "tempo": 120.0, "energy": 0.9}
        ]
        
        comparisons = []
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                result = service.compare_tracks(tracks[i], tracks[j])
                comparisons.append({
                    "track1": i,
                    "track2": j,
                    "comparison": result
                })
        
        assert len(comparisons) == 3  # 3 pares posibles
        assert all("comparison" in c for c in comparisons)


class TestBatchExport:
    """Tests de exportación en lote"""
    
    def test_batch_export_analyses(self):
        """Test de exportación en lote de análisis"""
        from ..services.export_service import ExportService
        service = ExportService()
        
        analyses = [
            {
                "track_basic_info": {"name": f"Track {i}"},
                "musical_analysis": {"key_signature": "C major"}
            }
            for i in range(5)
        ]
        
        exported = []
        for analysis in analyses:
            json_str = service.export_to_json(analysis, include_coaching=False)
            exported.append(json_str)
        
        assert len(exported) == len(analyses)
        assert all(isinstance(e, str) for e in exported)


class TestBatchValidation:
    """Tests de validación en lote"""
    
    def test_validate_batch_track_ids(self):
        """Test de validación en lote de track IDs"""
        def validate_track_ids(track_ids):
            valid = []
            invalid = []
            
            for track_id in track_ids:
                if track_id and isinstance(track_id, str) and len(track_id) > 0:
                    valid.append(track_id)
                else:
                    invalid.append(track_id)
            
            return {"valid": valid, "invalid": invalid}
        
        track_ids = ["123", "456", "", None, "789", 123]
        result = validate_track_ids(track_ids)
        
        assert len(result["valid"]) == 3
        assert len(result["invalid"]) == 3
    
    def test_validate_batch_audio_features(self):
        """Test de validación en lote de características de audio"""
        def validate_features_list(features_list):
            valid = []
            invalid = []
            
            for features in features_list:
                if isinstance(features, dict):
                    if "key" in features and "tempo" in features:
                        if 0 <= features.get("energy", 0) <= 1:
                            valid.append(features)
                        else:
                            invalid.append(features)
                    else:
                        invalid.append(features)
                else:
                    invalid.append(features)
            
            return {"valid": valid, "invalid": invalid}
        
        features_list = [
            {"key": 0, "tempo": 120.0, "energy": 0.8},
            {"key": 0, "tempo": 120.0, "energy": 1.5},  # Inválido
            {"key": 0},  # Incompleto
            "invalid"  # Tipo incorrecto
        ]
        
        result = validate_features_list(features_list)
        
        assert len(result["valid"]) == 1
        assert len(result["invalid"]) == 3


class TestBatchPerformance:
    """Tests de performance en lote"""
    
    def test_batch_processing_efficiency(self):
        """Test de eficiencia de procesamiento en lote"""
        import time
        
        def process_item(item):
            # Simular procesamiento
            time.sleep(0.001)
            return {"processed": item}
        
        items = list(range(100))
        
        start = time.time()
        results = [process_item(item) for item in items]
        elapsed = time.time() - start
        
        assert len(results) == 100
        # Debe completarse en tiempo razonable
        assert elapsed < 1.0
    
    def test_parallel_batch_processing(self):
        """Test de procesamiento en lote paralelo"""
        import time
        
        def process_item(item):
            time.sleep(0.01)
            return {"processed": item}
        
        items = list(range(20))
        
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(process_item, items))
        elapsed = time.time() - start
        
        assert len(results) == 20
        # Paralelo debe ser más rápido que secuencial
        assert elapsed < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

