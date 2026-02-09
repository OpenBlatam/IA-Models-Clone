"""
Tests para el analizador principal de música
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from ..core.music_analyzer import MusicAnalyzer


class TestMusicAnalyzer:
    """Tests para la clase MusicAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Fixture para crear una instancia de MusicAnalyzer"""
        with patch('music_analyzer_ai.core.music_analyzer.GenreDetector'), \
             patch('music_analyzer_ai.core.music_analyzer.HarmonicAnalyzer'), \
             patch('music_analyzer_ai.core.music_analyzer.EmotionAnalyzer'):
            return MusicAnalyzer()
    
    @pytest.fixture
    def mock_spotify_data(self):
        """Fixture con datos de ejemplo de Spotify"""
        return {
            "track_info": {
                "name": "Test Track",
                "artists": [{"name": "Test Artist"}],
                "album": {"name": "Test Album", "release_date": "2024-01-01"},
                "duration_ms": 200000,
                "popularity": 80,
                "external_urls": {"spotify": "https://spotify.com/track/123"},
                "preview_url": "https://preview.url"
            },
            "audio_features": {
                "key": 0,  # C
                "mode": 1,  # Major
                "tempo": 120.0,
                "time_signature": 4,
                "energy": 0.8,
                "danceability": 0.7,
                "valence": 0.6,
                "acousticness": 0.3,
                "instrumentalness": 0.2,
                "liveness": 0.1,
                "speechiness": 0.05,
                "loudness": -5.5
            },
            "audio_analysis": {
                "beats": [{"start": 0.0}, {"start": 0.5}, {"start": 1.0}],
                "bars": [{"start": 0.0}, {"start": 2.0}],
                "sections": [
                    {
                        "start": 0.0,
                        "duration": 10.0,
                        "key": 0,
                        "mode": 1,
                        "tempo": 120.0,
                        "key_confidence": 0.9,
                        "tempo_confidence": 0.95,
                        "loudness": -5.0
                    }
                ],
                "segments": [
                    {
                        "start": 0.0,
                        "pitches": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2],
                        "timbre": [1.0, 0.5, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
                        "loudness_max": -5.0
                    }
                ],
                "tatums": [{"start": 0.0}, {"start": 0.25}]
            }
        }
    
    def test_init(self, analyzer):
        """Test de inicialización"""
        assert analyzer is not None
        assert analyzer.genre_detector is not None
        assert analyzer.harmonic_analyzer is not None
        assert analyzer.emotion_analyzer is not None
    
    def test_analyze_track(self, analyzer, mock_spotify_data):
        """Test del análisis completo de una canción"""
        analyzer.genre_detector.detect_genre = Mock(return_value={"genre": "Rock"})
        analyzer.emotion_analyzer.analyze_emotions = Mock(return_value={"emotion": "Happy"})
        analyzer.harmonic_analyzer.analyze_harmonic_progression = Mock(return_value={"progression": "I-V-vi-IV"})
        
        result = analyzer.analyze_track(mock_spotify_data)
        
        assert "track_basic_info" in result
        assert "musical_analysis" in result
        assert "technical_analysis" in result
        assert "composition_analysis" in result
        assert "performance_analysis" in result
        assert "educational_insights" in result
        assert "genre_analysis" in result
        assert "emotion_analysis" in result
        assert "harmonic_analysis" in result
    
    def test_extract_basic_info(self, analyzer, mock_spotify_data):
        """Test de extracción de información básica"""
        track_info = mock_spotify_data["track_info"]
        result = analyzer._extract_basic_info(track_info)
        
        assert result["name"] == "Test Track"
        assert "Test Artist" in result["artists"]
        assert result["album"] == "Test Album"
        assert result["duration_ms"] == 200000
        assert result["duration_seconds"] == 200.0
        assert result["popularity"] == 80
    
    def test_analyze_musical_elements(self, analyzer, mock_spotify_data):
        """Test de análisis de elementos musicales"""
        audio_features = mock_spotify_data["audio_features"]
        audio_analysis = mock_spotify_data["audio_analysis"]
        
        result = analyzer._analyze_musical_elements(audio_features, audio_analysis)
        
        assert "key_signature" in result
        assert result["key_signature"] == "C major"
        assert result["root_note"] == "C"
        assert result["mode"] == "Major"
        assert result["tempo"]["bpm"] == 120.0
        assert result["time_signature"] == "4/4"
        assert "key_changes" in result
        assert "tempo_changes" in result
        assert "structure" in result
        assert "scale" in result
    
    def test_analyze_technical_aspects(self, analyzer, mock_spotify_data):
        """Test de análisis técnico"""
        audio_features = mock_spotify_data["audio_features"]
        audio_analysis = mock_spotify_data["audio_analysis"]
        
        result = analyzer._analyze_technical_aspects(audio_features, audio_analysis)
        
        assert "energy" in result
        assert result["energy"]["value"] == 0.8
        assert "danceability" in result
        assert result["danceability"]["value"] == 0.7
        assert "valence" in result
        assert "acousticness" in result
        assert "instrumentalness" in result
        assert "liveness" in result
        assert "speechiness" in result
        assert "loudness" in result
        assert "rhythm_structure" in result
    
    def test_categorize_tempo(self, analyzer):
        """Test de categorización de tempo"""
        assert "Lento" in analyzer._categorize_tempo(65)
        assert "Moderado" in analyzer._categorize_tempo(110)
        assert "Rápido" in analyzer._categorize_tempo(150)
        assert "Muy rápido" in analyzer._categorize_tempo(180)
        assert "Extremadamente rápido" in analyzer._categorize_tempo(220)
    
    def test_identify_scale(self, analyzer):
        """Test de identificación de escala"""
        # C Major
        result = analyzer._identify_scale(0, 1)
        assert result["name"] == "C major"
        assert "C" in result["notes"]
        assert "D" in result["notes"]
        assert "E" in result["notes"]
        
        # A Minor
        result = analyzer._identify_scale(9, 0)
        assert result["name"] == "A minor"
        assert "A" in result["notes"]
    
    def test_get_scale_notes(self, analyzer):
        """Test de obtención de notas de escala"""
        notes = analyzer._get_scale_notes(0, 1)  # C Major
        assert "C" in notes
        assert "D" in notes
        assert "E" in notes
        assert "F" in notes
        assert "G" in notes
        assert "A" in notes
        assert "B" in notes
    
    def test_get_common_chords(self, analyzer):
        """Test de obtención de acordes comunes"""
        # C Major
        chords = analyzer._get_common_chords(0, 1)
        assert len(chords) == 7
        assert "Cmaj" in chords
        
        # A Minor
        chords = analyzer._get_common_chords(9, 0)
        assert len(chords) == 7
        assert "Am" in chords
    
    def test_describe_energy(self, analyzer):
        """Test de descripción de energía"""
        assert "Muy baja" in analyzer._describe_energy(0.2)
        assert "Baja" in analyzer._describe_energy(0.4)
        assert "moderada" in analyzer._describe_energy(0.6)
        assert "Alta" in analyzer._describe_energy(0.8)
        assert "muy alta" in analyzer._describe_energy(0.95)
    
    def test_describe_danceability(self, analyzer):
        """Test de descripción de bailabilidad"""
        assert "No bailable" in analyzer._describe_danceability(0.2)
        assert "Poco bailable" in analyzer._describe_danceability(0.4)
        assert "Moderadamente" in analyzer._describe_danceability(0.6)
        assert "Muy bailable" in analyzer._describe_danceability(0.8)
        assert "Extremadamente" in analyzer._describe_danceability(0.95)
    
    def test_describe_valence(self, analyzer):
        """Test de descripción de valencia"""
        assert "triste" in analyzer._describe_valence(0.1)
        assert "Triste" in analyzer._describe_valence(0.3)
        assert "Neutral" in analyzer._describe_valence(0.5)
        assert "Feliz" in analyzer._describe_valence(0.7)
        assert "muy feliz" in analyzer._describe_valence(0.9)
    
    def test_identify_section_type(self, analyzer, mock_spotify_data):
        """Test de identificación de tipo de sección"""
        sections = mock_spotify_data["audio_analysis"]["sections"]
        
        # Intro (start < 10)
        section = {"start": 5.0, "duration": 10.0, "loudness": -10.0}
        assert analyzer._identify_section_type(section, sections) == "Intro"
        
        # Chorus (loudness > -8)
        section = {"start": 20.0, "duration": 10.0, "loudness": -5.0}
        assert analyzer._identify_section_type(section, sections) == "Chorus"
        
        # Bridge (duration < 20)
        section = {"start": 30.0, "duration": 15.0, "loudness": -10.0}
        assert analyzer._identify_section_type(section, sections) == "Bridge"
        
        # Verse (default)
        section = {"start": 50.0, "duration": 25.0, "loudness": -10.0}
        assert analyzer._identify_section_type(section, sections) == "Verse"
    
    def test_identify_composition_style(self, analyzer):
        """Test de identificación de estilo de composición"""
        # Acústico
        features = {"acousticness": 0.8, "energy": 0.5, "danceability": 0.5}
        assert analyzer._identify_composition_style(features) == "Acústico"
        
        # Dance/Electronic
        features = {"acousticness": 0.2, "energy": 0.8, "danceability": 0.8}
        assert analyzer._identify_composition_style(features) == "Dance/Electronic"
        
        # Rock/Metal
        features = {"acousticness": 0.2, "energy": 0.9, "danceability": 0.5}
        assert analyzer._identify_composition_style(features) == "Rock/Metal"
        
        # Pop/General (default)
        features = {"acousticness": 0.3, "energy": 0.6, "danceability": 0.6}
        assert analyzer._identify_composition_style(features) == "Pop/General"
    
    def test_assess_complexity(self, analyzer, mock_spotify_data):
        """Test de evaluación de complejidad"""
        audio_features = mock_spotify_data["audio_features"]
        audio_analysis = mock_spotify_data["audio_analysis"]
        
        result = analyzer._assess_complexity(audio_features, audio_analysis)
        
        assert "level" in result
        assert "score" in result
        assert "factors" in result
        assert result["level"] in ["Simple", "Moderada", "Compleja", "Muy compleja"]
        assert 0 <= result["score"] <= 1
    
    def test_calculate_dynamic_range(self, analyzer, mock_spotify_data):
        """Test de cálculo de rango dinámico"""
        segments = mock_spotify_data["audio_analysis"]["segments"]
        
        result = analyzer._calculate_dynamic_range(segments)
        
        assert "min" in result
        assert "max" in result
        assert "range" in result
        assert "description" in result
    
    def test_calculate_dynamic_range_empty(self, analyzer):
        """Test de cálculo de rango dinámico con datos vacíos"""
        result = analyzer._calculate_dynamic_range([])
        assert result["range"] == "Unknown"
    
    def test_suggest_musical_style(self, analyzer):
        """Test de sugerencia de estilo musical"""
        assert "lenta" in analyzer._suggest_musical_style(50, 1)
        assert "Balada" in analyzer._suggest_musical_style(80, 1)
        assert "Pop" in analyzer._suggest_musical_style(110, 1)
        assert "Dance" in analyzer._suggest_musical_style(150, 1)
        assert "EDM" in analyzer._suggest_musical_style(200, 1)
    
    def test_generate_learning_points(self, analyzer, mock_spotify_data):
        """Test de generación de puntos de aprendizaje"""
        audio_features = mock_spotify_data["audio_features"]
        audio_analysis = mock_spotify_data["audio_analysis"]
        
        points = analyzer._generate_learning_points(audio_features, audio_analysis)
        
        assert isinstance(points, list)
        assert len(points) > 0
        assert any("C" in point for point in points)
        assert any("BPM" in point for point in points)
    
    def test_generate_practice_suggestions(self, analyzer, mock_spotify_data):
        """Test de generación de sugerencias de práctica"""
        audio_features = mock_spotify_data["audio_features"]
        
        suggestions = analyzer._generate_practice_suggestions(audio_features)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert any("escala" in s.lower() or "scale" in s.lower() for s in suggestions)
    
    def test_analyze_composition(self, analyzer, mock_spotify_data):
        """Test de análisis de composición"""
        audio_features = mock_spotify_data["audio_features"]
        audio_analysis = mock_spotify_data["audio_analysis"]
        
        result = analyzer._analyze_composition(audio_features, audio_analysis)
        
        assert "structure" in result
        assert "harmonic_progressions" in result
        assert "composition_style" in result
        assert "complexity" in result
    
    def test_analyze_performance(self, analyzer, mock_spotify_data):
        """Test de análisis de interpretación"""
        audio_analysis = mock_spotify_data["audio_analysis"]
        
        result = analyzer._analyze_performance(audio_analysis)
        
        assert "timbre_analysis" in result
        assert "dynamic_range" in result
        assert "performance_characteristics" in result
    
    def test_generate_educational_insights(self, analyzer, mock_spotify_data):
        """Test de generación de insights educativos"""
        audio_features = mock_spotify_data["audio_features"]
        audio_analysis = mock_spotify_data["audio_analysis"]
        
        insights = analyzer._generate_educational_insights(audio_features, audio_analysis)
        
        assert "key_analysis" in insights
        assert "tempo_analysis" in insights
        assert "learning_points" in insights
        assert "practice_suggestions" in insights
    
    def test_analyze_track_with_missing_data(self, analyzer):
        """Test de análisis con datos faltantes"""
        incomplete_data = {
            "track_info": {"name": "Test"},
            "audio_features": {},
            "audio_analysis": {}
        }
        
        analyzer.genre_detector.detect_genre = Mock(return_value={})
        analyzer.emotion_analyzer.analyze_emotions = Mock(return_value={})
        analyzer.harmonic_analyzer.analyze_harmonic_progression = Mock(return_value={})
        
        result = analyzer.analyze_track(incomplete_data)
        
        # Debe manejar datos faltantes sin errores
        assert "track_basic_info" in result
        assert "musical_analysis" in result
    
    def test_analyze_track_with_partial_data(self, analyzer):
        """Test de análisis con datos parciales"""
        partial_data = {
            "track_info": {"name": "Test Track", "artists": []},
            "audio_features": {"key": 0, "mode": 1},  # Solo algunos campos
            "audio_analysis": {"sections": []}  # Solo algunas secciones
        }
        
        analyzer.genre_detector.detect_genre = Mock(return_value={})
        analyzer.emotion_analyzer.analyze_emotions = Mock(return_value={})
        analyzer.harmonic_analyzer.analyze_harmonic_progression = Mock(return_value={})
        
        result = analyzer.analyze_track(partial_data)
        
        assert result is not None
        assert "track_basic_info" in result
    
    def test_analyze_track_with_none_values(self, analyzer):
        """Test de análisis con valores None"""
        data_with_none = {
            "track_info": {"name": None, "artists": None},
            "audio_features": {"key": None, "mode": None, "tempo": None},
            "audio_analysis": None
        }
        
        analyzer.genre_detector.detect_genre = Mock(return_value={})
        analyzer.emotion_analyzer.analyze_emotions = Mock(return_value={})
        analyzer.harmonic_analyzer.analyze_harmonic_progression = Mock(return_value={})
        
        result = analyzer.analyze_track(data_with_none)
        
        # Debe manejar None sin crashear
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

