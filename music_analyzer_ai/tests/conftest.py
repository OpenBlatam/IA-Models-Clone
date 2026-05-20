"""
Configuración compartida para tests
"""

import pytest
import sys
from pathlib import Path

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

# Importar helpers (importación diferida para evitar errores circulares)
try:
    from test_helpers import TestHelpers, MockDataFactory
except ImportError:
    # Si falla, los helpers se importarán cuando se necesiten
    TestHelpers = None
    MockDataFactory = None


@pytest.fixture(scope="session")
def test_data_dir():
    """Directorio con datos de test"""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def mock_spotify_token():
    """Token mock de Spotify"""
    return "mock_spotify_token_12345"


@pytest.fixture
def sample_audio_features():
    """Características de audio de ejemplo"""
    return {
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
    }


@pytest.fixture
def sample_track_info():
    """Información de track de ejemplo"""
    return {
        "id": "test_track_123",
        "name": "Test Track",
        "artists": [{"name": "Test Artist"}],
        "album": {
            "name": "Test Album",
            "release_date": "2024-01-01"
        },
        "duration_ms": 200000,
        "popularity": 80,
        "external_urls": {"spotify": "https://spotify.com/track/123"},
        "preview_url": "https://preview.url"
    }


@pytest.fixture
def sample_audio_analysis():
    """Análisis de audio de ejemplo"""
    return {
        "beats": [
            {"start": 0.0, "confidence": 0.9},
            {"start": 0.5, "confidence": 0.9},
            {"start": 1.0, "confidence": 0.9}
        ],
        "bars": [
            {"start": 0.0, "confidence": 0.9},
            {"start": 2.0, "confidence": 0.9}
        ],
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
                "duration": 0.5,
                "pitches": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2],
                "timbre": [1.0, 0.5, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
                "loudness_max": -5.0,
                "loudness_start": -6.0
            }
        ],
        "tatums": [
            {"start": 0.0, "confidence": 0.9},
            {"start": 0.25, "confidence": 0.9}
        ]
    }


@pytest.fixture
def analyzer_with_mocks():
    """Fixture mejorado para MusicAnalyzer con mocks configurados"""
    with patch('music_analyzer_ai.core.music_analyzer.GenreDetector') as mock_genre, \
         patch('music_analyzer_ai.core.music_analyzer.HarmonicAnalyzer') as mock_harmonic, \
         patch('music_analyzer_ai.core.music_analyzer.EmotionAnalyzer') as mock_emotion:
        
        mock_genre.return_value.detect_genre.return_value = {
            "genre": "Rock",
            "confidence": 0.9,
            "subgenres": ["Alternative Rock"]
        }
        mock_harmonic.return_value.analyze_harmonic_progression.return_value = {
            "progression": "I-V-vi-IV",
            "chords": ["Cmaj", "Gmaj", "Amin", "Fmaj"],
            "complexity": "moderate"
        }
        mock_emotion.return_value.analyze_emotions.return_value = {
            "primary_emotion": "Happy",
            "valence": 0.7,
            "energy": 0.8,
            "emotions": ["joy", "excitement"]
        }
        
        from ..core.music_analyzer import MusicAnalyzer
        analyzer = MusicAnalyzer()
        analyzer.genre_detector = mock_genre.return_value
        analyzer.harmonic_analyzer = mock_harmonic.return_value
        analyzer.emotion_analyzer = mock_emotion.return_value
        
        yield analyzer


@pytest.fixture
def test_helpers():
    """Fixture para helpers de test"""
    if TestHelpers is None:
        from test_helpers import TestHelpers as TH
        return TH
    return TestHelpers


@pytest.fixture
def mock_data_factory():
    """Fixture para factory de datos mock"""
    if MockDataFactory is None:
        from test_helpers import MockDataFactory as MDF
        return MDF
    return MockDataFactory

