"""
Fixtures mejorados y reutilizables para tests
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import json


@pytest.fixture(scope="session")
def complete_spotify_data():
    """Fixture completo de datos de Spotify"""
    return {
        "track_info": {
            "id": "test_track_123",
            "name": "Test Track",
            "artists": [
                {"name": "Test Artist 1"},
                {"name": "Test Artist 2"}
            ],
            "album": {
                "id": "test_album_123",
                "name": "Test Album",
                "release_date": "2024-01-01",
                "images": [
                    {"url": "https://example.com/image.jpg", "height": 640, "width": 640}
                ]
            },
            "duration_ms": 200000,
            "popularity": 80,
            "external_urls": {
                "spotify": "https://open.spotify.com/track/test_track_123"
            },
            "preview_url": "https://p.scdn.co/mp3-preview/test.mp3"
        },
        "audio_features": {
            "key": 0,
            "mode": 1,
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
    }


@pytest.fixture
def multiple_tracks_data():
    """Fixture con múltiples tracks para comparación"""
    return [
        {
            "id": "track_1",
            "name": "Track 1",
            "audio_features": {
                "key": 0,
                "mode": 1,
                "tempo": 120.0,
                "energy": 0.8
            }
        },
        {
            "id": "track_2",
            "name": "Track 2",
            "audio_features": {
                "key": 0,
                "mode": 1,
                "tempo": 125.0,
                "energy": 0.7
            }
        },
        {
            "id": "track_3",
            "name": "Track 3",
            "audio_features": {
                "key": 1,
                "mode": 0,
                "tempo": 120.0,
                "energy": 0.9
            }
        }
    ]


@pytest.fixture
def mock_spotify_service():
    """Fixture para mock de SpotifyService"""
    with patch('music_analyzer_ai.services.spotify_service.SpotifyService') as mock_service:
        mock_instance = Mock()
        mock_instance.search_tracks.return_value = {
            "tracks": {
                "items": [
                    {
                        "id": "123",
                        "name": "Test Track",
                        "artists": [{"name": "Test Artist"}]
                    }
                ],
                "total": 1
            }
        }
        mock_instance.get_track.return_value = {
            "id": "123",
            "name": "Test Track",
            "artists": [{"name": "Test Artist"}]
        }
        mock_instance.get_track_audio_features.return_value = {
            "key": 0,
            "mode": 1,
            "tempo": 120.0,
            "energy": 0.8
        }
        mock_instance.get_track_audio_analysis.return_value = {
            "sections": [],
            "segments": []
        }
        mock_service.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def temp_storage():
    """Fixture para storage temporal"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_analysis_result():
    """Fixture con resultado de análisis completo"""
    return {
        "track_basic_info": {
            "name": "Test Track",
            "artists": ["Test Artist"],
            "album": "Test Album",
            "duration_ms": 200000,
            "duration_seconds": 200.0,
            "popularity": 80
        },
        "musical_analysis": {
            "key_signature": "C major",
            "root_note": "C",
            "mode": "Major",
            "tempo": {
                "bpm": 120.0,
                "category": "Moderado (Moderato)"
            },
            "time_signature": "4/4",
            "scale": {
                "name": "C major",
                "notes": ["C", "D", "E", "F", "G", "A", "B"]
            }
        },
        "technical_analysis": {
            "energy": {
                "value": 0.8,
                "description": "Alta energía"
            },
            "danceability": {
                "value": 0.7,
                "description": "Muy bailable"
            },
            "valence": {
                "value": 0.6,
                "description": "Feliz"
            }
        },
        "composition_analysis": {
            "structure": [],
            "composition_style": "Rock/Metal",
            "complexity": {
                "level": "Moderada",
                "score": 0.5
            }
        },
        "performance_analysis": {
            "timbre_analysis": [],
            "dynamic_range": {
                "range": 5.0,
                "description": "Rango dinámico moderado"
            }
        },
        "educational_insights": {
            "key_analysis": {
                "note": "C",
                "mode": "Major"
            },
            "learning_points": [
                "La canción está en C major",
                "Tempo: 120 BPM (Moderado (Moderato))"
            ]
        }
    }


@pytest.fixture
def mock_ml_services():
    """Fixture para mock de servicios ML"""
    with patch('music_analyzer_ai.services.ml_service.ML_AVAILABLE', True), \
         patch('music_analyzer_ai.services.ml_service.get_deep_analyzer') as mock_deep, \
         patch('music_analyzer_ai.services.ml_service.get_ml_analyzer') as mock_ml, \
         patch('music_analyzer_ai.services.ml_service.get_transformer_analyzer') as mock_transformer, \
         patch('music_analyzer_ai.services.ml_service.AudioFeatureExtractor') as mock_extractor, \
         patch('music_analyzer_ai.services.ml_service.create_default_pipeline') as mock_pipeline:
        
        yield {
            'deep': mock_deep,
            'ml': mock_ml,
            'transformer': mock_transformer,
            'extractor': mock_extractor,
            'pipeline': mock_pipeline
        }


@pytest.fixture
def error_scenarios():
    """Fixture con escenarios de error comunes"""
    return {
        "network_error": Exception("Network connection failed"),
        "timeout_error": TimeoutError("Request timed out"),
        "invalid_response": ValueError("Invalid response format"),
        "missing_data": KeyError("Required field missing"),
        "type_error": TypeError("Invalid type"),
        "permission_error": PermissionError("Access denied")
    }


@pytest.fixture
def edge_case_data():
    """Fixture con datos de casos edge"""
    return {
        "zero_values": {
            "key": 0,
            "mode": 0,
            "tempo": 0.0,
            "energy": 0.0,
            "danceability": 0.0
        },
        "max_values": {
            "key": 11,
            "mode": 1,
            "tempo": 300.0,
            "energy": 1.0,
            "danceability": 1.0
        },
        "invalid_key": {
            "key": -1,
            "mode": 1,
            "tempo": 120.0
        },
        "invalid_mode": {
            "key": 0,
            "mode": 2,  # Modo inválido
            "tempo": 120.0
        },
        "extreme_tempo": {
            "key": 0,
            "mode": 1,
            "tempo": 500.0  # Tempo extremo
        }
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

