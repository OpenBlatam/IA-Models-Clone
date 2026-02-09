"""
Tests para las rutas de análisis de sentimiento
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import status
from fastapi.testclient import TestClient
import io

from api.routes.sentiment import router
from services.sentiment_analysis import SentimentService
from services.audio_transcription import TranscriptionService


@pytest.fixture
def mock_sentiment_service():
    """Mock del servicio de análisis de sentimiento"""
    service = Mock(spec=SentimentService)
    
    # Mock de resultado de sentimiento
    sentiment_result = Mock()
    sentiment_result.label = Mock()
    sentiment_result.label.value = "positive"
    sentiment_result.score = 0.85
    sentiment_result.polarity = 0.7
    sentiment_result.emotions = {"joy": 0.8, "excitement": 0.6}
    
    service.analyze_text = Mock(return_value=sentiment_result)
    service.analyze_audio = Mock(return_value=sentiment_result)
    service.analyze_batch = Mock(return_value=[sentiment_result, sentiment_result])
    
    return service


@pytest.fixture
def mock_transcription_service():
    """Mock del servicio de transcripción"""
    service = Mock(spec=TranscriptionService)
    transcription_result = Mock()
    transcription_result.text = "This is a happy song"
    service.transcribe = Mock(return_value=transcription_result)
    return service


@pytest.fixture
def sample_audio_file():
    """Archivo de audio de prueba"""
    return io.BytesIO(b"fake audio content for sentiment analysis")


@pytest.fixture
def client(mock_sentiment_service, mock_transcription_service):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.sentiment.get_sentiment_service', return_value=mock_sentiment_service):
        with patch('api.routes.sentiment.get_transcription_service', return_value=mock_transcription_service):
            with patch('api.routes.sentiment.get_current_user', return_value={"user_id": "test_user"}):
                yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestAnalyzeTextSentiment:
    """Tests para análisis de sentimiento de texto"""
    
    def test_analyze_text_success(self, client, mock_sentiment_service):
        """Test de análisis exitoso"""
        response = client.post(
            "/sentiment/analyze-text",
            json={"text": "I love this happy song!"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "sentiment" in data
        assert data["sentiment"] == "positive"
        assert "score" in data
        assert data["score"] == 0.85
        assert "polarity" in data
        assert "emotions" in data
    
    def test_analyze_text_different_sentiments(self, client, mock_sentiment_service):
        """Test con diferentes tipos de sentimiento"""
        sentiments = ["positive", "negative", "neutral"]
        
        for sentiment in sentiments:
            mock_result = Mock()
            mock_result.label = Mock()
            mock_result.label.value = sentiment
            mock_result.score = 0.8
            mock_result.polarity = 0.5 if sentiment == "neutral" else (0.7 if sentiment == "positive" else -0.7)
            mock_result.emotions = {}
            mock_sentiment_service.analyze_text.return_value = mock_result
            
            response = client.post(
                "/sentiment/analyze-text",
                json={"text": f"Test {sentiment} text"}
            )
            assert response.status_code == status.HTTP_200_OK
            assert response.json()["sentiment"] == sentiment
    
    def test_analyze_text_empty(self, client):
        """Test con texto vacío"""
        response = client.post(
            "/sentiment/analyze-text",
            json={"text": ""}
        )
        
        # Puede ser válido o inválido dependiendo de la validación
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]
    
    def test_analyze_text_error_handling(self, client, mock_sentiment_service):
        """Test de manejo de errores"""
        mock_sentiment_service.analyze_text.side_effect = Exception("Analysis failed")
        
        response = client.post(
            "/sentiment/analyze-text",
            json={"text": "Test text"}
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Error analyzing sentiment" in response.json()["detail"]


@pytest.mark.unit
@pytest.mark.api
class TestAnalyzeAudioSentiment:
    """Tests para análisis de sentimiento de audio"""
    
    def test_analyze_audio_success(self, client, mock_sentiment_service, mock_transcription_service, sample_audio_file):
        """Test de análisis exitoso de audio"""
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            with patch('os.path.splitext', return_value=("test", ".wav")):
                with patch('os.unlink'):
                    mock_file = Mock()
                    mock_file.name = "/tmp/test.wav"
                    mock_file.__enter__ = Mock(return_value=mock_file)
                    mock_file.__exit__ = Mock(return_value=None)
                    mock_file.write = Mock()
                    mock_temp.return_value = mock_file
                    
                    response = client.post(
                        "/sentiment/analyze-audio",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "sentiment" in data
        assert "score" in data
        assert "file" in data


@pytest.mark.unit
@pytest.mark.api
class TestAnalyzeBatchSentiment:
    """Tests para análisis en batch"""
    
    def test_analyze_batch_success(self, client, mock_sentiment_service):
        """Test de análisis en batch exitoso"""
        response = client.post(
            "/sentiment/analyze-batch",
            json={"texts": ["Text 1", "Text 2", "Text 3"]}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 3


@pytest.mark.integration
@pytest.mark.api
class TestSentimentIntegration:
    """Tests de integración para análisis de sentimiento"""
    
    def test_full_sentiment_workflow(self, client, mock_sentiment_service, mock_transcription_service, sample_audio_file):
        """Test del flujo completo de análisis de sentimiento"""
        # 1. Analizar texto
        text_response = client.post(
            "/sentiment/analyze-text",
            json={"text": "This is a great song!"}
        )
        assert text_response.status_code == status.HTTP_200_OK
        
        # 2. Analizar audio
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            with patch('os.path.splitext', return_value=("test", ".wav")):
                with patch('os.unlink'):
                    mock_file = Mock()
                    mock_file.name = "/tmp/test.wav"
                    mock_file.__enter__ = Mock(return_value=mock_file)
                    mock_file.__exit__ = Mock(return_value=None)
                    mock_file.write = Mock()
                    mock_temp.return_value = mock_file
                    
                    audio_response = client.post(
                        "/sentiment/analyze-audio",
                        files={"file": ("test.wav", sample_audio_file, "audio/wav")}
                    )
                    assert audio_response.status_code == status.HTTP_200_OK
        
        # 3. Analizar en batch
        batch_response = client.post(
            "/sentiment/analyze-batch",
            json={"texts": ["Happy", "Sad", "Neutral"]}
        )
        assert batch_response.status_code == status.HTTP_200_OK



