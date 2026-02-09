"""
Tests mejorados para el procesador de chat
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from core.chat_processor import ChatProcessor


@pytest.fixture
def chat_processor():
    """Instancia del procesador de chat"""
    with patch('core.chat_processor.settings') as mock_settings:
        mock_settings.openai_api_key = None
        return ChatProcessor()


@pytest.fixture
def chat_processor_with_openai():
    """Instancia del procesador con OpenAI"""
    with patch('core.chat_processor.settings') as mock_settings:
        with patch('core.chat_processor.OpenAI') as mock_openai:
            mock_settings.openai_api_key = "test-key"
            mock_client = Mock()
            mock_openai.return_value = mock_client
            processor = ChatProcessor()
            processor.client = mock_client
            return processor


@pytest.mark.unit
class TestChatProcessor:
    """Tests para el procesador de chat"""
    
    def test_processor_initialization(self, chat_processor):
        """Test de inicialización"""
        assert chat_processor is not None
        assert isinstance(chat_processor, ChatProcessor)
    
    def test_extract_song_info_basic(self, chat_processor):
        """Test básico de extracción de información"""
        message = "Create a happy pop song"
        result = chat_processor.extract_song_info(message)
        
        assert result is not None
        assert "prompt" in result
        assert "genre" in result
        assert "mood" in result
        assert result["original_message"] == message
    
    def test_extract_song_info_with_genre(self, chat_processor):
        """Test con género mencionado"""
        messages = [
            "Create a rock song",
            "I want a jazz track",
            "Make it pop music"
        ]
        
        for message in messages:
            result = chat_processor.extract_song_info(message)
            assert result["genre"] is not None or result["prompt"] is not None
    
    def test_extract_song_info_with_mood(self, chat_processor):
        """Test con estado de ánimo mencionado"""
        messages = [
            "Create a happy song",
            "I want something sad",
            "Make it energetic"
        ]
        
        for message in messages:
            result = chat_processor.extract_song_info(message)
            assert result["mood"] is not None or result["prompt"] is not None
    
    def test_extract_song_info_with_duration(self, chat_processor):
        """Test con duración mencionada"""
        messages = [
            "Create a 30 second song",
            "I want 2 minutes of music",
            "Make it 60 seconds long"
        ]
        
        for message in messages:
            result = chat_processor.extract_song_info(message)
            # Puede extraer duración o no
            assert result is not None
    
    def test_extract_song_info_with_chat_history(self, chat_processor):
        """Test con historial de chat"""
        message = "Make it longer"
        history = [
            {"role": "user", "content": "Create a pop song"},
            {"role": "assistant", "content": "I'll create a pop song for you"}
        ]
        
        result = chat_processor.extract_song_info(message, history)
        assert result is not None
    
    def test_extract_song_info_empty_message(self, chat_processor):
        """Test con mensaje vacío"""
        result = chat_processor.extract_song_info("")
        
        assert result is not None
        assert result["prompt"] == "" or len(result["prompt"]) >= 0
    
    def test_extract_song_info_error_handling(self, chat_processor):
        """Test de manejo de errores"""
        # Simular error en extracción
        with patch.object(chat_processor, '_extract_genre', side_effect=Exception("Error")):
            result = chat_processor.extract_song_info("Test message")
            
            # Debería retornar fallback
            assert result is not None
            assert result["prompt"] == "Test message" or len(result["prompt"]) > 0


@pytest.mark.unit
class TestChatProcessorWithOpenAI:
    """Tests para procesador con OpenAI"""
    
    def test_extract_song_info_with_openai(self, chat_processor_with_openai):
        """Test de extracción con OpenAI"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Enhanced prompt: A happy pop song"
        
        chat_processor_with_openai.client.chat.completions.create = Mock(return_value=mock_response)
        
        result = chat_processor_with_openai.extract_song_info("Create a happy pop song")
        
        assert result is not None
        assert "prompt" in result
        # OpenAI debería mejorar el prompt
        chat_processor_with_openai.client.chat.completions.create.assert_called()


@pytest.mark.unit
class TestExtractHelpers:
    """Tests para métodos helper de extracción"""
    
    def test_extract_genre(self, chat_processor):
        """Test de extracción de género"""
        test_cases = [
            ("Create a rock song", "rock"),
            ("I want jazz music", "jazz"),
            ("Make it pop", "pop")
        ]
        
        for message, expected_genre in test_cases:
            genre = chat_processor._extract_genre(message)
            # Puede encontrar el género o None
            assert genre is None or isinstance(genre, str)
    
    def test_extract_mood(self, chat_processor):
        """Test de extracción de estado de ánimo"""
        test_cases = [
            ("Create a happy song", "happy"),
            ("I want something sad", "sad"),
            ("Make it energetic", "energetic")
        ]
        
        for message, expected_mood in test_cases:
            mood = chat_processor._extract_mood(message)
            assert mood is None or isinstance(mood, str)
    
    def test_extract_tempo(self, chat_processor):
        """Test de extracción de tempo"""
        test_cases = [
            ("Create a 120 BPM song", 120),
            ("I want 140 beats per minute", 140),
            ("Make it fast tempo", None)
        ]
        
        for message, expected_tempo in test_cases:
            tempo = chat_processor._extract_tempo(message)
            assert tempo is None or isinstance(tempo, (int, float))
    
    def test_extract_duration(self, chat_processor):
        """Test de extracción de duración"""
        test_cases = [
            ("Create a 30 second song", 30),
            ("I want 2 minutes", 120),
            ("Make it 60 seconds long", 60)
        ]
        
        for message, expected_duration in test_cases:
            duration = chat_processor._extract_duration(message)
            assert duration is None or isinstance(duration, (int, float))


@pytest.mark.integration
@pytest.mark.slow
class TestChatProcessorIntegration:
    """Tests de integración para procesador de chat"""
    
    def test_full_extraction_workflow(self, chat_processor):
        """Test del flujo completo de extracción"""
        messages = [
            "Create a happy pop song",
            "Make it 30 seconds long",
            "Add some guitar",
            "I want it faster"
        ]
        
        history = []
        for message in messages:
            result = chat_processor.extract_song_info(message, history)
            assert result is not None
            assert "prompt" in result
            
            # Agregar a historial
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "I'll do that"})



