"""
Tests refactorizados para las rutas de generación de letras
Usando clases base y helpers para eliminar duplicación
"""

import pytest
from unittest.mock import Mock
from fastapi import status

from test_helpers import (
    BaseAPITestCase,
    create_router_client,
    assert_standard_response,
    assert_error_response
)
from api.routes.lyrics import router
from services.lyrics_generator import LyricsGenerator


class TestLyricsRoutesRefactored(BaseAPITestCase):
    """Tests refactorizados usando BaseAPITestCase"""
    
    router = router
    
    @pytest.fixture
    def mock_lyrics_service(self):
        """Mock del servicio de generación de letras"""
        service = Mock(spec=LyricsGenerator)
        lyrics_obj = Mock()
        lyrics_obj.title = "Test Song"
        lyrics_obj.verses = ["Verse 1", "Verse 2"]
        lyrics_obj.chorus = "Chorus"
        service.generate_lyrics = Mock(return_value=lyrics_obj)
        return service
    
    def test_generate_lyrics_success(self, mock_lyrics_service):
        """Test de generación exitosa usando clase base"""
        client = self.create_client({
            "api.routes.lyrics.get_lyrics_generator": mock_lyrics_service,
            "api.routes.lyrics.get_current_user": {"user_id": "test_user"}
        })
        
        response = client.post(
            "/lyrics/generate",
            json={"prompt": "A happy pop song"}
        )
        
        self.assert_success_response(response)
        self.assert_response_contains_keys(response, ["title", "verses", "chorus"])
    
    def test_generate_lyrics_with_parameters(self, mock_lyrics_service):
        """Test con parámetros adicionales"""
        client = self.create_client({
            "api.routes.lyrics.get_lyrics_generator": mock_lyrics_service,
            "api.routes.lyrics.get_current_user": {"user_id": "test_user"}
        })
        
        response = client.post(
            "/lyrics/generate",
            json={
                "prompt": "A happy pop song",
                "genre": "pop",
                "mood": "happy",
                "language": "en"
            }
        )
        
        self.assert_success_response(response)
    
    def test_generate_lyrics_error_handling(self, mock_lyrics_service):
        """Test de manejo de errores"""
        mock_lyrics_service.generate_lyrics.side_effect = Exception("Generation failed")
        
        client = self.create_client({
            "api.routes.lyrics.get_lyrics_generator": mock_lyrics_service,
            "api.routes.lyrics.get_current_user": {"user_id": "test_user"}
        })
        
        response = client.post(
            "/lyrics/generate",
            json={"prompt": "Test"}
        )
        
        self.assert_error_response(response, expected_status=500)


# Ejemplo usando helpers funcionales en lugar de clase base
class TestLyricsRoutesFunctional:
    """Tests usando helpers funcionales"""
    
    @pytest.fixture
    def mock_lyrics_service(self):
        """Mock del servicio"""
        service = Mock(spec=LyricsGenerator)
        lyrics_obj = Mock()
        lyrics_obj.title = "Test Song"
        lyrics_obj.verses = ["Verse 1"]
        service.generate_lyrics = Mock(return_value=lyrics_obj)
        return service
    
    def test_generate_lyrics_using_helper(self, mock_lyrics_service):
        """Test usando create_router_client helper"""
        from test_helpers import create_router_client
        
        client = create_router_client(
            router=router,
            mocks={
                "api.routes.lyrics.get_lyrics_generator": mock_lyrics_service,
                "api.routes.lyrics.get_current_user": {"user_id": "test_user"}
            }
        )
        
        response = client.post(
            "/lyrics/generate",
            json={"prompt": "A happy pop song"}
        )
        
        assert_standard_response(
            response,
            expected_status=200,
            required_keys=["title", "verses"]
        )



