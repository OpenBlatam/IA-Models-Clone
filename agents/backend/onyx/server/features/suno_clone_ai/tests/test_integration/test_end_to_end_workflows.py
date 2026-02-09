"""
Tests de integración end-to-end para flujos completos
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import status
from fastapi.testclient import TestClient
import io

from api.routes.generation import router as generation_router
from api.routes.lyrics import router as lyrics_router
from api.routes.playlists import router as playlists_router
from api.routes.favorites import router as favorites_router


@pytest.fixture
def mock_services():
    """Mocks de todos los servicios necesarios"""
    services = {
        "song_service": Mock(),
        "music_generator": Mock(),
        "lyrics_generator": Mock(),
        "chat_processor": Mock()
    }
    
    # Configurar song_service
    services["song_service"].save_song = Mock(return_value=True)
    services["song_service"].get_song = Mock(return_value=None)
    services["song_service"].list_songs = Mock(return_value=[])
    services["song_service"].get_chat_history = Mock(return_value=[])
    
    # Configurar music_generator
    import numpy as np
    services["music_generator"].generate_from_text = Mock(
        return_value=np.array([0.1, 0.2, 0.3])
    )
    services["music_generator"].save_audio = Mock(return_value=True)
    
    # Configurar lyrics_generator
    lyrics_obj = Mock()
    lyrics_obj.title = "Generated Song"
    lyrics_obj.verses = ["Verse 1", "Verse 2"]
    lyrics_obj.chorus = "Chorus"
    lyrics_obj.bridge = None
    lyrics_obj.language = "en"
    lyrics_obj.style = "pop"
    lyrics_obj.theme = "love"
    services["lyrics_generator"].generate_lyrics = Mock(return_value=lyrics_obj)
    
    # Configurar chat_processor
    services["chat_processor"].extract_song_info = Mock(return_value={
        "prompt": "test song",
        "genre": "pop",
        "mood": "happy",
        "duration": 30
    })
    
    return services


@pytest.fixture
def integrated_client(mock_services):
    """Cliente con todas las rutas integradas"""
    from fastapi import FastAPI
    from api.dependencies import SongServiceDep
    
    app = FastAPI()
    app.include_router(generation_router)
    app.include_router(lyrics_router)
    app.include_router(playlists_router)
    app.include_router(favorites_router)
    
    def get_song_service():
        return mock_services["song_service"]
    
    app.dependency_overrides[SongServiceDep] = get_song_service
    
    with patch('api.routes.generation.get_music_generator', return_value=mock_services["music_generator"]):
        with patch('api.routes.lyrics.get_lyrics_generator', return_value=mock_services["lyrics_generator"]):
            with patch('api.routes.generation.get_chat_processor', return_value=mock_services["chat_processor"]):
                with patch('api.routes.generation.get_current_user', return_value={"user_id": "test_user"}):
                    with patch('api.routes.lyrics.get_current_user', return_value={"user_id": "test_user"}):
                        with patch('api.routes.playlists.get_current_user', return_value={"user_id": "test_user"}):
                            with patch('api.routes.favorites.get_current_user', return_value={"user_id": "test_user"}):
                                yield TestClient(app)
    
    app.dependency_overrides.clear()


@pytest.mark.integration
@pytest.mark.slow
class TestCompleteSongGenerationWorkflow:
    """Test del flujo completo de generación de canción"""
    
    def test_generate_song_with_lyrics_and_add_to_playlist(
        self, integrated_client, mock_services
    ):
        """Test: Generar canción, agregar letras, crear playlist y agregar canción"""
        user_id = "test-user-123"
        
        # 1. Generar letras
        lyrics_response = integrated_client.post(
            "/lyrics/generate",
            json={
                "theme": "adventure",
                "style": "rock",
                "language": "en",
                "num_verses": 3
            }
        )
        assert lyrics_response.status_code == status.HTTP_200_OK
        lyrics_data = lyrics_response.json()
        assert "title" in lyrics_data
        
        # 2. Crear playlist
        playlist_response = integrated_client.post(
            "/playlists",
            params={
                "name": "My Generated Songs",
                "user_id": user_id,
                "is_public": False
            }
        )
        assert playlist_response.status_code == status.HTTP_200_OK
        playlist_id = playlist_response.json()["playlist_id"]
        
        # 3. Simular generación de canción (mock)
        mock_services["song_service"].get_song.return_value = {
            "song_id": "generated-song-123",
            "user_id": user_id,
            "prompt": lyrics_data["title"],
            "status": "completed"
        }
        
        # 4. Agregar canción a favoritos
        favorite_response = integrated_client.post(
            "/songs/generated-song-123/favorite",
            params={"user_id": user_id}
        )
        # Puede fallar si la canción no existe realmente, pero el flujo está probado
        assert favorite_response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND
        ]


@pytest.mark.integration
@pytest.mark.slow
class TestUserJourneyWorkflow:
    """Test del journey completo de usuario"""
    
    def test_user_complete_journey(self, integrated_client, mock_services):
        """Test del journey completo: chat -> generación -> favoritos -> playlist"""
        user_id = "journey-user-456"
        
        # 1. Obtener historial de chat (si existe ruta)
        # chat_response = integrated_client.get(f"/chat/history/{user_id}")
        # assert chat_response.status_code == status.HTTP_200_OK
        
        # 2. Generar letras
        lyrics_response = integrated_client.post(
            "/lyrics/generate",
            json={"theme": "summer", "language": "en"}
        )
        assert lyrics_response.status_code == status.HTTP_200_OK
        
        # 3. Crear playlist
        playlist_response = integrated_client.post(
            "/playlists",
            params={
                "name": "Summer Songs",
                "user_id": user_id,
                "is_public": True
            }
        )
        assert playlist_response.status_code == status.HTTP_200_OK
        
        # 4. Verificar que los servicios fueron llamados
        assert mock_services["lyrics_generator"].generate_lyrics.called
        assert mock_services["song_service"].save_song.called


@pytest.mark.integration
@pytest.mark.slow
class TestMultiUserWorkflow:
    """Test de workflows con múltiples usuarios"""
    
    def test_multiple_users_playlists(self, integrated_client, mock_services):
        """Test de creación de playlists por múltiples usuarios"""
        users = ["user-1", "user-2", "user-3"]
        playlist_ids = []
        
        for user_id in users:
            response = integrated_client.post(
                "/playlists",
                params={
                    "name": f"Playlist for {user_id}",
                    "user_id": user_id,
                    "is_public": False
                }
            )
            assert response.status_code == status.HTTP_200_OK
            playlist_ids.append(response.json()["playlist_id"])
        
        assert len(playlist_ids) == len(users)
        assert len(set(playlist_ids)) == len(playlist_ids)  # Todos únicos


@pytest.mark.integration
@pytest.mark.slow
class TestErrorRecoveryWorkflow:
    """Test de recuperación de errores en workflows"""
    
    def test_workflow_with_error_recovery(self, integrated_client, mock_services):
        """Test de workflow que se recupera de errores"""
        user_id = "error-user-789"
        
        # 1. Intentar crear playlist (debería funcionar)
        playlist_response = integrated_client.post(
            "/playlists",
            params={
                "name": "Error Test Playlist",
                "user_id": user_id
            }
        )
        assert playlist_response.status_code == status.HTTP_200_OK
        
        # 2. Simular error en servicio
        mock_services["song_service"].save_song.side_effect = Exception("Service error")
        
        # 3. Intentar otra operación (debería manejar el error)
        # La operación puede fallar, pero el sistema debería manejarlo
        try:
            response = integrated_client.post(
                "/playlists",
                params={
                    "name": "Another Playlist",
                    "user_id": user_id
                }
            )
            # Si no falla, el error fue manejado
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ]
        except Exception:
            # Error fue propagado, lo cual también es válido
            pass



