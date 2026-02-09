"""
Tests para el servicio de notificaciones
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from services.notification_service import NotificationService


@pytest.mark.unit
class TestNotificationService:
    """Tests para el servicio de notificaciones"""
    
    @pytest.mark.asyncio
    async def test_notify_song_completed(self):
        """Test de notificación de canción completada"""
        with patch('services.notification_service.notify_song_status') as mock_notify:
            mock_notify.return_value = None
            
            await NotificationService.notify_song_completed(
                user_id="user-123",
                song_id="song-456",
                audio_url="https://example.com/audio.wav"
            )
            
            mock_notify.assert_called_once()
            call_args = mock_notify.call_args
            assert call_args[1]["user_id"] == "user-123"
            assert call_args[1]["song_id"] == "song-456"
            assert call_args[1]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_notify_song_completed_without_url(self):
        """Test de notificación sin URL"""
        with patch('services.notification_service.notify_song_status') as mock_notify:
            mock_notify.return_value = None
            
            await NotificationService.notify_song_completed(
                user_id="user-123",
                song_id="song-456"
            )
            
            mock_notify.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_notify_song_failed(self):
        """Test de notificación de fallo"""
        with patch('services.notification_service.notify_song_status') as mock_notify:
            mock_notify.return_value = None
            
            await NotificationService.notify_song_failed(
                user_id="user-123",
                song_id="song-456",
                error_message="Generation failed"
            )
            
            mock_notify.assert_called_once()
            call_args = mock_notify.call_args
            assert call_args[1]["status"] == "failed"
    
    @pytest.mark.asyncio
    async def test_notify_generation_started(self):
        """Test de notificación de inicio de generación"""
        with patch('services.notification_service.notify_song_status') as mock_status:
            with patch('services.notification_service.notify_generation_progress') as mock_progress:
                mock_status.return_value = None
                mock_progress.return_value = None
                
                await NotificationService.notify_generation_started(
                    user_id="user-123",
                    song_id="song-456"
                )
                
                mock_status.assert_called_once()
                mock_progress.assert_called_once()
                progress_call = mock_progress.call_args
                assert progress_call[1]["progress"] == 0.0
    
    @pytest.mark.asyncio
    async def test_notify_progress(self):
        """Test de notificación de progreso"""
        with patch('services.notification_service.notify_generation_progress') as mock_progress:
            mock_progress.return_value = None
            
            await NotificationService.notify_progress(
                user_id="user-123",
                song_id="song-456",
                progress=0.5,
                stage="generating"
            )
            
            mock_progress.assert_called_once()
            call_args = mock_progress.call_args
            assert call_args[1]["progress"] == 0.5
    
    @pytest.mark.asyncio
    async def test_notify_error_handling(self):
        """Test de manejo de errores en notificaciones"""
        with patch('services.notification_service.notify_song_status') as mock_notify:
            mock_notify.side_effect = Exception("Notification failed")
            
            # No debería lanzar excepción, solo loggear
            try:
                await NotificationService.notify_song_completed(
                    user_id="user-123",
                    song_id="song-456"
                )
                # Si llegamos aquí, el error fue manejado
                assert True
            except Exception:
                pytest.fail("Error should be handled internally")


@pytest.mark.integration
@pytest.mark.slow
class TestNotificationServiceIntegration:
    """Tests de integración para notificaciones"""
    
    @pytest.mark.asyncio
    async def test_full_notification_workflow(self):
        """Test del flujo completo de notificaciones"""
        with patch('services.notification_service.notify_song_status') as mock_status:
            with patch('services.notification_service.notify_generation_progress') as mock_progress:
                mock_status.return_value = None
                mock_progress.return_value = None
                
                user_id = "user-123"
                song_id = "song-456"
                
                # 1. Notificar inicio
                await NotificationService.notify_generation_started(user_id, song_id)
                
                # 2. Notificar progreso
                await NotificationService.notify_progress(user_id, song_id, 0.25, "processing")
                await NotificationService.notify_progress(user_id, song_id, 0.5, "generating")
                await NotificationService.notify_progress(user_id, song_id, 0.75, "finalizing")
                
                # 3. Notificar completado
                await NotificationService.notify_song_completed(user_id, song_id, "https://example.com/audio.wav")
                
                # Verificar que se llamaron las notificaciones
                assert mock_status.call_count >= 2  # started + completed
                assert mock_progress.call_count >= 4  # started + 3 progress updates



