"""
Tests mejorados para el manejador de errores
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException, status

from core.error_handler import ErrorHandler


@pytest.mark.unit
class TestErrorHandler:
    """Tests para el manejador de errores"""
    
    def test_handle_generation_error_cuda(self):
        """Test de manejo de error CUDA"""
        error = Exception("CUDA out of memory")
        result = ErrorHandler.handle_generation_error(error)
        
        assert isinstance(result, HTTPException)
        assert result.status_code == status.HTTP_507_INSUFFICIENT_STORAGE
        assert "memory" in result.detail.lower()
    
    def test_handle_generation_error_model_not_found(self):
        """Test de manejo de error de modelo no encontrado"""
        error = Exception("Model not found")
        result = ErrorHandler.handle_generation_error(error)
        
        assert isinstance(result, HTTPException)
        assert result.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "model" in result.detail.lower()
    
    def test_handle_generation_error_generic(self):
        """Test de manejo de error genérico"""
        error = Exception("Generic error")
        result = ErrorHandler.handle_generation_error(error)
        
        assert isinstance(result, HTTPException)
        assert result.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Error generating music" in result.detail
    
    def test_handle_generation_error_with_context(self):
        """Test de manejo de error con contexto"""
        error = Exception("Test error")
        context = {"user_id": "user-123", "song_id": "song-456"}
        
        with patch('core.error_handler.logger') as mock_logger:
            result = ErrorHandler.handle_generation_error(error, context)
            
            assert isinstance(result, HTTPException)
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            assert "Generation error" in call_args[0][0]
            assert call_args[1]["extra"] == context
    
    def test_handle_audio_processing_error_file_not_found(self):
        """Test de manejo de error de archivo no encontrado"""
        error = Exception("File not found")
        result = ErrorHandler.handle_audio_processing_error(error)
        
        assert isinstance(result, HTTPException)
        assert result.status_code == status.HTTP_404_NOT_FOUND
        assert "file not found" in result.detail.lower()
    
    def test_handle_audio_processing_error_format(self):
        """Test de manejo de error de formato"""
        error = Exception("Unsupported format")
        result = ErrorHandler.handle_audio_processing_error(error)
        
        assert isinstance(result, HTTPException)
        assert result.status_code == status.HTTP_400_BAD_REQUEST
        assert "format" in result.detail.lower()
    
    def test_handle_audio_processing_error_generic(self):
        """Test de manejo de error genérico de audio"""
        error = Exception("Generic audio error")
        result = ErrorHandler.handle_audio_processing_error(error)
        
        assert isinstance(result, HTTPException)
        assert result.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Error processing audio" in result.detail
    
    def test_handle_validation_error(self):
        """Test de manejo de error de validación"""
        error = ValueError("Invalid input")
        result = ErrorHandler.handle_validation_error(error)
        
        assert isinstance(result, HTTPException)
        assert result.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid input" in result.detail
    
    def test_handle_validation_error_with_context(self):
        """Test de manejo de error de validación con contexto"""
        error = ValueError("Invalid value")
        context = {"field": "duration", "value": -10}
        
        with patch('core.error_handler.logger') as mock_logger:
            result = ErrorHandler.handle_validation_error(error, context)
            
            assert isinstance(result, HTTPException)
            mock_logger.warning.assert_called_once()
    
    def test_handle_cache_error(self):
        """Test de manejo de error de caché"""
        error = Exception("Cache connection failed")
        result = ErrorHandler.handle_cache_error(error)
        
        # Los errores de caché no deberían detener el proceso
        assert result is None
    
    def test_handle_cache_error_logging(self):
        """Test de logging de errores de caché"""
        error = Exception("Cache error")
        context = {"key": "test_key"}
        
        with patch('core.error_handler.logger') as mock_logger:
            result = ErrorHandler.handle_cache_error(error, context)
            
            assert result is None
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert "Cache error" in call_args[0][0]
            assert call_args[1]["extra"] == context


@pytest.mark.integration
class TestErrorHandlerIntegration:
    """Tests de integración para el manejador de errores"""
    
    def test_error_handling_workflow(self):
        """Test del flujo completo de manejo de errores"""
        # 1. Error de generación
        gen_error = Exception("CUDA out of memory")
        gen_result = ErrorHandler.handle_generation_error(gen_error)
        assert isinstance(gen_result, HTTPException)
        
        # 2. Error de procesamiento de audio
        audio_error = Exception("File not found")
        audio_result = ErrorHandler.handle_audio_processing_error(audio_error)
        assert isinstance(audio_result, HTTPException)
        
        # 3. Error de validación
        val_error = ValueError("Invalid input")
        val_result = ErrorHandler.handle_validation_error(val_error)
        assert isinstance(val_result, HTTPException)
        
        # 4. Error de caché (no crítico)
        cache_error = Exception("Cache unavailable")
        cache_result = ErrorHandler.handle_cache_error(cache_error)
        assert cache_result is None
    
    def test_error_handler_different_error_types(self):
        """Test con diferentes tipos de errores"""
        error_types = [
            (Exception("CUDA error"), "generation"),
            (Exception("File not found"), "audio"),
            (ValueError("Invalid"), "validation"),
            (Exception("Cache error"), "cache")
        ]
        
        for error, error_type in error_types:
            if error_type == "generation":
                result = ErrorHandler.handle_generation_error(error)
                assert isinstance(result, HTTPException)
            elif error_type == "audio":
                result = ErrorHandler.handle_audio_processing_error(error)
                assert isinstance(result, HTTPException)
            elif error_type == "validation":
                result = ErrorHandler.handle_validation_error(error)
                assert isinstance(result, HTTPException)
            elif error_type == "cache":
                result = ErrorHandler.handle_cache_error(error)
                assert result is None
