"""
Tests mejorados para validadores core
"""

import pytest
from core.validators import Validator, ValidationError, validate_and_raise
import uuid
from datetime import datetime


@pytest.mark.unit
class TestValidator:
    """Tests para la clase Validator"""
    
    def test_validate_uuid_valid(self):
        """Test de validación de UUID válido"""
        valid_uuid = str(uuid.uuid4())
        assert Validator.validate_uuid(valid_uuid) is True
    
    def test_validate_uuid_invalid(self):
        """Test de validación de UUID inválido"""
        invalid_uuids = [
            "not-a-uuid",
            "123",
            "",
            "123e4567-e89b-12d3-a456",
            None
        ]
        
        for invalid_uuid in invalid_uuids:
            if invalid_uuid is not None:
                assert Validator.validate_uuid(invalid_uuid) is False
    
    def test_validate_email_valid(self):
        """Test de validación de email válido"""
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "user+tag@example.com"
        ]
        
        for email in valid_emails:
            assert Validator.validate_email(email) is True
    
    def test_validate_email_invalid(self):
        """Test de validación de email inválido"""
        invalid_emails = [
            "not-an-email",
            "@example.com",
            "user@",
            "user@domain",
            "user name@example.com"
        ]
        
        for email in invalid_emails:
            assert Validator.validate_email(email) is False
    
    def test_validate_url_valid(self):
        """Test de validación de URL válida"""
        valid_urls = [
            "https://example.com",
            "http://example.com/path",
            "https://example.com:8080/path?query=value"
        ]
        
        for url in valid_urls:
            assert Validator.validate_url(url) is True
    
    def test_validate_url_invalid(self):
        """Test de validación de URL inválida"""
        invalid_urls = [
            "not-a-url",
            "example.com",
            "ftp://example.com",
            ""
        ]
        
        for url in invalid_urls:
            assert Validator.validate_url(url) is False
    
    def test_validate_iso_datetime_valid(self):
        """Test de validación de datetime ISO válido"""
        valid_datetimes = [
            "2024-01-01T00:00:00",
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00+00:00"
        ]
        
        for dt in valid_datetimes:
            assert Validator.validate_iso_datetime(dt) is True
    
    def test_validate_iso_datetime_invalid(self):
        """Test de validación de datetime ISO inválido"""
        invalid_datetimes = [
            "not-a-datetime",
            "2024-01-01",
            "2024/01/01",
            ""
        ]
        
        for dt in invalid_datetimes:
            assert Validator.validate_iso_datetime(dt) is False
    
    def test_validate_audio_format_valid(self):
        """Test de validación de formato de audio válido"""
        valid_formats = [
            "song.wav",
            "song.mp3",
            "song.ogg",
            "song.flac",
            "song.m4a"
        ]
        
        for filename in valid_formats:
            assert Validator.validate_audio_format(filename) is True
    
    def test_validate_audio_format_invalid(self):
        """Test de validación de formato de audio inválido"""
        invalid_formats = [
            "song.exe",
            "song.txt",
            "song",
            "song.unknown"
        ]
        
        for filename in invalid_formats:
            assert Validator.validate_audio_format(filename) is False
    
    def test_validate_prompt_valid(self):
        """Test de validación de prompt válido"""
        valid_prompts = [
            "A happy song",
            "x" * 100,  # Límite mínimo
            "x" * 1000  # Límite máximo
        ]
        
        for prompt in valid_prompts:
            assert Validator.validate_prompt(prompt) is True
    
    def test_validate_prompt_invalid(self):
        """Test de validación de prompt inválido"""
        invalid_prompts = [
            "",  # Muy corto
            "x" * 1001,  # Muy largo
            None,
            123  # Tipo incorrecto
        ]
        
        for prompt in invalid_prompts:
            if prompt is not None:
                assert Validator.validate_prompt(prompt) is False
    
    def test_validate_bpm_valid(self):
        """Test de validación de BPM válido"""
        valid_bpms = [20, 60, 120, 180, 300]
        
        for bpm in valid_bpms:
            assert Validator.validate_bpm(bpm) is True
    
    def test_validate_bpm_invalid(self):
        """Test de validación de BPM inválido"""
        invalid_bpms = [0, 19, 301, -10, "120"]
        
        for bpm in invalid_bpms:
            if isinstance(bpm, (int, float)):
                assert Validator.validate_bpm(bpm) is False
    
    def test_validate_duration_valid(self):
        """Test de validación de duración válida"""
        valid_durations = [1, 30, 180, 3600]
        
        for duration in valid_durations:
            assert Validator.validate_duration(duration) is True
    
    def test_validate_duration_invalid(self):
        """Test de validación de duración inválida"""
        invalid_durations = [0, -1, 3601, "30"]
        
        for duration in invalid_durations:
            if isinstance(duration, (int, float)):
                assert Validator.validate_duration(duration) is False
    
    def test_validate_price_valid(self):
        """Test de validación de precio válido"""
        valid_prices = [0, 5.99, 10.0, 100]
        
        for price in valid_prices:
            assert Validator.validate_price(price) is True
    
    def test_validate_price_invalid(self):
        """Test de validación de precio inválido"""
        invalid_prices = [-1, -10.5, "10"]
        
        for price in invalid_prices:
            if isinstance(price, (int, float)):
                assert Validator.validate_price(price) is False
    
    def test_validate_rating_valid(self):
        """Test de validación de rating válido"""
        valid_ratings = [1, 2, 3, 4, 5]
        
        for rating in valid_ratings:
            assert Validator.validate_rating(rating) is True
    
    def test_validate_rating_invalid(self):
        """Test de validación de rating inválido"""
        invalid_ratings = [0, 6, -1, "5"]
        
        for rating in invalid_ratings:
            if isinstance(rating, int):
                assert Validator.validate_rating(rating) is False


@pytest.mark.unit
class TestValidationError:
    """Tests para ValidationError"""
    
    def test_validation_error_creation(self):
        """Test de creación de ValidationError"""
        error = ValidationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)


@pytest.mark.unit
class TestValidateAndRaise:
    """Tests para validate_and_raise"""
    
    def test_validate_and_raise_success(self):
        """Test cuando la validación es exitosa"""
        def validator(value):
            return value > 0
        
        # No debería lanzar excepción
        try:
            validate_and_raise(validator, 10, "Value must be positive")
            assert True
        except ValidationError:
            pytest.fail("Should not raise ValidationError")
    
    def test_validate_and_raise_failure(self):
        """Test cuando la validación falla"""
        def validator(value):
            return value > 0
        
        with pytest.raises(ValidationError) as exc_info:
            validate_and_raise(validator, -10, "Value must be positive")
        
        assert "Value must be positive" in str(exc_info.value)


@pytest.mark.integration
class TestValidatorsIntegration:
    """Tests de integración para validadores"""
    
    def test_multiple_validations(self):
        """Test de múltiples validaciones"""
        # Validar múltiples campos
        assert Validator.validate_uuid(str(uuid.uuid4()))
        assert Validator.validate_email("test@example.com")
        assert Validator.validate_url("https://example.com")
        assert Validator.validate_bpm(120)
        assert Validator.validate_duration(30)
        assert Validator.validate_price(9.99)
        assert Validator.validate_rating(5)
    
    def test_validation_workflow(self):
        """Test del flujo completo de validación"""
        # Simular validación de datos de canción
        song_data = {
            "song_id": str(uuid.uuid4()),
            "duration": 30,
            "bpm": 120,
            "price": 9.99,
            "rating": 5
        }
        
        assert Validator.validate_uuid(song_data["song_id"])
        assert Validator.validate_duration(song_data["duration"])
        assert Validator.validate_bpm(song_data["bpm"])
        assert Validator.validate_price(song_data["price"])
        assert Validator.validate_rating(song_data["rating"])
