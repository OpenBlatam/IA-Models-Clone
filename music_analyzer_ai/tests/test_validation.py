"""
Tests de validación de datos
"""

import pytest
from unittest.mock import Mock


class TestDataValidator:
    """Tests para DataValidator"""
    
    @pytest.fixture
    def validator(self):
        """Fixture para crear DataValidator"""
        from ..validation.data_validator import DataValidator
        return DataValidator()
    
    def test_validate_track_id(self, validator):
        """Test de validación de track ID"""
        if validator is None:
            pytest.skip("DataValidator not available")
        
        assert validator.validate_track_id("valid_track_id_123") == True
        assert validator.validate_track_id("") == False
        assert validator.validate_track_id(None) == False
    
    def test_validate_audio_features(self, validator, sample_audio_features):
        """Test de validación de características de audio"""
        if validator is None:
            pytest.skip("DataValidator not available")
        
        assert validator.validate_audio_features(sample_audio_features) == True
        
        # Características inválidas
        invalid_features = sample_audio_features.copy()
        invalid_features["energy"] = 1.5  # Fuera de rango
        assert validator.validate_audio_features(invalid_features) == False
    
    def test_validate_user_id(self, validator):
        """Test de validación de user ID"""
        if validator is None:
            pytest.skip("DataValidator not available")
        
        assert validator.validate_user_id("user123") == True
        assert validator.validate_user_id("") == False
        assert validator.validate_user_id(None) == False
    
    def test_validate_playlist_data(self, validator):
        """Test de validación de datos de playlist"""
        if validator is None:
            pytest.skip("DataValidator not available")
        
        valid_playlist = {
            "name": "My Playlist",
            "is_public": False,
            "tracks": []
        }
        
        assert validator.validate_playlist_data(valid_playlist) == True
        
        invalid_playlist = {
            "name": "",  # Nombre vacío
            "is_public": False
        }
        
        assert validator.validate_playlist_data(invalid_playlist) == False


class TestInputSanitization:
    """Tests de sanitización de entrada"""
    
    def test_sanitize_string(self):
        """Test de sanitización de strings"""
        def sanitize_string(input_str, max_length=100):
            if not isinstance(input_str, str):
                return None
            # Remover caracteres peligrosos
            dangerous = ["<", ">", "&", '"', "'"]
            for char in dangerous:
                input_str = input_str.replace(char, "")
            # Limitar longitud
            if len(input_str) > max_length:
                input_str = input_str[:max_length]
            return input_str
        
        assert sanitize_string("normal text") == "normal text"
        assert sanitize_string("<script>alert('xss')</script>") == "scriptalertxssscript"
        assert len(sanitize_string("a" * 200)) == 100
    
    def test_sanitize_numeric(self):
        """Test de sanitización de números"""
        def sanitize_numeric(value, min_val=0, max_val=100):
            try:
                num = float(value)
                if num < min_val or num > max_val:
                    return None
                return num
            except (ValueError, TypeError):
                return None
        
        assert sanitize_numeric("50") == 50.0
        assert sanitize_numeric("150") == None  # Fuera de rango
        assert sanitize_numeric("invalid") == None
        assert sanitize_numeric(None) == None
    
    def test_sanitize_list(self):
        """Test de sanitización de listas"""
        def sanitize_list(input_list, max_items=100):
            if not isinstance(input_list, list):
                return None
            if len(input_list) > max_items:
                return None
            return input_list[:max_items]
        
        assert sanitize_list([1, 2, 3]) == [1, 2, 3]
        assert sanitize_list(list(range(150))) == None  # Demasiados items
        assert sanitize_list("not a list") == None


class TestRangeValidation:
    """Tests de validación de rangos"""
    
    def test_validate_tempo_range(self):
        """Test de validación de rango de tempo"""
        def validate_tempo(tempo):
            return 0 <= tempo <= 300
        
        assert validate_tempo(120) == True
        assert validate_tempo(0) == True
        assert validate_tempo(300) == True
        assert validate_tempo(-10) == False
        assert validate_tempo(500) == False
    
    def test_validate_energy_range(self):
        """Test de validación de rango de energía"""
        def validate_energy(energy):
            return 0.0 <= energy <= 1.0
        
        assert validate_energy(0.5) == True
        assert validate_energy(0.0) == True
        assert validate_energy(1.0) == True
        assert validate_energy(-0.1) == False
        assert validate_energy(1.5) == False
    
    def test_validate_key_range(self):
        """Test de validación de rango de key"""
        def validate_key(key):
            return 0 <= key <= 11
        
        assert validate_key(0) == True
        assert validate_key(11) == True
        assert validate_key(5) == True
        assert validate_key(-1) == False
        assert validate_key(12) == False


class TestTypeValidation:
    """Tests de validación de tipos"""
    
    def test_validate_string_type(self):
        """Test de validación de tipo string"""
        def is_string(value):
            return isinstance(value, str)
        
        assert is_string("text") == True
        assert is_string(123) == False
        assert is_string(None) == False
        assert is_string([]) == False
    
    def test_validate_dict_type(self):
        """Test de validación de tipo dict"""
        def is_dict(value):
            return isinstance(value, dict)
        
        assert is_dict({}) == True
        assert is_dict({"key": "value"}) == True
        assert is_dict([]) == False
        assert is_dict("string") == False
    
    def test_validate_list_type(self):
        """Test de validación de tipo list"""
        def is_list(value):
            return isinstance(value, list)
        
        assert is_list([]) == True
        assert is_list([1, 2, 3]) == True
        assert is_list({}) == False
        assert is_list("string") == False


class TestRequiredFields:
    """Tests de campos requeridos"""
    
    def test_check_required_fields(self):
        """Test de verificación de campos requeridos"""
        def has_required_fields(data, required):
            return all(field in data for field in required)
        
        data = {"name": "Test", "artists": ["Artist"], "album": "Album"}
        required = ["name", "artists"]
        
        assert has_required_fields(data, required) == True
        
        data_missing = {"name": "Test"}
        assert has_required_fields(data_missing, required) == False
    
    def test_validate_non_empty_fields(self):
        """Test de validación de campos no vacíos"""
        def validate_non_empty(data, fields):
            for field in fields:
                if field not in data or not data[field]:
                    return False
            return True
        
        data = {"name": "Test", "artists": ["Artist"]}
        assert validate_non_empty(data, ["name", "artists"]) == True
        
        data_empty = {"name": "", "artists": []}
        assert validate_non_empty(data_empty, ["name", "artists"]) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

