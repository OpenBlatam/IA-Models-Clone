"""
Ejemplos de tests para referencia
"""

import pytest
from unittest.mock import Mock, patch
from test_helpers import TestHelpers, MockDataFactory


class TestExamples:
    """Ejemplos de diferentes tipos de tests"""
    
    def test_example_basic(self):
        """Ejemplo de test básico"""
        # Arrange
        value = 5
        
        # Act
        result = value * 2
        
        # Assert
        assert result == 10
    
    def test_example_with_fixture(self, sample_audio_features):
        """Ejemplo usando fixture"""
        assert sample_audio_features["tempo"] == 120.0
        assert sample_audio_features["key"] == 0
    
    def test_example_with_helpers(self, test_helpers):
        """Ejemplo usando helpers"""
        mock_response = test_helpers.create_mock_response(
            status_code=200,
            data={"success": True}
        )
        
        assert mock_response.status_code == 200
        assert mock_response.json()["success"] == True
    
    def test_example_with_mock_factory(self, mock_data_factory):
        """Ejemplo usando mock factory"""
        user = mock_data_factory.create_user("user123", "test@example.com")
        
        assert user["id"] == "user123"
        assert user["email"] == "test@example.com"
    
    def test_example_with_patch(self):
        """Ejemplo usando patch"""
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = "test content"
            
            # Simular lectura de archivo
            with open("test.txt") as f:
                content = f.read()
            
            assert content == "test content"
            mock_open.assert_called_once_with("test.txt")
    
    @pytest.mark.parametrize("input_value,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
        (10, 20),
    ])
    def test_example_parametrized(self, input_value, expected):
        """Ejemplo de test parametrizado"""
        result = input_value * 2
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_example_async(self):
        """Ejemplo de test asíncrono"""
        async def async_function():
            return "async result"
        
        result = await async_function()
        assert result == "async result"
    
    def test_example_with_assertions(self, test_helpers):
        """Ejemplo con assertions personalizados"""
        actual = {
            "id": "123",
            "name": "Test",
            "value": 100
        }
        
        expected = {
            "id": "123",
            "name": "Test"
        }
        
        # Verificar que actual contiene expected
        test_helpers.assert_dict_contains(actual, expected)
    
    def test_example_with_exception(self):
        """Ejemplo de test que verifica excepciones"""
        with pytest.raises(ValueError, match="Invalid value"):
            raise ValueError("Invalid value")
    
    def test_example_with_mock_side_effect(self):
        """Ejemplo con side_effect en mock"""
        mock_func = Mock(side_effect=[1, 2, 3])
        
        assert mock_func() == 1
        assert mock_func() == 2
        assert mock_func() == 3
    
    def test_example_with_context_manager(self):
        """Ejemplo con context manager"""
        class TestContext:
            def __enter__(self):
                return self
            
            def __exit__(self, *args):
                pass
        
        with TestContext() as ctx:
            assert ctx is not None


class TestIntegrationExamples:
    """Ejemplos de tests de integración"""
    
    def test_example_integration_flow(self, test_helpers):
        """Ejemplo de flujo de integración"""
        # Simular flujo completo
        track = test_helpers.create_mock_spotify_track()
        features = test_helpers.create_mock_audio_features()
        analysis = test_helpers.create_mock_analysis_result()
        
        # Verificar que todos los datos están presentes
        assert track["id"] is not None
        assert features["tempo"] > 0
        assert "track_basic_info" in analysis


class TestPerformanceExamples:
    """Ejemplos de tests de performance"""
    
    def test_example_performance_check(self):
        """Ejemplo de verificación de performance"""
        import time
        
        start = time.time()
        # Simular operación
        time.sleep(0.01)
        duration = time.time() - start
        
        # Verificar que la operación es rápida
        assert duration < 0.1  # Debe completarse en menos de 100ms


class TestSecurityExamples:
    """Ejemplos de tests de seguridad"""
    
    def test_example_input_validation(self):
        """Ejemplo de validación de entrada"""
        def process_input(user_input):
            if not isinstance(user_input, str):
                raise ValueError("Input must be a string")
            if len(user_input) > 1000:
                raise ValueError("Input too long")
            return user_input.strip()
        
        # Test válido
        result = process_input("valid input")
        assert result == "valid input"
        
        # Test inválido - tipo incorrecto
        with pytest.raises(ValueError, match="must be a string"):
            process_input(123)
        
        # Test inválido - muy largo
        with pytest.raises(ValueError, match="too long"):
            process_input("x" * 1001)

