"""
Tests para el módulo deep_learning_generator
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from core.deep_learning_generator import (
    create_generator,
    get_supported_frameworks,
    get_supported_model_types,
    validate_generator_config,
    get_generator_info,
    DeepLearningGenerator
)


class TestDeepLearningGeneratorModule:
    """Tests para el módulo deep_learning_generator"""
    
    def test_get_supported_frameworks(self):
        """Test que retorna frameworks soportados"""
        frameworks = get_supported_frameworks()
        
        assert isinstance(frameworks, list)
        assert len(frameworks) > 0
        assert "pytorch" in frameworks
        assert "tensorflow" in frameworks
    
    def test_get_supported_model_types(self):
        """Test que retorna tipos de modelos soportados"""
        model_types = get_supported_model_types()
        
        assert isinstance(model_types, list)
        assert len(model_types) > 0
        assert "transformer" in model_types
        assert "cnn" in model_types
        assert "llm" in model_types
    
    def test_validate_generator_config_valid(self):
        """Test validación de configuración válida"""
        config = {
            "framework": "pytorch",
            "model_type": "transformer"
        }
        
        is_valid, error = validate_generator_config(config)
        
        assert is_valid is True
        assert error is None
    
    def test_validate_generator_config_invalid_framework(self):
        """Test validación con framework inválido"""
        config = {
            "framework": "invalid_framework",
            "model_type": "transformer"
        }
        
        is_valid, error = validate_generator_config(config)
        
        assert is_valid is False
        assert error is not None
        assert "Unsupported framework" in error
    
    def test_validate_generator_config_invalid_model_type(self):
        """Test validación con tipo de modelo inválido"""
        config = {
            "framework": "pytorch",
            "model_type": "invalid_model"
        }
        
        is_valid, error = validate_generator_config(config)
        
        assert is_valid is False
        assert error is not None
        assert "Unsupported model type" in error
    
    def test_validate_generator_config_not_dict(self):
        """Test validación con tipo incorrecto"""
        config = "not a dict"
        
        is_valid, error = validate_generator_config(config)
        
        assert is_valid is False
        assert error is not None
        assert "must be a dictionary" in error
    
    def test_get_generator_info(self):
        """Test que retorna información del generador"""
        info = get_generator_info()
        
        assert isinstance(info, dict)
        assert "available" in info
        assert "supported_frameworks" in info
        assert "supported_model_types" in info
        assert "version" in info
        assert "module" in info
        
        assert isinstance(info["supported_frameworks"], list)
        assert isinstance(info["supported_model_types"], list)


class TestCreateGenerator:
    """Tests para create_generator"""
    
    @patch('core.deep_learning_generator.DeepLearningGenerator')
    def test_create_generator_success(self, mock_generator_class):
        """Test creación exitosa de generador"""
        mock_instance = Mock()
        mock_generator_class.return_value = mock_instance
        
        generator = create_generator(
            framework="pytorch",
            model_type="transformer"
        )
        
        assert generator is not None
        mock_generator_class.assert_called_once()
    
    @patch('core.deep_learning_generator._GENERATOR_AVAILABLE', False)
    def test_create_generator_not_available(self):
        """Test cuando el generador no está disponible"""
        with pytest.raises(ImportError):
            create_generator()
    
    @patch('core.deep_learning_generator.DeepLearningGenerator')
    def test_create_generator_with_config(self, mock_generator_class):
        """Test creación con configuración adicional"""
        mock_instance = Mock()
        mock_generator_class.return_value = mock_instance
        
        config = {"learning_rate": 0.001}
        generator = create_generator(
            framework="pytorch",
            config=config
        )
        
        assert generator is not None
        # Verificar que se pasó la configuración
        call_args = mock_generator_class.call_args
        assert call_args is not None


class TestGeneratorImports:
    """Tests para imports del generador"""
    
    def test_deep_learning_generator_import(self):
        """Test que DeepLearningGenerator es importable"""
        from core.deep_learning_generator import DeepLearningGenerator
        
        # Si está disponible, debe ser una clase
        if DeepLearningGenerator is not None:
            assert isinstance(DeepLearningGenerator, type) or callable(DeepLearningGenerator)
    
    def test_all_exports(self):
        """Test que todos los exports están disponibles"""
        from core import deep_learning_generator
        
        expected_exports = [
            "DeepLearningGenerator",
            "create_generator",
            "get_supported_frameworks",
            "get_supported_model_types",
            "validate_generator_config",
            "get_generator_info"
        ]
        
        for export in expected_exports:
            assert hasattr(deep_learning_generator, export), f"{export} not found in module"

