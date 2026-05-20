"""
Tests de configuración
"""

import pytest
from unittest.mock import Mock, patch
import os


class TestConfiguration:
    """Tests de configuración"""
    
    def test_load_from_environment(self):
        """Test de carga desde variables de entorno"""
        def load_config():
            return {
                "api_key": os.getenv("SPOTIFY_API_KEY", ""),
                "api_secret": os.getenv("SPOTIFY_API_SECRET", ""),
                "debug": os.getenv("DEBUG", "false").lower() == "true"
            }
        
        with patch.dict(os.environ, {
            "SPOTIFY_API_KEY": "test_key",
            "SPOTIFY_API_SECRET": "test_secret",
            "DEBUG": "true"
        }):
            config = load_config()
            
            assert config["api_key"] == "test_key"
            assert config["api_secret"] == "test_secret"
            assert config["debug"] == True
    
    def test_configuration_validation(self):
        """Test de validación de configuración"""
        def validate_config(config):
            errors = []
            required = ["api_key", "api_secret"]
            
            for field in required:
                if field not in config or not config[field]:
                    errors.append(f"Missing required config: {field}")
            
            if "timeout" in config and config["timeout"] < 0:
                errors.append("Timeout must be non-negative")
            
            return {"valid": len(errors) == 0, "errors": errors}
        
        valid_config = {
            "api_key": "key",
            "api_secret": "secret",
            "timeout": 30
        }
        result = validate_config(valid_config)
        assert result["valid"] == True
        
        invalid_config = {"api_key": "key"}  # Falta api_secret
        result = validate_config(invalid_config)
        assert result["valid"] == False
    
    def test_configuration_defaults(self):
        """Test de valores por defecto de configuración"""
        def get_config_with_defaults(config):
            defaults = {
                "timeout": 30,
                "retry_count": 3,
                "cache_ttl": 3600,
                "max_results": 50
            }
            
            return {**defaults, **config}
        
        partial_config = {"timeout": 60}
        full_config = get_config_with_defaults(partial_config)
        
        assert full_config["timeout"] == 60  # Sobrescrito
        assert full_config["retry_count"] == 3  # Default
        assert full_config["cache_ttl"] == 3600  # Default


class TestConfigurationFiles:
    """Tests de archivos de configuración"""
    
    def test_load_from_json(self):
        """Test de carga desde JSON"""
        import json
        
        def load_json_config(json_string):
            return json.loads(json_string)
        
        config_json = '{"api_key": "test", "timeout": 30}'
        config = load_json_config(config_json)
        
        assert config["api_key"] == "test"
        assert config["timeout"] == 30
    
    def test_load_from_yaml(self):
        """Test de carga desde YAML"""
        def load_yaml_config(yaml_string):
            # Simulación simple de YAML
            config = {}
            for line in yaml_string.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    config[key.strip()] = value.strip().strip('"')
            return config
        
        yaml_config = """
        api_key: "test"
        timeout: 30
        """
        
        config = load_yaml_config(yaml_config)
        
        assert config["api_key"] == "test"
        assert config["timeout"] == "30"


class TestConfigurationSecrets:
    """Tests de configuración de secretos"""
    
    def test_secret_masking(self):
        """Test de enmascaramiento de secretos"""
        def mask_secrets(config):
            masked = config.copy()
            secret_fields = ["api_key", "api_secret", "password", "token"]
            
            for field in secret_fields:
                if field in masked:
                    value = masked[field]
                    if value:
                        masked[field] = "*" * min(len(value), 8)
            
            return masked
        
        config = {
            "api_key": "secret_key_12345",
            "api_secret": "secret_secret",
            "timeout": 30
        }
        
        masked = mask_secrets(config)
        
        assert masked["api_key"] == "********"
        assert masked["api_secret"] == "********"
        assert masked["timeout"] == 30  # No es secreto
    
    def test_secret_validation(self):
        """Test de validación de secretos"""
        def validate_secret(secret, min_length=8):
            if not secret:
                return {"valid": False, "error": "Secret cannot be empty"}
            
            if len(secret) < min_length:
                return {"valid": False, "error": f"Secret must be at least {min_length} characters"}
            
            return {"valid": True}
        
        result1 = validate_secret("short")
        assert result1["valid"] == False
        
        result2 = validate_secret("long_secret_key")
        assert result2["valid"] == True


class TestConfigurationReload:
    """Tests de recarga de configuración"""
    
    def test_hot_reload(self):
        """Test de recarga en caliente"""
        config = {"timeout": 30}
        reload_count = [0]
        
        def reload_config():
            reload_count[0] += 1
            # Simular recarga
            return {"timeout": 30, "reloaded": True}
        
        new_config = reload_config()
        
        assert new_config["reloaded"] == True
        assert reload_count[0] == 1
    
    def test_configuration_change_detection(self):
        """Test de detección de cambios en configuración"""
        def detect_changes(old_config, new_config):
            changes = {}
            
            for key in set(old_config.keys()) | set(new_config.keys()):
                old_value = old_config.get(key)
                new_value = new_config.get(key)
                
                if old_value != new_value:
                    changes[key] = {
                        "old": old_value,
                        "new": new_value
                    }
            
            return changes
        
        old_config = {"timeout": 30, "retry_count": 3}
        new_config = {"timeout": 60, "retry_count": 3, "cache_ttl": 3600}
        
        changes = detect_changes(old_config, new_config)
        
        assert "timeout" in changes
        assert "cache_ttl" in changes
        assert "retry_count" not in changes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

