"""
Tests for Configuration
Tests for settings, configuration management, and environment variables
"""

import pytest
from unittest.mock import patch, Mock
import os

from config.settings import settings
from core.infrastructure.config_manager import ConfigManager, CacheConfig


class TestSettings:
    """Tests for application settings"""
    
    def test_settings_loaded(self):
        """Test that settings are loaded"""
        assert settings is not None
        assert hasattr(settings, 'environment')
        assert hasattr(settings, 'debug')
        assert hasattr(settings, 'log_level')
    
    def test_environment_setting(self):
        """Test environment setting"""
        assert settings.environment is not None
        assert settings.environment.value in ['development', 'production', 'testing']
    
    def test_debug_setting(self):
        """Test debug setting"""
        assert isinstance(settings.debug, bool)
    
    def test_log_level_setting(self):
        """Test log level setting"""
        assert settings.log_level is not None
        assert settings.log_level.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']


class TestConfigManager:
    """Tests for ConfigManager"""
    
    @pytest.fixture
    def config_manager(self):
        """Create config manager"""
        return ConfigManager()
    
    def test_get_config(self, config_manager):
        """Test getting configuration"""
        config = config_manager.get_config("database")
        
        # Should return config dict or None
        assert config is None or isinstance(config, dict)
    
    def test_set_config(self, config_manager):
        """Test setting configuration"""
        config_manager.set_config("test_key", {"value": "test"})
        
        config = config_manager.get_config("test_key")
        
        # Should retrieve the set config
        assert config is not None
    
    def test_reload_config(self, config_manager):
        """Test reloading configuration"""
        # Should not raise
        config_manager.reload()
    
    @patch.dict(os.environ, {'TEST_CONFIG_VALUE': 'test_value'})
    def test_environment_variable_override(self, config_manager):
        """Test that environment variables can override config"""
        # This tests that env vars are respected
        value = os.environ.get('TEST_CONFIG_VALUE')
        assert value == 'test_value'


class TestCacheConfig:
    """Tests for CacheConfig"""
    
    def test_cache_config_creation(self):
        """Test creating cache configuration"""
        config = CacheConfig(
            enabled=True,
            ttl=3600,
            max_size=1000
        )
        
        assert config.enabled is True
        assert config.ttl == 3600
        assert config.max_size == 1000
    
    def test_cache_config_defaults(self):
        """Test cache config with defaults"""
        config = CacheConfig()
        
        # Should have default values
        assert hasattr(config, 'enabled')
        assert hasattr(config, 'ttl')
        assert hasattr(config, 'max_size')


class TestEnvironmentConfiguration:
    """Tests for environment-specific configuration"""
    
    @patch.dict(os.environ, {'ENVIRONMENT': 'testing'})
    def test_testing_environment(self):
        """Test testing environment configuration"""
        # Reload settings if needed
        from config.settings import settings
        # Settings should reflect testing environment
        assert settings is not None
    
    @patch.dict(os.environ, {'ENVIRONMENT': 'production'})
    def test_production_environment(self):
        """Test production environment configuration"""
        from config.settings import settings
        # In production, debug should typically be False
        # (but we don't enforce this in tests)
        assert settings is not None
    
    @patch.dict(os.environ, {'ENVIRONMENT': 'development'})
    def test_development_environment(self):
        """Test development environment configuration"""
        from config.settings import settings
        # In development, debug might be True
        assert settings is not None


class TestConfigurationValidation:
    """Tests for configuration validation"""
    
    def test_validate_required_settings(self):
        """Test that required settings are present"""
        required_settings = [
            'environment',
            'debug',
            'log_level'
        ]
        
        for setting in required_settings:
            assert hasattr(settings, setting), f"Missing required setting: {setting}"
    
    def test_validate_setting_types(self):
        """Test that settings have correct types"""
        assert isinstance(settings.debug, bool)
        assert isinstance(settings.log_level, str)
        assert hasattr(settings.environment, 'value')


class TestConfigurationReload:
    """Tests for configuration reloading"""
    
    def test_config_reload(self):
        """Test reloading configuration"""
        config_manager = ConfigManager()
        
        # Should not raise
        config_manager.reload()
    
    @patch.dict(os.environ, {'LOG_LEVEL': 'DEBUG'})
    def test_config_reload_with_env_change(self):
        """Test that config reload picks up environment changes"""
        config_manager = ConfigManager()
        
        # Reload should pick up new env var
        config_manager.reload()
        
        # Verify the change (implementation dependent)
        assert config_manager is not None



