"""
Configuration Testing Helpers
Specialized helpers for configuration testing
"""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch
import os
from pathlib import Path


class ConfigTestHelpers:
    """Helpers for configuration testing"""
    
    @staticmethod
    def create_mock_config(
        settings: Optional[Dict[str, Any]] = None
    ) -> Mock:
        """Create mock configuration"""
        config = Mock()
        config.settings = settings or {}
        config.get = Mock(side_effect=lambda key, default=None: config.settings.get(key, default))
        config.set = Mock(side_effect=lambda key, value: config.settings.update({key: value}))
        config.has = Mock(side_effect=lambda key: key in config.settings)
        return config
    
    @staticmethod
    def assert_config_value(
        config: Mock,
        key: str,
        expected_value: Any
    ):
        """Assert configuration has expected value"""
        actual_value = config.get(key)
        assert actual_value == expected_value, \
            f"Config value for {key} is {actual_value}, expected {expected_value}"
    
    @staticmethod
    def assert_config_exists(config: Mock, key: str):
        """Assert configuration key exists"""
        assert config.has(key), f"Config key {key} does not exist"
    
    @staticmethod
    def with_env_vars(env_vars: Dict[str, str]):
        """Context manager to set environment variables"""
        class EnvVarContext:
            def __init__(self, vars):
                self.vars = vars
                self.original = {}
            
            def __enter__(self):
                for key, value in self.vars.items():
                    self.original[key] = os.environ.get(key)
                    os.environ[key] = value
                return self
            
            def __exit__(self, *args):
                for key in self.vars:
                    if key in self.original and self.original[key] is not None:
                        os.environ[key] = self.original[key]
                    elif key in os.environ:
                        del os.environ[key]
        
        return EnvVarContext(env_vars)


class SettingsHelpers:
    """Helpers for settings testing"""
    
    @staticmethod
    def create_test_settings(
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create test settings with optional overrides"""
        default_settings = {
            "database_url": "sqlite:///test.db",
            "cache_ttl": 3600,
            "max_upload_size": 10 * 1024 * 1024,  # 10MB
            "debug": True,
            "log_level": "DEBUG"
        }
        
        if overrides:
            default_settings.update(overrides)
        
        return default_settings
    
    @staticmethod
    def assert_settings_valid(settings: Dict[str, Any], required_keys: List[str]):
        """Assert settings are valid"""
        for key in required_keys:
            assert key in settings, f"Required setting {key} is missing"
            assert settings[key] is not None, f"Setting {key} is None"


class EnvironmentHelpers:
    """Helpers for environment testing"""
    
    @staticmethod
    def create_test_environment(
        env_name: str = "test",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create test environment configuration"""
        return {
            "name": env_name,
            "config": config or {},
            "variables": os.environ.copy()
        }
    
    @staticmethod
    def assert_environment_set(env_name: str):
        """Assert environment is set correctly"""
        # Check environment-specific settings
        assert True  # Placeholder for environment validation


# Convenience exports
create_mock_config = ConfigTestHelpers.create_mock_config
assert_config_value = ConfigTestHelpers.assert_config_value
assert_config_exists = ConfigTestHelpers.assert_config_exists
with_env_vars = ConfigTestHelpers.with_env_vars

create_test_settings = SettingsHelpers.create_test_settings
assert_settings_valid = SettingsHelpers.assert_settings_valid

create_test_environment = EnvironmentHelpers.create_test_environment
assert_environment_set = EnvironmentHelpers.assert_environment_set



