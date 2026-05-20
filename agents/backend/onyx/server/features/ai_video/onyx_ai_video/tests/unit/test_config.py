from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import pytest
import os
import tempfile
import yaml
import json
from pathlib import Path
from unittest.mock import Mock, patch
from ...config.config_manager import (
from ...config.settings import (
        import onyx_ai_video.config.settings
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Unit tests for configuration management modules.
"""


    OnyxConfigManager, OnyxAIVideoConfig, get_config_manager,
    get_config, save_config, reload_config, update_config,
    create_config_file, get_default_config
)
    OnyxSettings, OnyxEnvironmentConfig, load_onyx_settings,
    get_onyx_settings, update_onyx_settings, validate_onyx_integration,
    get_onyx_environment_info, setup_onyx_environment,
    create_onyx_env_file, get_onyx_settings_singleton
)


class TestOnyxAIVideoConfig:
    """Test OnyxAIVideoConfig model."""
    
    @pytest.mark.unit
    def test_default_config(self) -> Any:
        """Test default configuration creation."""
        config = OnyxAIVideoConfig()
        
        assert config.system_name == "Onyx AI Video System"
        assert config.version == "1.0.0"
        assert config.environment == "development"
        assert config.debug is False
        
        # Check nested configs
        assert config.logging.level == "INFO"
        assert config.llm.provider == "openai"
        assert config.video.default_quality == VideoQuality.MEDIUM
        assert config.plugins.auto_load is True
        assert config.performance.enable_monitoring is True
        assert config.security.enable_encryption is True
        assert config.onyx.use_onyx_logging is True
    
    @pytest.mark.unit
    def test_custom_config(self) -> Any:
        """Test custom configuration."""
        config = OnyxAIVideoConfig(
            system_name="Custom System",
            version="2.0.0",
            environment="production",
            debug=True,
            logging={"level": "DEBUG"},
            llm={"provider": "anthropic", "model": "claude-3"}
        )
        
        assert config.system_name == "Custom System"
        assert config.version == "2.0.0"
        assert config.environment == "production"
        assert config.debug is True
        assert config.logging.level == "DEBUG"
        assert config.llm.provider == "anthropic"
        assert config.llm.model == "claude-3"
    
    @pytest.mark.unit
    def test_environment_validation(self) -> Any:
        """Test environment validation."""
        # Valid environments
        valid_envs = ["development", "testing", "staging", "production"]
        for env in valid_envs:
            config = OnyxAIVideoConfig(environment=env)
            assert config.environment == env
        
        # Invalid environment
        with pytest.raises(ValueError, match="Environment must be one of"):
            OnyxAIVideoConfig(environment="invalid")
    
    @pytest.mark.unit
    def test_security_config_validation(self) -> Any:
        """Test security configuration validation."""
        # Test with encryption enabled but no key
        with pytest.raises(ValueError, match="Encryption key required"):
            OnyxAIVideoConfig(
                security={"enable_encryption": True, "encryption_key": None}
            )
        
        # Test with encryption key from environment
        with patch.dict(os.environ, {"AI_VIDEO_ENCRYPTION_KEY": "test-key"}):
            config = OnyxAIVideoConfig(
                security={"enable_encryption": True, "encryption_key": None}
            )
            assert config.security.encryption_key == "test-key"
    
    @pytest.mark.unit
    def test_config_serialization(self) -> Any:
        """Test config serialization."""
        config = OnyxAIVideoConfig(
            system_name="Test System",
            custom={"test_key": "test_value"}
        )
        
        config_dict = config.dict()
        
        assert config_dict["system_name"] == "Test System"
        assert config_dict["custom"]["test_key"] == "test_value"
        assert "logging" in config_dict
        assert "llm" in config_dict
        assert "video" in config_dict


class TestOnyxConfigManager:
    """Test OnyxConfigManager."""
    
    @pytest.mark.unit
    def test_initialization(self) -> Any:
        """Test config manager initialization."""
        manager = OnyxConfigManager()
        
        assert manager.config_path is None
        assert manager.config is None
        assert manager._config_cache == {}
    
    @pytest.mark.unit
    def test_load_config_from_file_yaml(self, temp_dir) -> Any:
        """Test loading config from YAML file."""
        config_data = {
            "system_name": "Test System",
            "version": "1.0.0",
            "environment": "testing",
            "logging": {"level": "DEBUG"},
            "llm": {"provider": "test"}
        }
        
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config_data, f)
        
        manager = OnyxConfigManager(str(config_file))
        config = manager.load_config()
        
        assert config.system_name == "Test System"
        assert config.version == "1.0.0"
        assert config.environment == "testing"
        assert config.logging.level == "DEBUG"
        assert config.llm.provider == "test"
    
    @pytest.mark.unit
    def test_load_config_from_file_json(self, temp_dir) -> Any:
        """Test loading config from JSON file."""
        config_data = {
            "system_name": "Test System",
            "version": "1.0.0",
            "environment": "testing",
            "logging": {"level": "DEBUG"},
            "llm": {"provider": "test"}
        }
        
        config_file = temp_dir / "test_config.json"
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(config_data, f)
        
        manager = OnyxConfigManager(str(config_file))
        config = manager.load_config()
        
        assert config.system_name == "Test System"
        assert config.version == "1.0.0"
        assert config.environment == "testing"
        assert config.logging.level == "DEBUG"
        assert config.llm.provider == "test"
    
    @pytest.mark.unit
    def test_load_config_unsupported_format(self, temp_dir) -> Any:
        """Test loading config with unsupported format."""
        config_file = temp_dir / "test_config.txt"
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("invalid config")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        manager = OnyxConfigManager(str(config_file))
        
        with pytest.raises(ConfigurationError, match="Unsupported config file format"):
            manager.load_config()
    
    @pytest.mark.unit
    def test_load_config_file_not_found(self) -> Any:
        """Test loading config when file doesn't exist."""
        manager = OnyxConfigManager("nonexistent.yaml")
        config = manager.load_config()
        
        # Should use default config
        assert config.system_name == "Onyx AI Video System"
        assert config.version == "1.0.0"
    
    @pytest.mark.unit
    def test_override_from_env(self) -> Any:
        """Test environment variable override."""
        with patch.dict(os.environ, {
            "AI_VIDEO_ENVIRONMENT": "production",
            "AI_VIDEO_DEBUG": "true",
            "AI_VIDEO_LOGGING_LEVEL": "ERROR",
            "AI_VIDEO_LLM_PROVIDER": "anthropic"
        }):
            manager = OnyxConfigManager()
            config = manager.load_config()
            
            assert config.environment == "production"
            assert config.debug is True
            assert config.logging.level == "ERROR"
            assert config.llm.provider == "anthropic"
    
    @pytest.mark.unit
    def test_validate_config(self, temp_dir) -> bool:
        """Test configuration validation."""
        config_data = {
            "video": {
                "default_duration": 30,
                "max_duration": 60
            },
            "plugins": {
                "max_workers": 5
            },
            "performance": {
                "cache_size": 100,
                "max_concurrent_requests": 5
            }
        }
        
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config_data, f)
        
        manager = OnyxConfigManager(str(config_file))
        config = manager.load_config()
        
        # Should not raise validation errors
        assert config.video.default_duration == 30
        assert config.video.max_duration == 60
        assert config.plugins.max_workers == 5
        assert config.performance.cache_size == 100
        assert config.performance.max_concurrent_requests == 5
    
    @pytest.mark.unit
    def test_validate_config_invalid(self) -> bool:
        """Test configuration validation with invalid values."""
        manager = OnyxConfigManager()
        
        # Test invalid LLM temperature
        with pytest.raises(ConfigurationError, match="LLM temperature must be between"):
            manager._validate_config(OnyxAIVideoConfig(
                llm={"temperature": 3.0}
            ))
        
        # Test invalid max tokens
        with pytest.raises(ConfigurationError, match="LLM max_tokens must be between"):
            manager._validate_config(OnyxAIVideoConfig(
                llm={"max_tokens": 50000}
            ))
        
        # Test invalid duration
        with pytest.raises(ConfigurationError, match="Default duration cannot exceed"):
            manager._validate_config(OnyxAIVideoConfig(
                video={"default_duration": 1000, "max_duration": 500}
            ))
    
    @pytest.mark.unit
    def test_create_directories(self, temp_dir) -> Any:
        """Test directory creation."""
        config_data = {
            "video": {
                "output_directory": str(temp_dir / "output"),
                "temp_directory": str(temp_dir / "temp")
            },
            "plugins": {
                "plugins_directory": str(temp_dir / "plugins")
            }
        }
        
        manager = OnyxConfigManager()
        config = OnyxAIVideoConfig(**config_data)
        
        manager._create_directories(config)
        
        assert (temp_dir / "output").exists()
        assert (temp_dir / "temp").exists()
        assert (temp_dir / "plugins").exists()
    
    @pytest.mark.unit
    def test_get_config(self, temp_dir) -> Optional[Dict[str, Any]]:
        """Test get_config method."""
        config_data = {"system_name": "Test System"}
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config_data, f)
        
        manager = OnyxConfigManager(str(config_file))
        config = manager.get_config()
        
        assert config.system_name == "Test System"
    
    @pytest.mark.unit
    def test_reload_config(self, temp_dir) -> Any:
        """Test reload_config method."""
        config_data = {"system_name": "Test System"}
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config_data, f)
        
        manager = OnyxConfigManager(str(config_file))
        config1 = manager.get_config()
        
        # Update config file
        config_data["system_name"] = "Updated System"
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config_data, f)
        
        config2 = manager.reload_config()
        
        assert config1.system_name == "Test System"
        assert config2.system_name == "Updated System"
    
    @pytest.mark.unit
    def test_get_section(self, temp_dir) -> Optional[Dict[str, Any]]:
        """Test get_section method."""
        config_data = {
            "logging": {"level": "DEBUG"},
            "llm": {"provider": "test"}
        }
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config_data, f)
        
        manager = OnyxConfigManager(str(config_file))
        
        logging_config = manager.get_section("logging")
        assert logging_config.level == "DEBUG"
        
        llm_config = manager.get_section("llm")
        assert llm_config.provider == "test"
    
    @pytest.mark.unit
    def test_update_config(self, temp_dir) -> Any:
        """Test update_config method."""
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump({"system_name": "Original"}, f)
        
        manager = OnyxConfigManager(str(config_file))
        manager.get_config()  # Load initial config
        
        updated_config = manager.update_config({
            "system_name": "Updated",
            "logging.level": "ERROR"
        })
        
        assert updated_config.system_name == "Updated"
        assert updated_config.logging.level == "ERROR"
    
    @pytest.mark.unit
    def test_save_config_yaml(self, temp_dir) -> Any:
        """Test save_config method with YAML."""
        config_file = temp_dir / "test_config.yaml"
        manager = OnyxConfigManager(str(config_file))
        
        config = OnyxAIVideoConfig(system_name="Test System")
        manager.config = config
        
        manager.save_config()
        
        assert config_file.exists()
        with open(config_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            saved_data = yaml.safe_load(f)
            assert saved_data["system_name"] == "Test System"
    
    @pytest.mark.unit
    def test_save_config_json(self, temp_dir) -> Any:
        """Test save_config method with JSON."""
        config_file = temp_dir / "test_config.json"
        manager = OnyxConfigManager(str(config_file))
        
        config = OnyxAIVideoConfig(system_name="Test System")
        manager.config = config
        
        manager.save_config()
        
        assert config_file.exists()
        with open(config_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            saved_data = json.load(f)
            assert saved_data["system_name"] == "Test System"
    
    @pytest.mark.unit
    def test_get_env_config(self, temp_dir) -> Optional[Dict[str, Any]]:
        """Test get_env_config method."""
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump({
                "system_name": "Test System",
                "logging": {"level": "DEBUG"}
            }, f)
        
        manager = OnyxConfigManager(str(config_file))
        manager.get_config()
        
        env_config = manager.get_env_config()
        
        assert env_config["AI_VIDEO_SYSTEM_NAME"] == "Test System"
        assert env_config["AI_VIDEO_LOGGING_LEVEL"] == "DEBUG"
    
    @pytest.mark.unit
    def test_validate_onyx_integration(self) -> bool:
        """Test Onyx integration validation."""
        manager = OnyxConfigManager()
        
        # Test without Onyx modules
        with patch('builtins.__import__', side_effect=ImportError):
            result = manager.validate_onyx_integration()
            assert result is False
        
        # Test with Onyx modules
        with patch('builtins.__import__', return_value=Mock()):
            result = manager.validate_onyx_integration()
            assert result is True


class TestOnyxSettings:
    """Test OnyxSettings."""
    
    @pytest.mark.unit
    def test_default_settings(self) -> Any:
        """Test default Onyx settings."""
        settings = OnyxSettings()
        
        assert settings.use_onyx_logging is True
        assert settings.use_onyx_llm is True
        assert settings.use_onyx_telemetry is True
        assert settings.use_onyx_encryption is True
        assert settings.use_onyx_threading is True
        assert settings.use_onyx_retry is True
        assert settings.use_onyx_gpu is True
        assert settings.onyx_default_llm == "gpt-4"
        assert settings.onyx_temperature == 0.7
        assert settings.onyx_max_workers == 10
    
    @pytest.mark.unit
    def test_custom_settings(self) -> Any:
        """Test custom Onyx settings."""
        settings = OnyxSettings(
            use_onyx_logging=False,
            use_onyx_llm=False,
            onyx_default_llm="claude-3",
            onyx_temperature=0.5,
            onyx_max_workers=5
        )
        
        assert settings.use_onyx_logging is False
        assert settings.use_onyx_llm is False
        assert settings.onyx_default_llm == "claude-3"
        assert settings.onyx_temperature == 0.5
        assert settings.onyx_max_workers == 5
    
    @pytest.mark.unit
    def test_find_onyx_root(self) -> Any:
        """Test Onyx root directory finding."""
        settings = OnyxSettings()
        
        # Test with environment variable
        with patch.dict(os.environ, {"ONYX_ROOT": "/test/onyx"}):
            settings = OnyxSettings()
            assert settings.onyx_root_path == "/test/onyx"
        
        # Test with ONYX_HOME
        with patch.dict(os.environ, {"ONYX_HOME": "/test/onyx_home"}):
            settings = OnyxSettings()
            assert settings.onyx_root_path == "/test/onyx_home"
    
    @pytest.mark.unit
    def test_to_dict(self) -> Any:
        """Test settings serialization."""
        settings = OnyxSettings(
            use_onyx_logging=False,
            onyx_default_llm="test-model"
        )
        
        settings_dict = settings.to_dict()
        
        assert settings_dict["use_onyx_logging"] is False
        assert settings_dict["onyx_default_llm"] == "test-model"
        assert "onyx_root_path" in settings_dict
        assert "onyx_config_path" in settings_dict


class TestOnyxEnvironmentConfig:
    """Test OnyxEnvironmentConfig."""
    
    @pytest.mark.unit
    def test_default_environment_config(self) -> Any:
        """Test default environment configuration."""
        config = OnyxEnvironmentConfig()
        
        assert config.ONYX_DEFAULT_LLM == "gpt-4"
        assert config.ONYX_VISION_LLM == "gpt-4-vision-preview"
        assert config.ONYX_TEMPERATURE == 0.7
        assert config.ONYX_MAX_TOKENS == 4000
        assert config.ONYX_MAX_WORKERS == 10
        assert config.ONYX_ENABLE_GPU is True
        assert config.ONYX_CACHE_ENABLED is True
        assert config.ONYX_VALIDATE_ACCESS is True
        assert config.ONYX_USE_LOGGING is True
        assert config.ONYX_USE_LLM is True
    
    @pytest.mark.unit
    def test_environment_override(self) -> Any:
        """Test environment variable override."""
        with patch.dict(os.environ, {
            "ONYX_DEFAULT_LLM": "claude-3",
            "ONYX_TEMPERATURE": "0.5",
            "ONYX_MAX_WORKERS": "5",
            "ONYX_ENABLE_GPU": "false"
        }):
            config = OnyxEnvironmentConfig()
            
            assert config.ONYX_DEFAULT_LLM == "claude-3"
            assert config.ONYX_TEMPERATURE == 0.5
            assert config.ONYX_MAX_WORKERS == 5
            assert config.ONYX_ENABLE_GPU is False


class TestOnyxSettingsFunctions:
    """Test Onyx settings utility functions."""
    
    @pytest.mark.unit
    def test_load_onyx_settings(self) -> Any:
        """Test load_onyx_settings function."""
        with patch.dict(os.environ, {
            "ONYX_DEFAULT_LLM": "test-model",
            "ONYX_TEMPERATURE": "0.8"
        }):
            settings = load_onyx_settings()
            
            assert settings.onyx_default_llm == "test-model"
            assert settings.onyx_temperature == 0.8
    
    @pytest.mark.unit
    def test_get_onyx_settings_singleton(self) -> Optional[Dict[str, Any]]:
        """Test get_onyx_settings singleton."""
        # Clear singleton
        onyx_ai_video.config.settings._onyx_settings = None
        
        settings1 = get_onyx_settings()
        settings2 = get_onyx_settings()
        
        assert settings1 is settings2
    
    @pytest.mark.unit
    def test_update_onyx_settings(self) -> Any:
        """Test update_onyx_settings function."""
        settings = OnyxSettings()
        
        updated_settings = update_onyx_settings({
            "use_onyx_logging": False,
            "onyx_default_llm": "updated-model"
        })
        
        assert updated_settings.use_onyx_logging is False
        assert updated_settings.onyx_default_llm == "updated-model"
    
    @pytest.mark.unit
    def test_validate_onyx_integration(self) -> bool:
        """Test validate_onyx_integration function."""
        # Test without Onyx modules
        with patch('builtins.__import__', side_effect=ImportError):
            result = validate_onyx_integration()
            assert result["onyx_available"] is False
        
        # Test with Onyx modules
        with patch('builtins.__import__', return_value=Mock()):
            result = validate_onyx_integration()
            assert result["onyx_available"] is True
    
    @pytest.mark.unit
    def test_get_onyx_environment_info(self) -> Optional[Dict[str, Any]]:
        """Test get_onyx_environment_info function."""
        with patch.dict(os.environ, {
            "ONYX_ROOT": "/test/onyx",
            "ONYX_DEFAULT_LLM": "test-model"
        }):
            info = get_onyx_environment_info()
            
            assert info["environment_variables"]["ONYX_ROOT"] == "/test/onyx"
            assert info["environment_variables"]["ONYX_DEFAULT_LLM"] == "test-model"
            assert "current_working_directory" in info
            assert "python_path" in info
            assert "sys_path" in info
    
    @pytest.mark.unit
    def test_setup_onyx_environment(self) -> Any:
        """Test setup_onyx_environment function."""
        # Clear environment
        for key in list(os.environ.keys()):
            if key.startswith("ONYX_"):
                del os.environ[key]
        
        result = setup_onyx_environment()
        assert result is True
    
    @pytest.mark.unit
    def test_create_onyx_env_file(self, temp_dir) -> Any:
        """Test create_onyx_env_file function."""
        env_file = temp_dir / ".env.onyx"
        
        create_onyx_env_file(str(env_file))
        
        assert env_file.exists()
        with open(env_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            assert "ONYX_ROOT=" in content
            assert "ONYX_DEFAULT_LLM=" in content
            assert "ONYX_USE_LOGGING=" in content


class TestConfigUtilities:
    """Test configuration utility functions."""
    
    @pytest.mark.unit
    def test_get_config_manager(self) -> Optional[Dict[str, Any]]:
        """Test get_config_manager function."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        assert manager1 is manager2
    
    @pytest.mark.unit
    def test_get_config(self, temp_dir) -> Optional[Dict[str, Any]]:
        """Test get_config function."""
        config_data = {"system_name": "Test System"}
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config_data, f)
        
        config = get_config(str(config_file))
        assert config.system_name == "Test System"
    
    @pytest.mark.unit
    def test_save_config(self, temp_dir) -> Any:
        """Test save_config function."""
        config_file = temp_dir / "test_config.yaml"
        config = OnyxAIVideoConfig(system_name="Test System")
        
        save_config(config, str(config_file))
        
        assert config_file.exists()
        with open(config_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            saved_data = yaml.safe_load(f)
            assert saved_data["system_name"] == "Test System"
    
    @pytest.mark.unit
    def test_reload_config(self, temp_dir) -> Any:
        """Test reload_config function."""
        config_data = {"system_name": "Original"}
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config_data, f)
        
        config1 = get_config(str(config_file))
        
        # Update file
        config_data["system_name"] = "Updated"
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config_data, f)
        
        config2 = reload_config(str(config_file))
        
        assert config1.system_name == "Original"
        assert config2.system_name == "Updated"
    
    @pytest.mark.unit
    def test_update_config(self, temp_dir) -> Any:
        """Test update_config function."""
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump({"system_name": "Original"}, f)
        
        updated_config = update_config({
            "system_name": "Updated",
            "logging.level": "ERROR"
        }, str(config_file))
        
        assert updated_config.system_name == "Updated"
        assert updated_config.logging.level == "ERROR"
    
    @pytest.mark.unit
    def test_get_default_config(self) -> Optional[Dict[str, Any]]:
        """Test get_default_config function."""
        config = get_default_config()
        
        assert config["system_name"] == "Onyx AI Video System"
        assert config["version"] == "1.0.0"
        assert "logging" in config
        assert "llm" in config
        assert "video" in config
        assert "plugins" in config
        assert "performance" in config
        assert "security" in config
        assert "onyx" in config
    
    @pytest.mark.unit
    def test_create_config_file(self, temp_dir) -> Any:
        """Test create_config_file function."""
        config_file = temp_dir / "test_config.yaml"
        
        create_config_file(str(config_file))
        
        assert config_file.exists()
        with open(config_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            config = yaml.safe_load(f)
            assert config["system_name"] == "Onyx AI Video System"
            assert config["version"] == "1.0.0"
        
        # Test JSON format
        json_config_file = temp_dir / "test_config.json"
        create_config_file(str(json_config_file), template={"test": "value"})
        
        assert json_config_file.exists()
        with open(json_config_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            config = json.load(f)
            assert config["test"] == "value" 