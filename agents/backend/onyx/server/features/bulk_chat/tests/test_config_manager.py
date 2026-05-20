"""
Tests for Config Manager
=========================
"""

import pytest
import asyncio
from ..core.config_manager import ConfigManager


@pytest.fixture
def config_manager():
    """Create config manager for testing."""
    return ConfigManager()


def test_set_config(config_manager):
    """Test setting a configuration."""
    config_manager.set_config(
        key="test_config",
        value="test_value",
        version=1
    )
    
    assert "test_config" in config_manager.configs


def test_get_config(config_manager):
    """Test getting a configuration."""
    config_manager.set_config("test_config", "test_value", version=1)
    
    value = config_manager.get_config("test_config")
    
    assert value == "test_value" or value.get("value") == "test_value"


def test_get_config_version(config_manager):
    """Test getting specific config version."""
    config_manager.set_config("test_config", "value1", version=1)
    config_manager.set_config("test_config", "value2", version=2)
    
    value_v1 = config_manager.get_config("test_config", version=1)
    value_v2 = config_manager.get_config("test_config", version=2)
    
    assert value_v1 is not None
    assert value_v2 is not None


def test_delete_config(config_manager):
    """Test deleting a configuration."""
    config_manager.set_config("test_config", "test_value")
    
    assert "test_config" in config_manager.configs
    
    config_manager.delete_config("test_config")
    
    assert "test_config" not in config_manager.configs


def test_hot_reload(config_manager):
    """Test hot reload of configurations."""
    config_manager.set_config("test_config", "value1")
    
    # Enable hot reload
    config_manager.enable_hot_reload()
    
    # Update config
    config_manager.set_config("test_config", "value2")
    
    # Should reflect new value
    value = config_manager.get_config("test_config")
    assert value == "value2" or value.get("value") == "value2"


def test_get_config_manager_summary(config_manager):
    """Test getting config manager summary."""
    config_manager.set_config("config1", "value1")
    config_manager.set_config("config2", "value2")
    
    summary = config_manager.get_config_manager_summary()
    
    assert summary is not None
    assert "total_configs" in summary or "configs" in summary


