"""
Tests for Plugin Manager
========================
"""

import pytest
from ..core.plugins import PluginManager, PluginType


@pytest.fixture
def plugin_manager():
    """Create plugin manager for testing."""
    return PluginManager()


def test_register_plugin(plugin_manager):
    """Test registering a plugin."""
    def preprocess_hook(message):
        return message.upper()
    
    plugin_manager.register_plugin(
        plugin_id="test_plugin",
        plugin_type=PluginType.PRE_PROCESSING,
        hook=preprocess_hook
    )
    
    assert "test_plugin" in plugin_manager.plugins
    assert plugin_manager.plugins["test_plugin"].plugin_type == PluginType.PRE_PROCESSING


def test_execute_preprocessing(plugin_manager):
    """Test executing preprocessing plugins."""
    def upper_plugin(message):
        return message.upper()
    
    plugin_manager.register_plugin(
        "upper",
        PluginType.PRE_PROCESSING,
        upper_plugin
    )
    
    result = plugin_manager.execute_preprocessing("hello")
    assert result == "HELLO"


def test_execute_postprocessing(plugin_manager):
    """Test executing postprocessing plugins."""
    def add_suffix_plugin(response):
        return response + " [processed]"
    
    plugin_manager.register_plugin(
        "suffix",
        PluginType.POST_PROCESSING,
        add_suffix_plugin
    )
    
    result = plugin_manager.execute_postprocessing("Hello")
    assert result == "Hello [processed]"


def test_unregister_plugin(plugin_manager):
    """Test unregistering a plugin."""
    plugin_manager.register_plugin(
        "test_plugin",
        PluginType.PRE_PROCESSING,
        lambda x: x
    )
    
    assert "test_plugin" in plugin_manager.plugins
    
    plugin_manager.unregister_plugin("test_plugin")
    
    assert "test_plugin" not in plugin_manager.plugins


def test_get_plugins_by_type(plugin_manager):
    """Test getting plugins by type."""
    plugin_manager.register_plugin("pre1", PluginType.PRE_PROCESSING, lambda x: x)
    plugin_manager.register_plugin("pre2", PluginType.PRE_PROCESSING, lambda x: x)
    plugin_manager.register_plugin("post1", PluginType.POST_PROCESSING, lambda x: x)
    
    pre_plugins = plugin_manager.get_plugins_by_type(PluginType.PRE_PROCESSING)
    
    assert len(pre_plugins) == 2
    assert all(p.plugin_type == PluginType.PRE_PROCESSING for p in pre_plugins)


