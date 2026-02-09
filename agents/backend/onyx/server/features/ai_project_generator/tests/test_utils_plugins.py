"""
Tests for PluginSystem utility
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from ..utils.plugin_system import PluginSystem


class TestPluginSystem:
    """Test suite for PluginSystem"""

    def test_init(self, temp_dir):
        """Test PluginSystem initialization"""
        plugins_dir = temp_dir / "plugins"
        system = PluginSystem(plugins_dir=plugins_dir)
        assert system.plugins_dir == plugins_dir
        assert plugins_dir.exists()
        assert system.plugins == {}
        assert system.hooks == {}

    def test_init_default_dir(self):
        """Test PluginSystem with default directory"""
        system = PluginSystem()
        assert system.plugins_dir.exists()

    def test_register_plugin(self):
        """Test registering a plugin"""
        system = PluginSystem()
        
        # Mock module
        mock_module = Mock()
        mock_module.test_hook = Mock()
        
        with patch('importlib.import_module', return_value=mock_module):
            result = system.register_plugin(
                plugin_name="test_plugin",
                plugin_module="test_module",
                hooks={"before_generate": "test_hook"}
            )
            
            assert result is True
            assert "test_plugin" in system.plugins
            assert system.plugins["test_plugin"]["active"] is True

    def test_register_plugin_without_hooks(self):
        """Test registering plugin without hooks"""
        system = PluginSystem()
        
        mock_module = Mock()
        
        with patch('importlib.import_module', return_value=mock_module):
            result = system.register_plugin(
                plugin_name="simple_plugin",
                plugin_module="simple_module"
            )
            
            assert result is True
            assert system.plugins["simple_plugin"]["hooks"] == {}

    def test_register_plugin_error(self):
        """Test registering plugin with error"""
        system = PluginSystem()
        
        with patch('importlib.import_module', side_effect=ImportError("Module not found")):
            result = system.register_plugin(
                plugin_name="error_plugin",
                plugin_module="nonexistent_module"
            )
            
            assert result is False

    @pytest.mark.asyncio
    async def test_trigger_hook(self):
        """Test triggering a hook"""
        system = PluginSystem()
        
        hook_func = AsyncMock(return_value="hook_result")
        system.hooks["test_hook"] = [hook_func]
        
        results = await system.trigger_hook("test_hook", arg1="value1")
        
        assert len(results) == 1
        assert results[0] == "hook_result"
        hook_func.assert_called_once_with(arg1="value1")

    @pytest.mark.asyncio
    async def test_trigger_hook_multiple(self):
        """Test triggering hook with multiple handlers"""
        system = PluginSystem()
        
        hook1 = AsyncMock(return_value="result1")
        hook2 = AsyncMock(return_value="result2")
        system.hooks["multi_hook"] = [hook1, hook2]
        
        results = await system.trigger_hook("multi_hook")
        
        assert len(results) == 2
        assert "result1" in results
        assert "result2" in results

    @pytest.mark.asyncio
    async def test_trigger_hook_not_found(self):
        """Test triggering non-existent hook"""
        system = PluginSystem()
        
        results = await system.trigger_hook("nonexistent_hook")
        
        assert results == []

    @pytest.mark.asyncio
    async def test_trigger_hook_error(self):
        """Test hook error handling"""
        system = PluginSystem()
        
        def failing_hook():
            raise Exception("Hook error")
        
        system.hooks["error_hook"] = [failing_hook]
        
        # Should handle error gracefully
        results = await system.trigger_hook("error_hook")
        # Should not crash

    def test_list_plugins(self):
        """Test listing plugins"""
        system = PluginSystem()
        
        mock_module = Mock()
        with patch('importlib.import_module', return_value=mock_module):
            system.register_plugin("plugin1", "module1")
            system.register_plugin("plugin2", "module2")
        
        plugins = system.list_plugins()
        
        assert len(plugins) == 2
        plugin_names = [p["name"] for p in plugins]
        assert "plugin1" in plugin_names
        assert "plugin2" in plugin_names

    def test_enable_plugin(self):
        """Test enabling a plugin"""
        system = PluginSystem()
        
        mock_module = Mock()
        with patch('importlib.import_module', return_value=mock_module):
            system.register_plugin("test_plugin", "test_module")
            system.plugins["test_plugin"]["active"] = False
        
        system.enable_plugin("test_plugin")
        
        assert system.plugins["test_plugin"]["active"] is True

    def test_disable_plugin(self):
        """Test disabling a plugin"""
        system = PluginSystem()
        
        mock_module = Mock()
        with patch('importlib.import_module', return_value=mock_module):
            system.register_plugin("test_plugin", "test_module")
        
        system.disable_plugin("test_plugin")
        
        assert system.plugins["test_plugin"]["active"] is False

