"""
Tests for the Blaze AI Plugin System.

This module contains comprehensive tests for the plugin system including
PluginConfig, PluginMetadata, PluginInfo, PluginLoader, and PluginManager classes.
"""

import unittest
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
from engines.plugins import (
    PluginConfig,
    PluginMetadata,
    PluginInfo,
    PluginLoader,
    PluginManager,
    create_plugin_manager,
    create_standard_plugin_manager,
    get_plugin_manager
)
from engines.base import EnginePriority


class TestPluginConfig(unittest.TestCase):
    """Test cases for PluginConfig class."""
    
    def test_default_config(self):
        """Test PluginConfig with default values."""
        config = PluginConfig()
        self.assertEqual(config.plugin_directories, ["plugins", "extensions", "custom_engines"])
        self.assertFalse(config.enable_hot_reload)
        self.assertEqual(config.hot_reload_interval, 30.0)
        self.assertTrue(config.enable_plugin_validation)
        self.assertFalse(config.allow_unsafe_plugins)
        self.assertEqual(config.plugin_cache_size, 100)
        self.assertTrue(config.enable_plugin_metrics)
        self.assertEqual(config.plugin_timeout, 30.0)
    
    def test_custom_config(self):
        """Test PluginConfig with custom values."""
        custom_dirs = ["custom_plugins", "my_extensions"]
        config = PluginConfig(
            plugin_directories=custom_dirs,
            enable_hot_reload=True,
            hot_reload_interval=60.0,
            enable_plugin_validation=False,
            allow_unsafe_plugins=True,
            plugin_cache_size=200,
            enable_plugin_metrics=False,
            plugin_timeout=60.0
        )
        
        self.assertEqual(config.plugin_directories, custom_dirs)
        self.assertTrue(config.enable_hot_reload)
        self.assertEqual(config.hot_reload_interval, 60.0)
        self.assertFalse(config.enable_plugin_validation)
        self.assertTrue(config.allow_unsafe_plugins)
        self.assertEqual(config.plugin_cache_size, 200)
        self.assertFalse(config.enable_plugin_metrics)
        self.assertEqual(config.plugin_timeout, 60.0)


class TestPluginMetadata(unittest.TestCase):
    """Test cases for PluginMetadata class."""
    
    def test_required_fields(self):
        """Test PluginMetadata with required fields only."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0"
        )
        
        self.assertEqual(metadata.name, "test_plugin")
        self.assertEqual(metadata.version, "1.0.0")
        self.assertEqual(metadata.description, "")
        self.assertEqual(metadata.author, "")
        self.assertEqual(metadata.license, "")
        self.assertEqual(metadata.homepage, "")
        self.assertEqual(metadata.repository, "")
        self.assertEqual(metadata.tags, [])
        self.assertEqual(metadata.dependencies, [])
        self.assertEqual(metadata.requirements, {})
        self.assertEqual(metadata.engine_types, [])
        self.assertEqual(metadata.priority, EnginePriority.NORMAL)
        self.assertGreater(metadata.created_at, 0)
        self.assertEqual(metadata.updated_at, metadata.created_at)
    
    def test_all_fields(self):
        """Test PluginMetadata with all fields."""
        metadata = PluginMetadata(
            name="full_plugin",
            version="2.1.0",
            description="A comprehensive test plugin",
            author="Test Author",
            license="MIT",
            homepage="https://example.com",
            repository="https://github.com/example/plugin",
            tags=["test", "demo"],
            dependencies=["numpy", "pandas"],
            requirements={"python": ">=3.8"},
            engine_types=["llm", "diffusion"],
            priority=EnginePriority.HIGH
        )
        
        self.assertEqual(metadata.name, "full_plugin")
        self.assertEqual(metadata.version, "2.1.0")
        self.assertEqual(metadata.description, "A comprehensive test plugin")
        self.assertEqual(metadata.author, "Test Author")
        self.assertEqual(metadata.license, "MIT")
        self.assertEqual(metadata.homepage, "https://example.com")
        self.assertEqual(metadata.repository, "https://github.com/example/plugin")
        self.assertEqual(metadata.tags, ["test", "demo"])
        self.assertEqual(metadata.dependencies, ["numpy", "pandas"])
        self.assertEqual(metadata.requirements, {"python": ">=3.8"})
        self.assertEqual(metadata.engine_types, ["llm", "diffusion"])
        self.assertEqual(metadata.priority, EnginePriority.HIGH)


class TestPluginInfo(unittest.TestCase):
    """Test cases for PluginInfo class."""
    
    def test_plugin_info_creation(self):
        """Test PluginInfo creation."""
        metadata = PluginMetadata(name="test", version="1.0.0")
        plugin_path = Path("/tmp/test_plugin")
        
        plugin_info = PluginInfo(
            metadata=metadata,
            plugin_path=plugin_path
        )
        
        self.assertEqual(plugin_info.metadata, metadata)
        self.assertEqual(plugin_info.plugin_path, plugin_path)
        self.assertFalse(plugin_info.is_loaded)
        self.assertEqual(plugin_info.load_time, 0.0)
        self.assertEqual(plugin_info.error_count, 0)
        self.assertIsNone(plugin_info.last_error)
        self.assertEqual(plugin_info.engine_templates, [])


class TestPluginLoader(unittest.TestCase):
    """Test cases for PluginLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.plugin_dir = Path(self.temp_dir) / "plugins"
        self.plugin_dir.mkdir()
        
        self.config = PluginConfig(
            plugin_directories=[str(self.plugin_dir)]
        )
        
        # Mock the logger to avoid logging issues during tests
        with patch('engines.plugins.get_logger') as mock_logger:
            mock_logger.return_value = Mock()
            self.loader = PluginLoader(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test PluginLoader initialization."""
        self.assertEqual(self.loader.config, self.config)
        self.assertEqual(self.loader.plugins, {})
        self.assertEqual(self.loader.plugin_cache, {})
        self.assertEqual(self.loader._plugin_watchers, [])
    
    def test_scan_plugins_empty_directory(self):
        """Test scanning an empty plugin directory."""
        plugins = self.loader.list_plugins()
        self.assertEqual(plugins, [])
    
    def test_scan_plugins_with_python_file(self):
        """Test scanning a directory with a Python plugin file."""
        # Create a simple Python plugin file
        plugin_file = self.plugin_dir / "test_plugin.py"
        plugin_content = '''"""
# version: 1.0.0
# author: Test Author
# description: A test plugin
"""
'''
        plugin_file.write_text(plugin_content)
        
        # Re-scan plugins
        self.loader._scan_plugins()
        
        plugins = self.loader.list_plugins()
        self.assertIn("test_plugin", plugins)
    
    def test_scan_plugins_with_plugin_json(self):
        """Test scanning a directory with a plugin.json file."""
        # Create a plugin directory with plugin.json
        plugin_dir = self.plugin_dir / "json_plugin"
        plugin_dir.mkdir()
        
        plugin_json = plugin_dir / "plugin.json"
        plugin_data = {
            "name": "json_plugin",
            "version": "1.0.0",
            "description": "A JSON-based plugin",
            "author": "Test Author"
        }
        plugin_json.write_text(json.dumps(plugin_data))
        
        # Re-scan plugins
        self.loader._scan_plugins()
        
        plugins = self.loader.list_plugins()
        self.assertIn("json_plugin", plugins)
    
    def test_get_plugin_info(self):
        """Test getting plugin information."""
        # Create a plugin first
        plugin_file = self.plugin_dir / "test_plugin.py"
        plugin_file.write_text('"""# version: 1.0.0"""')
        
        self.loader._scan_plugins()
        
        plugin_info = self.loader.get_plugin_info("test_plugin")
        self.assertIsNotNone(plugin_info)
        self.assertEqual(plugin_info.metadata.name, "test_plugin")
    
    def test_get_plugin_info_nonexistent(self):
        """Test getting information for a non-existent plugin."""
        plugin_info = self.loader.get_plugin_info("nonexistent")
        self.assertIsNone(plugin_info)
    
    def test_reload_plugin(self):
        """Test reloading a plugin."""
        # Create a plugin first
        plugin_file = self.plugin_dir / "test_plugin.py"
        plugin_file.write_text('"""# version: 1.0.0"""')
        
        self.loader._scan_plugins()
        self.assertIn("test_plugin", self.loader.list_plugins())
        
        # Reload the plugin
        success = self.loader.reload_plugin("test_plugin")
        self.assertTrue(success)
        self.assertIn("test_plugin", self.loader.list_plugins())
    
    def test_unload_plugin(self):
        """Test unloading a plugin."""
        # Create a plugin first
        plugin_file = self.plugin_dir / "test_plugin.py"
        plugin_file.write_text('"""# version: 1.0.0"""')
        
        self.loader._scan_plugins()
        self.assertIn("test_plugin", self.loader.list_plugins())
        
        # Unload the plugin
        success = self.loader.unload_plugin("test_plugin")
        self.assertTrue(success)
        self.assertNotIn("test_plugin", self.loader.list_plugins())
    
    def test_get_plugin_metrics(self):
        """Test getting plugin metrics."""
        metrics = self.loader.get_plugin_metrics()
        
        self.assertIn("total_plugins", metrics)
        self.assertIn("loaded_plugins", metrics)
        self.assertIn("failed_plugins", metrics)
        self.assertIn("total_engines", metrics)
        self.assertIn("plugins", metrics)
        
        self.assertEqual(metrics["total_plugins"], 0)
        self.assertEqual(metrics["loaded_plugins"], 0)
        self.assertEqual(metrics["failed_plugins"], 0)
        self.assertEqual(metrics["total_engines"], 0)


class TestPluginManager(unittest.TestCase):
    """Test cases for PluginManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.plugin_dir = Path(self.temp_dir) / "plugins"
        self.plugin_dir.mkdir()
        
        self.config = PluginConfig(
            plugin_directories=[str(self.plugin_dir)]
        )
        
        # Mock the logger to avoid logging issues during tests
        with patch('engines.plugins.get_logger') as mock_logger:
            mock_logger.return_value = Mock()
            self.manager = PluginManager(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test PluginManager initialization."""
        self.assertEqual(self.manager.config, self.config)
        self.assertIsNotNone(self.manager.loader)
        self.assertEqual(self.manager._watchers, [])
    
    def test_register_unregister_watcher(self):
        """Test registering and unregistering plugin watchers."""
        mock_watcher = Mock()
        
        # Register watcher
        self.manager.register_plugin_watcher(mock_watcher)
        self.assertIn(mock_watcher, self.manager._watchers)
        
        # Unregister watcher
        self.manager.unregister_plugin_watcher(mock_watcher)
        self.assertNotIn(mock_watcher, self.manager._watchers)
    
    def test_install_plugin(self):
        """Test installing a plugin."""
        # Create a source plugin file
        source_plugin = self.temp_dir / "source_plugin.py"
        source_plugin.write_text('"""# version: 1.0.0"""')
        
        # Install the plugin
        success = self.manager.install_plugin(str(source_plugin))
        self.assertTrue(success)
        
        # Check if plugin was installed
        target_plugin = self.plugin_dir / "source_plugin.py"
        self.assertTrue(target_plugin.exists())
    
    def test_remove_plugin(self):
        """Test removing a plugin."""
        # Create a plugin first
        plugin_file = self.plugin_dir / "test_plugin.py"
        plugin_file.write_text('"""# version: 1.0.0"""')
        
        # Load the plugin
        self.manager.loader._scan_plugins()
        self.assertIn("test_plugin", self.manager.loader.list_plugins())
        
        # Remove the plugin
        success = self.manager.remove_plugin("test_plugin")
        self.assertTrue(success)
        
        # Check if plugin was removed
        self.assertNotIn("test_plugin", self.manager.loader.list_plugins())
        self.assertFalse(plugin_file.exists())
    
    def test_get_all_engine_templates(self):
        """Test getting all engine templates."""
        templates = self.manager.get_all_engine_templates()
        self.assertEqual(templates, [])
    
    def test_search_plugins(self):
        """Test searching plugins."""
        # Create a plugin with specific content
        plugin_file = self.plugin_dir / "demo_plugin.py"
        plugin_file.write_text('"""# description: A demo plugin for testing"""')
        
        # Load the plugin
        self.manager.loader._scan_plugins()
        
        # Search for plugins
        results = self.manager.search_plugins("demo")
        self.assertIn("demo_plugin", results)
        
        # Search for non-existent content
        results = self.manager.search_plugins("nonexistent")
        self.assertEqual(results, [])


class TestFactoryFunctions(unittest.TestCase):
    """Test cases for factory functions."""
    
    def test_create_plugin_manager(self):
        """Test create_plugin_manager function."""
        config = PluginConfig(enable_hot_reload=True)
        manager = create_plugin_manager(config)
        
        self.assertIsInstance(manager, PluginManager)
        self.assertEqual(manager.config, config)
    
    def test_create_standard_plugin_manager(self):
        """Test create_standard_plugin_manager function."""
        manager = create_standard_plugin_manager()
        
        self.assertIsInstance(manager, PluginManager)
        self.assertTrue(manager.config.enable_hot_reload)
        self.assertTrue(manager.config.enable_plugin_validation)
        self.assertFalse(manager.config.allow_unsafe_plugins)
    
    def test_get_plugin_manager(self):
        """Test get_plugin_manager function."""
        # Test that it returns a singleton instance
        manager1 = get_plugin_manager()
        manager2 = get_plugin_manager()
        
        self.assertIsInstance(manager1, PluginManager)
        self.assertIs(manager1, manager2)


if __name__ == '__main__':
    unittest.main()
