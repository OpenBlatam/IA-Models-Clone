"""
Tests for Plugin System
Tests for plugin registration, loading, and execution
"""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from core.plugin_system import (
    BasePlugin,
    PluginRegistry,
    PluginType,
    PluginMetadata
)


class TestBasePlugin:
    """Tests for BasePlugin"""
    
    def test_create_plugin(self):
        """Test creating a plugin"""
        class TestPlugin(BasePlugin):
            def __init__(self):
                super().__init__(
                    name="test_plugin",
                    version="1.0.0",
                    plugin_type=PluginType.PROCESSOR
                )
        
        plugin = TestPlugin()
        
        assert plugin.name == "test_plugin"
        assert plugin.version == "1.0.0"
        assert plugin.plugin_type == PluginType.PROCESSOR
    
    @pytest.mark.asyncio
    async def test_plugin_initialize(self):
        """Test plugin initialization"""
        class TestPlugin(BasePlugin):
            def __init__(self):
                super().__init__(
                    name="test_plugin",
                    version="1.0.0",
                    plugin_type=PluginType.PROCESSOR
                )
                self.initialized = False
            
            async def initialize(self):
                self.initialized = True
        
        plugin = TestPlugin()
        await plugin.initialize()
        
        assert plugin.initialized is True
    
    @pytest.mark.asyncio
    async def test_plugin_shutdown(self):
        """Test plugin shutdown"""
        class TestPlugin(BasePlugin):
            def __init__(self):
                super().__init__(
                    name="test_plugin",
                    version="1.0.0",
                    plugin_type=PluginType.PROCESSOR
                )
                self.shutdown_called = False
            
            async def shutdown(self):
                self.shutdown_called = True
        
        plugin = TestPlugin()
        await plugin.shutdown()
        
        assert plugin.shutdown_called is True


class TestPluginRegistry:
    """Tests for PluginRegistry"""
    
    @pytest.fixture
    def plugin_registry(self):
        """Create plugin registry"""
        return PluginRegistry()
    
    def test_register_plugin(self, plugin_registry):
        """Test registering a plugin"""
        class TestPlugin(BasePlugin):
            def __init__(self):
                super().__init__(
                    name="test_plugin",
                    version="1.0.0",
                    plugin_type=PluginType.PROCESSOR
                )
        
        plugin = TestPlugin()
        plugin_registry.register(plugin)
        
        assert plugin_registry.get_plugin("test_plugin") == plugin
    
    def test_get_plugin_not_found(self, plugin_registry):
        """Test getting non-existent plugin"""
        plugin = plugin_registry.get_plugin("non_existent")
        
        assert plugin is None
    
    def test_get_plugins_by_type(self, plugin_registry):
        """Test getting plugins by type"""
        class ProcessorPlugin(BasePlugin):
            def __init__(self, name):
                super().__init__(
                    name=name,
                    version="1.0.0",
                    plugin_type=PluginType.PROCESSOR
                )
        
        class AnalyzerPlugin(BasePlugin):
            def __init__(self, name):
                super().__init__(
                    name=name,
                    version="1.0.0",
                    plugin_type=PluginType.ANALYZER
                )
        
        plugin_registry.register(ProcessorPlugin("processor1"))
        plugin_registry.register(ProcessorPlugin("processor2"))
        plugin_registry.register(AnalyzerPlugin("analyzer1"))
        
        processors = plugin_registry.get_plugins_by_type(PluginType.PROCESSOR)
        
        assert len(processors) == 2
        assert all(p.plugin_type == PluginType.PROCESSOR for p in processors)
    
    @pytest.mark.asyncio
    async def test_initialize_all_plugins(self, plugin_registry):
        """Test initializing all plugins"""
        initialized_plugins = []
        
        class TestPlugin(BasePlugin):
            def __init__(self, name):
                super().__init__(
                    name=name,
                    version="1.0.0",
                    plugin_type=PluginType.PROCESSOR
                )
            
            async def initialize(self):
                initialized_plugins.append(self.name)
        
        plugin1 = TestPlugin("plugin1")
        plugin2 = TestPlugin("plugin2")
        
        plugin_registry.register(plugin1)
        plugin_registry.register(plugin2)
        
        await plugin_registry.initialize_all()
        
        assert len(initialized_plugins) == 2
        assert "plugin1" in initialized_plugins
        assert "plugin2" in initialized_plugins
    
    @pytest.mark.asyncio
    async def test_shutdown_all_plugins(self, plugin_registry):
        """Test shutting down all plugins"""
        shutdown_plugins = []
        
        class TestPlugin(BasePlugin):
            def __init__(self, name):
                super().__init__(
                    name=name,
                    version="1.0.0",
                    plugin_type=PluginType.PROCESSOR
                )
            
            async def shutdown(self):
                shutdown_plugins.append(self.name)
        
        plugin1 = TestPlugin("plugin1")
        plugin2 = TestPlugin("plugin2")
        
        plugin_registry.register(plugin1)
        plugin_registry.register(plugin2)
        
        await plugin_registry.shutdown_all()
        
        assert len(shutdown_plugins) == 2


class TestPluginExecution:
    """Tests for plugin execution"""
    
    @pytest.mark.asyncio
    async def test_plugin_process_data(self):
        """Test plugin processing data"""
        class DataProcessorPlugin(BasePlugin):
            def __init__(self):
                super().__init__(
                    name="data_processor",
                    version="1.0.0",
                    plugin_type=PluginType.PROCESSOR
                )
            
            async def process(self, data):
                return {"processed": data, "enhanced": True}
        
        plugin = DataProcessorPlugin()
        result = await plugin.process({"input": "data"})
        
        assert result["processed"]["input"] == "data"
        assert result["enhanced"] is True
    
    @pytest.mark.asyncio
    async def test_plugin_chain_execution(self):
        """Test chaining multiple plugins"""
        class Plugin1(BasePlugin):
            def __init__(self):
                super().__init__(
                    name="plugin1",
                    version="1.0.0",
                    plugin_type=PluginType.PROCESSOR
                )
            
            async def process(self, data):
                data["step1"] = "done"
                return data
        
        class Plugin2(BasePlugin):
            def __init__(self):
                super().__init__(
                    name="plugin2",
                    version="1.0.0",
                    plugin_type=PluginType.PROCESSOR
                )
            
            async def process(self, data):
                data["step2"] = "done"
                return data
        
        plugin1 = Plugin1()
        plugin2 = Plugin2()
        
        data = {"input": "data"}
        data = await plugin1.process(data)
        data = await plugin2.process(data)
        
        assert data["step1"] == "done"
        assert data["step2"] == "done"



