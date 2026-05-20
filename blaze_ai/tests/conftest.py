"""
Pytest configuration for Blaze AI tests.

This file provides common fixtures and configuration for all tests
in the Blaze AI system.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Mock the logging system to avoid output during tests
@pytest.fixture(autouse=True)
def mock_logging():
    """Mock the logging system for all tests."""
    with patch('engines.plugins.get_logger') as mock_logger, \
         patch('engines.cache.get_logger') as mock_cache_logger, \
         patch('engines.base.get_logger') as mock_base_logger:
        
        mock_logger.return_value = Mock()
        mock_cache_logger.return_value = Mock()
        mock_base_logger.return_value = Mock()
        yield

@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_plugin_config():
    """Provide a sample plugin configuration for testing."""
    from engines.plugins import PluginConfig
    
    return PluginConfig(
        plugin_directories=["test_plugins"],
        enable_hot_reload=False,
        enable_plugin_validation=True,
        allow_unsafe_plugins=False
    )

@pytest.fixture
def sample_plugin_metadata():
    """Provide a sample plugin metadata for testing."""
    from engines.plugins import PluginMetadata
    from engines.base import EnginePriority
    
    return PluginMetadata(
        name="test_plugin",
        version="1.0.0",
        description="A test plugin for unit testing",
        author="Test Author",
        license="MIT",
        tags=["test", "unit"],
        priority=EnginePriority.NORMAL
    )

@pytest.fixture
def mock_engine_class():
    """Provide a mock engine class for testing."""
    from engines.base import Engine
    
    class MockEngine(Engine):
        def __init__(self, name="mock_engine"):
            self.name = name
            self.status = "idle"
        
        async def execute(self, operation, params):
            return {"result": f"Mock execution of {operation}"}
        
        def get_health_status(self):
            from engines.base import HealthStatus
            return HealthStatus.HEALTHY
        
        def get_config(self):
            return {"name": self.name}
        
        def update_config(self, config):
            pass
        
        def get_metrics(self):
            return {"executions": 0}
        
        def reset_metrics(self):
            pass
    
    return MockEngine

@pytest.fixture
def sample_plugin_file(temp_test_dir):
    """Create a sample plugin file for testing."""
    plugin_file = temp_test_dir / "test_plugin.py"
    plugin_content = '''"""
# version: 1.0.0
# author: Test Author
# description: A test plugin for unit testing
# tags: test, unit
"""
from engines.base import Engine

class TestEngine(Engine):
    def __init__(self):
        self.name = "test_engine"
    
    async def execute(self, operation, params):
        return {"result": "test_execution"}
    
    def get_health_status(self):
        from engines.base import HealthStatus
        return HealthStatus.HEALTHY
    
    def get_config(self):
        return {"name": self.name}
    
    def update_config(self, config):
        pass
    
    def get_metrics(self):
        return {"executions": 0}
    
    def reset_metrics(self):
        pass
'''
    plugin_file.write_text(plugin_content)
    return plugin_file

@pytest.fixture
def sample_plugin_json(temp_test_dir):
    """Create a sample plugin.json file for testing."""
    plugin_dir = temp_test_dir / "json_plugin"
    plugin_dir.mkdir()
    
    plugin_json = plugin_dir / "plugin.json"
    plugin_data = {
        "name": "json_plugin",
        "version": "1.0.0",
        "description": "A JSON-based test plugin",
        "author": "Test Author",
        "license": "MIT",
        "tags": ["test", "json"],
        "dependencies": [],
        "requirements": {"python": ">=3.8"},
        "engine_types": ["llm"],
        "priority": "normal"
    }
    
    plugin_json.write_text(str(plugin_data).replace("'", '"'))
    return plugin_dir

# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names."""
    for item in items:
        if "test_plugins" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "test_cache" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "test_" in item.nodeid:
            item.add_marker(pytest.mark.unit)
