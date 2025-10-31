from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
from ..core.models import VideoRequest, VideoResponse, VideoQuality, VideoFormat
from ..config.config_manager import OnyxConfigManager, OnyxAIVideoConfig
from ..utils.logger import OnyxLogger
from ..utils.performance import PerformanceMonitor
from ..utils.security import SecurityManager, SecurityConfig
from ..core.integration import OnyxIntegrationManager, OnyxIntegrationConfig
from ..workflows.video_workflow import OnyxVideoWorkflow
from ..plugins.plugin_manager import OnyxPluginManager
from ..api.main import OnyxAIVideoSystem
    import yaml
    import yaml
from typing import Any, List, Dict, Optional
import logging
"""
Pytest configuration and fixtures for Onyx AI Video System tests.
"""


# Import system components


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config(temp_dir) -> Dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "system_name": "Test AI Video System",
        "version": "1.0.0",
        "environment": "testing",
        "debug": True,
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_path": str(temp_dir / "test.log"),
            "max_size": 1,
            "backup_count": 1,
            "use_onyx_logging": False
        },
        "llm": {
            "provider": "mock",
            "model": "test-model",
            "temperature": 0.7,
            "max_tokens": 1000,
            "timeout": 10,
            "retry_attempts": 1,
            "use_onyx_llm": False
        },
        "video": {
            "default_quality": "low",
            "default_format": "mp4",
            "default_duration": 10,
            "max_duration": 60,
            "output_directory": str(temp_dir / "output"),
            "temp_directory": str(temp_dir / "temp"),
            "cleanup_temp": True
        },
        "plugins": {
            "plugins_directory": str(temp_dir / "plugins"),
            "auto_load": False,
            "enable_all": False,
            "max_workers": 2,
            "timeout": 30,
            "retry_attempts": 1
        },
        "performance": {
            "enable_monitoring": False,
            "metrics_interval": 10,
            "cache_enabled": False,
            "cache_size": 10,
            "cache_ttl": 60,
            "gpu_enabled": False,
            "max_concurrent_requests": 2
        },
        "security": {
            "enable_encryption": False,
            "encryption_key": "test-key",
            "validate_input": True,
            "max_input_length": 1000,
            "rate_limit_enabled": False,
            "rate_limit_requests": 10,
            "use_onyx_security": False
        },
        "onyx": {
            "use_onyx_logging": False,
            "use_onyx_llm": False,
            "use_onyx_telemetry": False,
            "use_onyx_encryption": False,
            "use_onyx_threading": False,
            "use_onyx_retry": False,
            "use_onyx_gpu": False,
            "onyx_config_path": None
        },
        "custom": {}
    }


@pytest.fixture
def config_manager(sample_config, temp_dir) -> Any:
    """Create a test configuration manager."""
    config_file = temp_dir / "test_config.yaml"
    
    # Create config file
    with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        yaml.dump(sample_config, f)
    
    return OnyxConfigManager(str(config_file))


@pytest.fixture
def test_config(sample_config) -> OnyxAIVideoConfig:
    """Create a test configuration object."""
    return OnyxAIVideoConfig(**sample_config)


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock(spec=OnyxLogger)
    logger.info = Mock()
    logger.error = Mock()
    logger.warning = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def mock_performance_monitor():
    """Create a mock performance monitor."""
    monitor = Mock(spec=PerformanceMonitor)
    monitor.start_operation = Mock()
    monitor.end_operation = Mock()
    monitor.get_system_metrics = Mock(return_value={})
    monitor.get_performance_summary = Mock(return_value={})
    return monitor


@pytest.fixture
def mock_security_manager():
    """Create a mock security manager."""
    manager = Mock(spec=SecurityManager)
    manager.validate_access = AsyncMock(return_value=True)
    manager.check_rate_limit = Mock(return_value=(True, {}))
    manager.validate_input = Mock(return_value=(True, "valid input"))
    manager.encrypt_data = Mock(side_effect=lambda x: x)
    manager.decrypt_data = Mock(side_effect=lambda x: x)
    return manager


@pytest.fixture
def mock_onyx_integration():
    """Create a mock Onyx integration manager."""
    integration = Mock(spec=OnyxIntegrationManager)
    integration.initialize = AsyncMock()
    integration.process_video_request = AsyncMock()
    integration.get_system_status = AsyncMock(return_value={})
    return integration


@pytest.fixture
def mock_video_workflow():
    """Create a mock video workflow."""
    workflow = Mock(spec=OnyxVideoWorkflow)
    workflow.initialize = AsyncMock()
    workflow.generate_video = AsyncMock()
    workflow.generate_video_with_vision = AsyncMock()
    workflow.shutdown = AsyncMock()
    return workflow


@pytest.fixture
def mock_plugin_manager():
    """Create a mock plugin manager."""
    manager = Mock(spec=OnyxPluginManager)
    manager.initialize = AsyncMock()
    manager.load_plugins = AsyncMock()
    manager.get_plugins = Mock(return_value={})
    manager.shutdown = AsyncMock()
    return manager


@pytest.fixture
async def sample_video_request() -> VideoRequest:
    """Create a sample video request for testing."""
    return VideoRequest(
        input_text="Create a test video about artificial intelligence",
        user_id="test_user_001",
        quality=VideoQuality.LOW,
        duration=10,
        output_format=VideoFormat.MP4
    )


@pytest.fixture
def sample_video_response(sample_video_request) -> VideoResponse:
    """Create a sample video response for testing."""
    return VideoResponse(
        request_id=sample_video_request.request_id,
        status="completed",
        output_url="http://example.com/video.mp4",
        output_path="/tmp/test_video.mp4",
        duration=10.5,
        file_size=1024000,
        resolution="1920x1080",
        fps=30.0,
        processing_time=5.2,
        steps_completed=["text_processing", "video_generation"],
        metadata={
            "generated_at": "2024-01-01T12:00:00Z",
            "test_mode": True
        }
    )


@pytest.fixture
def mock_system(test_config, mock_logger, mock_performance_monitor, 
                mock_security_manager, mock_onyx_integration, 
                mock_video_workflow, mock_plugin_manager) -> Any:
    """Create a mock system for testing."""
    system = Mock(spec=OnyxAIVideoSystem)
    system.config = test_config
    system.logger = mock_logger
    system.performance_monitor = mock_performance_monitor
    system.security_manager = mock_security_manager
    system.onyx_integration = mock_onyx_integration
    system.video_workflow = mock_video_workflow
    system.plugin_manager = mock_plugin_manager
    system.initialized = True
    system.shutdown_requested = False
    system.metrics = Mock()
    system.active_requests = {}
    
    # Mock methods
    system.initialize = AsyncMock()
    system.shutdown = AsyncMock()
    system.generate_video = AsyncMock()
    system.generate_video_with_vision = AsyncMock()
    system.get_system_status = AsyncMock()
    system.get_metrics = AsyncMock()
    
    return system


@pytest.fixture
def real_system(temp_dir, sample_config) -> Any:
    """Create a real system instance for integration tests."""
    # Create necessary directories
    (temp_dir / "output").mkdir(exist_ok=True)
    (temp_dir / "temp").mkdir(exist_ok=True)
    (temp_dir / "logs").mkdir(exist_ok=True)
    (temp_dir / "plugins").mkdir(exist_ok=True)
    
    # Create config file
    config_file = temp_dir / "test_config.yaml"
    with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        yaml.dump(sample_config, f)
    
    return OnyxAIVideoSystem(str(config_file))


@pytest.fixture
def sample_plugin_config():
    """Sample plugin configuration for testing."""
    return {
        "name": "test_plugin",
        "version": "1.0.0",
        "enabled": True,
        "parameters": {
            "test_param": "test_value"
        },
        "timeout": 30,
        "max_workers": 1,
        "dependencies": [],
        "conflicts": [],
        "gpu_required": False,
        "memory_required": 128,
        "cpu_cores_required": 1,
        "description": "Test plugin for unit testing",
        "author": "Test Author",
        "category": "test"
    }


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    llm.agenerate = AsyncMock()
    llm.agenerate.return_value.generations = [[Mock(text="Generated text")]]
    return llm


@pytest.fixture
def mock_onyx_functions():
    """Mock Onyx functions for testing."""
    with patch('onyx.core.functions') as mock_functions:
        mock_functions.process_document = AsyncMock(return_value="processed")
        mock_functions.validate_user_access = AsyncMock(return_value=True)
        mock_functions.format_response = Mock(return_value="formatted")
        mock_functions.handle_error = Mock(return_value="handled")
        yield mock_functions


@pytest.fixture
def mock_onyx_utils():
    """Mock Onyx utilities for testing."""
    with patch('onyx.utils.logger') as mock_logger:
        with patch('onyx.utils.threadpool_concurrency') as mock_threading:
            with patch('onyx.utils.timing') as mock_timing:
                with patch('onyx.utils.retry_wrapper') as mock_retry:
                    with patch('onyx.utils.telemetry') as mock_telemetry:
                        with patch('onyx.utils.encryption') as mock_encryption:
                            with patch('onyx.utils.file') as mock_file:
                                with patch('onyx.utils.text_processing') as mock_text:
                                    with patch('onyx.utils.gpu_utils') as mock_gpu:
                                        with patch('onyx.utils.error_handling') as mock_error:
                                            with patch('onyx.llm.factory') as mock_llm_factory:
                                                with patch('onyx.llm.interfaces') as mock_llm_interfaces:
                                                    with patch('onyx.llm.utils') as mock_llm_utils:
                                                        with patch('onyx.db.engine') as mock_db:
                                                            with patch('onyx.db.models') as mock_models:
                                                                # Setup mocks
                                                                mock_logger.setup_logger = Mock(return_value=Mock())
                                                                mock_threading.ThreadSafeDict = Mock
                                                                mock_threading.run_functions_in_parallel = Mock()
                                                                mock_timing.time_function = Mock()
                                                                mock_retry.retry_wrapper = Mock()
                                                                mock_telemetry.TelemetryLogger = Mock()
                                                                mock_encryption.encrypt_data = Mock(side_effect=lambda x: x)
                                                                mock_encryption.decrypt_data = Mock(side_effect=lambda x: x)
                                                                mock_file.get_file_extension = Mock(return_value=".txt")
                                                                mock_file.get_file_size = Mock(return_value=1024)
                                                                mock_text.clean_text = Mock(side_effect=lambda x: x)
                                                                mock_text.extract_keywords = Mock(return_value=["test"])
                                                                mock_gpu.get_gpu_info = Mock(return_value={"available": False})
                                                                mock_gpu.is_gpu_available = Mock(return_value=False)
                                                                mock_error.handle_exception = Mock()
                                                                mock_llm_factory.get_default_llms = Mock(return_value=[Mock()])
                                                                mock_llm_factory.get_llm = Mock(return_value=Mock())
                                                                mock_llm_factory.get_default_llm_with_vision = Mock(return_value=Mock())
                                                                mock_llm_interfaces.LLM = Mock
                                                                mock_llm_utils.get_max_input_tokens_from_llm_provider = Mock(return_value=4000)
                                                                mock_db.get_session_with_current_tenant = AsyncMock()
                                                                mock_models.Persona = Mock
                                                                
                                                                yield {
                                                                    'logger': mock_logger,
                                                                    'threading': mock_threading,
                                                                    'timing': mock_timing,
                                                                    'retry': mock_retry,
                                                                    'telemetry': mock_telemetry,
                                                                    'encryption': mock_encryption,
                                                                    'file': mock_file,
                                                                    'text': mock_text,
                                                                    'gpu': mock_gpu,
                                                                    'error': mock_error,
                                                                    'llm_factory': mock_llm_factory,
                                                                    'llm_interfaces': mock_llm_interfaces,
                                                                    'llm_utils': mock_llm_utils,
                                                                    'db': mock_db,
                                                                    'models': mock_models
                                                                }


# Test markers
def pytest_configure(config) -> Any:
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "system: mark test as a system test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as a security test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Test utilities
class AsyncTestCase:
    """Base class for async test cases."""
    
    @pytest.fixture(autouse=True)
    def setup_async(self, event_loop) -> Any:
        """Setup async test environment."""
        self.loop = event_loop
        asyncio.set_event_loop(self.loop)


async def create_mock_video_request(**kwargs) -> VideoRequest:
    """Create a mock video request with default values."""
    defaults = {
        "input_text": "Test video request",
        "user_id": "test_user",
        "quality": VideoQuality.LOW,
        "duration": 10,
        "output_format": VideoFormat.MP4
    }
    defaults.update(kwargs)
    return VideoRequest(**defaults)


def create_mock_video_response(request_id: str, **kwargs) -> VideoResponse:
    """Create a mock video response with default values."""
    defaults = {
        "request_id": request_id,
        "status": "completed",
        "output_url": "http://example.com/video.mp4",
        "processing_time": 5.0
    }
    defaults.update(kwargs)
    return VideoResponse(**defaults) 