"""
Advanced mock system for HeyGen AI tests.
Enterprise-level mocking capabilities with comprehensive functionality.
"""

import pytest
import asyncio
import time
import json
import random
from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from unittest.mock import Mock, MagicMock, AsyncMock, patch, call
import threading
import concurrent.futures
from contextlib import contextmanager

class MockType(Enum):
    """Mock types."""
    SIMPLE = "simple"
    ADVANCED = "advanced"
    ASYNC = "async"
    CONTEXT_MANAGER = "context_manager"
    PATCH = "patch"

class MockBehavior(Enum):
    """Mock behavior types."""
    RETURN_VALUE = "return_value"
    SIDE_EFFECT = "side_effect"
    RAISE_EXCEPTION = "raise_exception"
    CALLBACK = "callback"
    DELAYED_RESPONSE = "delayed_response"

@dataclass
class MockConfig:
    """Mock configuration."""
    mock_type: MockType = MockType.SIMPLE
    behavior: MockBehavior = MockBehavior.RETURN_VALUE
    return_value: Any = None
    side_effect: Optional[Callable] = None
    exception: Optional[Exception] = None
    delay: float = 0.0
    callback: Optional[Callable] = None
    call_count: int = 0
    max_calls: int = -1
    auto_reset: bool = True

class AdvancedMockSystem:
    """Advanced mock system with enterprise features."""
    
    def __init__(self):
        self.mocks: Dict[str, Mock] = {}
        self.mock_configs: Dict[str, MockConfig] = {}
        self.call_history: Dict[str, List[Dict[str, Any]]] = {}
        self.patches: List[Any] = []
    
    def create_mock(self, name: str, config: MockConfig = None) -> Mock:
        """Create an advanced mock with configuration."""
        if config is None:
            config = MockConfig()
        
        # Create base mock
        if config.mock_type == MockType.ASYNC:
            mock = AsyncMock()
        else:
            mock = Mock()
        
        # Configure mock behavior
        self._configure_mock(mock, config)
        
        # Store mock and config
        self.mocks[name] = mock
        self.mock_configs[name] = config
        self.call_history[name] = []
        
        return mock
    
    def _configure_mock(self, mock: Mock, config: MockConfig):
        """Configure mock based on config."""
        if config.behavior == MockBehavior.RETURN_VALUE:
            mock.return_value = config.return_value
        elif config.behavior == MockBehavior.SIDE_EFFECT:
            mock.side_effect = config.side_effect
        elif config.behavior == MockBehavior.RAISE_EXCEPTION:
            mock.side_effect = config.exception
        elif config.behavior == MockBehavior.CALLBACK:
            def callback_wrapper(*args, **kwargs):
                result = config.callback(*args, **kwargs)
                self._record_call(mock, args, kwargs, result)
                return result
            mock.side_effect = callback_wrapper
        elif config.behavior == MockBehavior.DELAYED_RESPONSE:
            def delayed_wrapper(*args, **kwargs):
                time.sleep(config.delay)
                result = config.return_value
                self._record_call(mock, args, kwargs, result)
                return result
            mock.side_effect = delayed_wrapper
        
        # Configure call count limits
        if config.max_calls > 0:
            def call_limit_wrapper(*args, **kwargs):
                if mock.call_count >= config.max_calls:
                    raise StopIteration("Mock call limit exceeded")
                return mock.return_value
            mock.side_effect = call_limit_wrapper
    
    def _record_call(self, mock: Mock, args: tuple, kwargs: dict, result: Any):
        """Record mock call history."""
        call_info = {
            "timestamp": datetime.now().isoformat(),
            "args": args,
            "kwargs": kwargs,
            "result": result,
            "call_count": mock.call_count
        }
        
        # Find mock name
        mock_name = None
        for name, m in self.mocks.items():
            if m is mock:
                mock_name = name
                break
        
        if mock_name:
            self.call_history[mock_name].append(call_info)
    
    def get_mock(self, name: str) -> Optional[Mock]:
        """Get mock by name."""
        return self.mocks.get(name)
    
    def reset_mock(self, name: str):
        """Reset mock to initial state."""
        if name in self.mocks:
            self.mocks[name].reset_mock()
            if self.mock_configs[name].auto_reset:
                self.call_history[name] = []
    
    def reset_all_mocks(self):
        """Reset all mocks."""
        for mock in self.mocks.values():
            mock.reset_mock()
        self.call_history.clear()
    
    def get_call_history(self, name: str) -> List[Dict[str, Any]]:
        """Get call history for a mock."""
        return self.call_history.get(name, [])
    
    def assert_called_with(self, name: str, *args, **kwargs):
        """Assert mock was called with specific arguments."""
        mock = self.get_mock(name)
        assert mock is not None, f"Mock '{name}' not found"
        mock.assert_called_with(*args, **kwargs)
    
    def assert_called_times(self, name: str, times: int):
        """Assert mock was called specific number of times."""
        mock = self.get_mock(name)
        assert mock is not None, f"Mock '{name}' not found"
        assert mock.call_count == times, f"Expected {times} calls, got {mock.call_count}"
    
    def assert_not_called(self, name: str):
        """Assert mock was not called."""
        mock = self.get_mock(name)
        assert mock is not None, f"Mock '{name}' not found"
        assert mock.call_count == 0, f"Mock was called {mock.call_count} times"
    
    def create_patch_mock(self, target: str, **kwargs) -> Any:
        """Create a patch mock."""
        patch_mock = patch(target, **kwargs)
        self.patches.append(patch_mock)
        return patch_mock
    
    def start_patches(self):
        """Start all patches."""
        for patch_mock in self.patches:
            patch_mock.start()
    
    def stop_patches(self):
        """Stop all patches."""
        for patch_mock in self.patches:
            patch_mock.stop()
        self.patches.clear()
    
    def create_context_manager_mock(self, name: str, enter_value: Any = None, exit_value: Any = None):
        """Create a context manager mock."""
        mock = MagicMock()
        mock.__enter__ = Mock(return_value=enter_value)
        mock.__exit__ = Mock(return_value=exit_value)
        
        self.mocks[name] = mock
        self.mock_configs[name] = MockConfig(mock_type=MockType.CONTEXT_MANAGER)
        self.call_history[name] = []
        
        return mock
    
    def create_async_mock(self, name: str, return_value: Any = None, delay: float = 0.0):
        """Create an async mock."""
        async def async_function(*args, **kwargs):
            if delay > 0:
                await asyncio.sleep(delay)
            return return_value
        
        mock = AsyncMock(side_effect=async_function)
        
        self.mocks[name] = mock
        self.mock_configs[name] = MockConfig(
            mock_type=MockType.ASYNC,
            return_value=return_value,
            delay=delay
        )
        self.call_history[name] = []
        
        return mock
    
    def create_mock_with_side_effects(self, name: str, side_effects: List[Any]):
        """Create a mock with multiple side effects."""
        mock = Mock(side_effect=side_effects)
        
        self.mocks[name] = mock
        self.mock_configs[name] = MockConfig(
            behavior=MockBehavior.SIDE_EFFECT,
            side_effect=side_effects
        )
        self.call_history[name] = []
        
        return mock
    
    def create_mock_that_raises(self, name: str, exception: Exception, after_calls: int = 0):
        """Create a mock that raises exception after certain number of calls."""
        call_count = 0
        
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > after_calls:
                raise exception
            return f"success_{call_count}"
        
        mock = Mock(side_effect=side_effect)
        
        self.mocks[name] = mock
        self.mock_configs[name] = MockConfig(
            behavior=MockBehavior.SIDE_EFFECT,
            side_effect=side_effect
        )
        self.call_history[name] = []
        
        return mock
    
    def create_mock_with_callback(self, name: str, callback: Callable):
        """Create a mock with custom callback."""
        def callback_wrapper(*args, **kwargs):
            result = callback(*args, **kwargs)
            self._record_call(self.mocks[name], args, kwargs, result)
            return result
        
        mock = Mock(side_effect=callback_wrapper)
        
        self.mocks[name] = mock
        self.mock_configs[name] = MockConfig(
            behavior=MockBehavior.CALLBACK,
            callback=callback
        )
        self.call_history[name] = []
        
        return mock
    
    def get_mock_statistics(self) -> Dict[str, Any]:
        """Get statistics for all mocks."""
        stats = {}
        for name, mock in self.mocks.items():
            stats[name] = {
                "call_count": mock.call_count,
                "call_history_length": len(self.call_history.get(name, [])),
                "mock_type": self.mock_configs[name].mock_type.value,
                "behavior": self.mock_configs[name].behavior.value,
                "last_call": self.call_history[name][-1] if self.call_history.get(name) else None
            }
        return stats

class TestAdvancedMockSystem:
    """Test advanced mock system."""
    
    def __init__(self):
        self.mock_system = AdvancedMockSystem()
    
    def test_simple_mock_creation(self):
        """Test simple mock creation."""
        mock = self.mock_system.create_mock("simple_mock")
        
        # Test basic functionality
        result = mock("test_arg", keyword="test_value")
        assert result is not None
        
        # Test call tracking
        assert mock.call_count == 1
        mock.assert_called_with("test_arg", keyword="test_value")
    
    def test_mock_with_return_value(self):
        """Test mock with specific return value."""
        config = MockConfig(
            behavior=MockBehavior.RETURN_VALUE,
            return_value="test_result"
        )
        
        mock = self.mock_system.create_mock("return_mock", config)
        result = mock()
        
        assert result == "test_result"
        assert mock.call_count == 1
    
    def test_mock_with_side_effect(self):
        """Test mock with side effect function."""
        def side_effect_func(*args, **kwargs):
            return f"processed_{args[0] if args else 'no_args'}"
        
        config = MockConfig(
            behavior=MockBehavior.SIDE_EFFECT,
            side_effect=side_effect_func
        )
        
        mock = self.mock_system.create_mock("side_effect_mock", config)
        result = mock("test_input")
        
        assert result == "processed_test_input"
        assert mock.call_count == 1
    
    def test_mock_that_raises_exception(self):
        """Test mock that raises exception."""
        config = MockConfig(
            behavior=MockBehavior.RAISE_EXCEPTION,
            exception=ValueError("Test error")
        )
        
        mock = self.mock_system.create_mock("exception_mock", config)
        
        with pytest.raises(ValueError, match="Test error"):
            mock()
    
    def test_mock_with_delayed_response(self):
        """Test mock with delayed response."""
        config = MockConfig(
            behavior=MockBehavior.DELAYED_RESPONSE,
            return_value="delayed_result",
            delay=0.1
        )
        
        mock = self.mock_system.create_mock("delayed_mock", config)
        
        start_time = time.time()
        result = mock()
        end_time = time.time()
        
        assert result == "delayed_result"
        assert end_time - start_time >= 0.1
        assert mock.call_count == 1
    
    def test_mock_with_callback(self):
        """Test mock with callback function."""
        def callback_func(*args, **kwargs):
            return {"args": args, "kwargs": kwargs, "processed": True}
        
        config = MockConfig(
            behavior=MockBehavior.CALLBACK,
            callback=callback_func
        )
        
        mock = self.mock_system.create_mock("callback_mock", config)
        result = mock("test_arg", keyword="test_value")
        
        assert result["args"] == ("test_arg",)
        assert result["kwargs"] == {"keyword": "test_value"}
        assert result["processed"] is True
        
        # Check call history
        history = self.mock_system.get_call_history("callback_mock")
        assert len(history) == 1
        assert history[0]["args"] == ("test_arg",)
        assert history[0]["kwargs"] == {"keyword": "test_value"}
    
    def test_async_mock(self):
        """Test async mock functionality."""
        mock = self.mock_system.create_async_mock("async_mock", "async_result", 0.01)
        
        async def test_async():
            result = await mock()
            return result
        
        result = asyncio.run(test_async())
        
        assert result == "async_result"
        assert mock.call_count == 1
    
    def test_context_manager_mock(self):
        """Test context manager mock."""
        mock = self.mock_system.create_context_manager_mock(
            "context_mock",
            enter_value="entered",
            exit_value="exited"
        )
        
        with mock as value:
            assert value == "entered"
        
        assert mock.__enter__.call_count == 1
        assert mock.__exit__.call_count == 1
    
    def test_mock_with_multiple_side_effects(self):
        """Test mock with multiple side effects."""
        side_effects = ["first", "second", "third"]
        mock = self.mock_system.create_mock_with_side_effects("multi_mock", side_effects)
        
        assert mock() == "first"
        assert mock() == "second"
        assert mock() == "third"
        assert mock.call_count == 3
    
    def test_mock_that_raises_after_calls(self):
        """Test mock that raises exception after certain calls."""
        mock = self.mock_system.create_mock_that_raises(
            "raise_mock",
            ValueError("Too many calls"),
            after_calls=2
        )
        
        # First two calls should succeed
        assert mock() == "success_1"
        assert mock() == "success_2"
        
        # Third call should raise exception
        with pytest.raises(ValueError, match="Too many calls"):
            mock()
    
    def test_mock_with_custom_callback(self):
        """Test mock with custom callback."""
        def custom_callback(*args, **kwargs):
            return {
                "timestamp": datetime.now().isoformat(),
                "args_count": len(args),
                "kwargs_count": len(kwargs),
                "total_calls": self.mock_system.get_mock("custom_mock").call_count + 1
            }
        
        mock = self.mock_system.create_mock_with_callback("custom_mock", custom_callback)
        
        result = mock("arg1", "arg2", key1="value1")
        
        assert result["args_count"] == 2
        assert result["kwargs_count"] == 1
        assert result["total_calls"] == 1
        assert "timestamp" in result
    
    def test_mock_call_assertions(self):
        """Test mock call assertions."""
        mock = self.mock_system.create_mock("assertion_mock")
        
        # Test various calls
        mock("first_call")
        mock("second_call", keyword="value")
        mock("third_call", key1="val1", key2="val2")
        
        # Test assertions
        self.mock_system.assert_called_times("assertion_mock", 3)
        self.mock_system.assert_called_with("assertion_mock", "first_call")
        self.mock_system.assert_called_with("assertion_mock", "second_call", keyword="value")
        
        # Test not called
        new_mock = self.mock_system.create_mock("not_called_mock")
        self.mock_system.assert_not_called("not_called_mock")
    
    def test_mock_reset(self):
        """Test mock reset functionality."""
        mock = self.mock_system.create_mock("reset_mock")
        
        # Make some calls
        mock("call1")
        mock("call2")
        
        assert mock.call_count == 2
        
        # Reset mock
        self.mock_system.reset_mock("reset_mock")
        
        assert mock.call_count == 0
        
        # Test reset all
        mock1 = self.mock_system.create_mock("mock1")
        mock2 = self.mock_system.create_mock("mock2")
        
        mock1("call1")
        mock2("call2")
        
        self.mock_system.reset_all_mocks()
        
        assert mock1.call_count == 0
        assert mock2.call_count == 0
    
    def test_mock_statistics(self):
        """Test mock statistics collection."""
        # Create multiple mocks with different behaviors
        mock1 = self.mock_system.create_mock("mock1")
        mock2 = self.mock_system.create_mock("mock2")
        mock3 = self.mock_system.create_async_mock("mock3", "async_result")
        
        # Make some calls
        mock1("call1")
        mock1("call2")
        mock2("call3")
        
        # Get statistics
        stats = self.mock_system.get_mock_statistics()
        
        assert "mock1" in stats
        assert "mock2" in stats
        assert "mock3" in stats
        
        assert stats["mock1"]["call_count"] == 2
        assert stats["mock2"]["call_count"] == 1
        assert stats["mock3"]["call_count"] == 0
        
        assert stats["mock1"]["mock_type"] == "simple"
        assert stats["mock3"]["mock_type"] == "async"
    
    def test_patch_mock(self):
        """Test patch mock functionality."""
        # Create a patch mock
        patch_mock = self.mock_system.create_patch_mock(
            "builtins.open",
            return_value=Mock(read=Mock(return_value="test content"))
        )
        
        # Start patch
        self.mock_system.start_patches()
        
        try:
            # Test patched function
            with open("test.txt", "r") as f:
                content = f.read()
            
            assert content == "test content"
        finally:
            # Stop patch
            self.mock_system.stop_patches()
    
    def test_mock_integration(self):
        """Test mock integration with real code."""
        # Create mocks for external services
        api_mock = self.mock_system.create_mock("api_service")
        db_mock = self.mock_system.create_mock("database")
        
        # Configure mocks
        api_mock.get_data.return_value = {"status": "success", "data": [1, 2, 3]}
        db_mock.query.return_value = [{"id": 1, "name": "test"}]
        
        # Simulate application code
        def application_function():
            api_data = api_mock.get_data()
            db_data = db_mock.query("SELECT * FROM users")
            return {
                "api_status": api_data["status"],
                "api_count": len(api_data["data"]),
                "db_count": len(db_data)
            }
        
        # Test application function
        result = application_function()
        
        assert result["api_status"] == "success"
        assert result["api_count"] == 3
        assert result["db_count"] == 1
        
        # Verify mock calls
        api_mock.get_data.assert_called_once()
        db_mock.query.assert_called_once_with("SELECT * FROM users")

# Test fixtures
@pytest.fixture
def mock_system():
    """Mock system fixture."""
    return AdvancedMockSystem()

@pytest.fixture
def mock_config():
    """Mock configuration fixture."""
    return MockConfig()

# Test markers
pytestmark = pytest.mark.usefixtures("mock_system", "mock_config")

if __name__ == "__main__":
    # Run the mock tests
    pytest.main([__file__, "-v"])
