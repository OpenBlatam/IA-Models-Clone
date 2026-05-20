"""
Test Configuration
==================

Pytest configuration and fixtures for the Business Agents System.
"""

import pytest
import asyncio
from typing import Dict, Any, Generator
from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient

from ..main import app
from ..business_agents import BusinessAgentManager, BusinessArea
from ..core.container import ServiceContainer, reset_container
from ..core.dependencies import get_container

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def mock_agent_manager():
    """Create a mock agent manager for testing."""
    manager = Mock(spec=BusinessAgentManager)
    manager.list_agents = Mock(return_value=[])
    manager.get_agent = Mock(return_value=None)
    manager.list_workflows = Mock(return_value=[])
    manager.get_workflow = Mock(return_value=None)
    manager.execute_agent_capability = AsyncMock(return_value={"status": "completed"})
    manager.execute_business_workflow = AsyncMock(return_value={"status": "completed"})
    manager.create_business_workflow = AsyncMock()
    manager.generate_business_document = AsyncMock()
    return manager

@pytest.fixture(scope="function")
def test_container(mock_agent_manager):
    """Create a test service container."""
    # Reset global container
    reset_container()
    
    # Create test container
    container = ServiceContainer()
    container.register_singleton("agent_manager", mock_agent_manager)
    
    # Mock services
    from ..services import HealthService, SystemInfoService, MetricsService
    from ..services import AgentService, WorkflowService, DocumentService
    
    container.register_singleton("health_service", Mock(spec=HealthService))
    container.register_singleton("system_info_service", Mock(spec=SystemInfoService))
    container.register_singleton("metrics_service", Mock(spec=MetricsService))
    container.register_singleton("agent_service", Mock(spec=AgentService))
    container.register_singleton("workflow_service", Mock(spec=WorkflowService))
    container.register_singleton("document_service", Mock(spec=DocumentService))
    
    return container

@pytest.fixture(scope="function")
def test_client(test_container):
    """Create a test client with mocked dependencies."""
    # Override the dependency
    app.dependency_overrides[get_container] = lambda: test_container
    
    with TestClient(app) as client:
        yield client
    
    # Clean up
    app.dependency_overrides.clear()

@pytest.fixture
def sample_agent_data():
    """Sample agent data for testing."""
    return {
        "id": "test_agent_001",
        "name": "Test Marketing Agent",
        "business_area": "marketing",
        "description": "Test marketing agent for unit tests",
        "capabilities": [
            {
                "name": "test_capability",
                "description": "Test capability",
                "input_types": ["test_input"],
                "output_types": ["test_output"],
                "estimated_duration": 60
            }
        ],
        "is_active": True
    }

@pytest.fixture
def sample_workflow_data():
    """Sample workflow data for testing."""
    return {
        "name": "Test Workflow",
        "description": "Test workflow for unit tests",
        "business_area": "marketing",
        "steps": [
            {
                "name": "Test Step",
                "step_type": "task",
                "description": "Test step description",
                "agent_type": "test_agent_001",
                "parameters": {"test_param": "test_value"}
            }
        ],
        "variables": {"test_var": "test_value"}
    }

@pytest.fixture
def sample_document_data():
    """Sample document data for testing."""
    return {
        "document_type": "business_plan",
        "title": "Test Document",
        "description": "Test document for unit tests",
        "business_area": "marketing",
        "variables": {"test_var": "test_value"},
        "format": "markdown"
    }

@pytest.fixture
def sample_capability_execution_data():
    """Sample capability execution data for testing."""
    return {
        "agent_id": "test_agent_001",
        "capability_name": "test_capability",
        "inputs": {"test_input": "test_value"},
        "parameters": {"test_param": "test_value"}
    }

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test."""
    yield
    # Reset container after each test
    reset_container()

# Async test utilities
@pytest.fixture
def async_test():
    """Decorator for async tests."""
    def decorator(func):
        return pytest.mark.asyncio(func)
    return decorator

# Mock external dependencies
@pytest.fixture(autouse=True)
def mock_external_services():
    """Mock external services for testing."""
    # Mock OpenAI API
    with pytest.MonkeyPatch().context() as m:
        m.setattr("openai.ChatCompletion.create", Mock())
        m.setattr("anthropic.Anthropic", Mock())
        yield
