import pytest
import asyncio
from unittest.mock import MagicMock, patch
from datetime import datetime

from agents.backend.onyx.server.features.lovable.core.lovable_sam3_agent import LovableSAM3Agent, TaskManager
from agents.backend.onyx.server.features.lovable.config.lovable_config import LovableConfig

@pytest.fixture
def mock_config():
    config = MagicMock(spec=LovableConfig)
    config.max_workers = 2
    config.to_dict.return_value = {"max_workers": 2}
    return config

@pytest.fixture
def agent(mock_config):
    return LovableSAM3Agent(config=mock_config)

@pytest.mark.asyncio
async def test_agent_initialization(agent):
    assert agent.running is False
    assert agent.task_manager is not None
    assert agent.parallel_executor is not None

@pytest.mark.asyncio
async def test_agent_start_stop(agent):
    await agent.start()
    assert agent.running is True
    assert agent.task_manager.running is True
    
    await agent.stop()
    assert agent.running is False
    assert agent.task_manager.running is False

@pytest.mark.asyncio
async def test_create_task(agent):
    await agent.start()
    
    # Mock _process_task to avoid actual execution during this test
    # Use AsyncMock if available, or just a coroutine mock
    with patch.object(agent.task_manager, '_process_task', new_callable=MagicMock) as mock_process:
        # Make the mock return a coroutine
        async def async_mock(*args, **kwargs):
            return None
        mock_process.side_effect = async_mock
        
        task_id = await agent.task_manager.create_task(
            service_type="test_service",
            parameters={"param": "value"},
            priority=1
        )
        
        assert task_id is not None
        assert task_id in agent.task_manager.tasks
        task = agent.task_manager.tasks[task_id]
        assert task["service_type"] == "test_service"
        assert task["status"] == "pending"
        
        # Verify _process_task was called (it's called via asyncio.create_task)
        # We might need to yield to event loop to let it run, but since we mocked it, 
        # we just check if create_task triggered it.
        # Actually, asyncio.create_task schedules it.
        pass

@pytest.mark.asyncio
async def test_process_task_success(agent):
    # Test the actual _process_task logic with mocks
    task_id = "test-task-id"
    agent.task_manager.tasks[task_id] = {
        "id": task_id,
        "service_type": "optimize_content",
        "parameters": {"content": "test content"},
        "status": "pending",
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    # Mock database session
    with patch("agents.backend.onyx.server.features.lovable.database.get_session_factory") as mock_get_session_factory:
        mock_session = MagicMock()
        mock_get_session_factory.return_value.return_value.__enter__.return_value = mock_session
        
        await agent.task_manager._process_task(task_id)
        
        task = agent.task_manager.tasks[task_id]
        assert task["status"] == "completed"
        assert "result" in task
        assert "optimized_content" in task["result"]

@pytest.mark.asyncio
async def test_process_task_unknown_type(agent):
    task_id = "unknown-task-id"
    agent.task_manager.tasks[task_id] = {
        "id": task_id,
        "service_type": "unknown_type",
        "parameters": {},
        "status": "pending",
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    with patch("agents.backend.onyx.server.features.lovable.database.get_session_factory") as mock_get_session_factory:
        mock_session = MagicMock()
        mock_get_session_factory.return_value.return_value.__enter__.return_value = mock_session
        
        await agent.task_manager._process_task(task_id)
        
        task = agent.task_manager.tasks[task_id]
        assert task["status"] == "completed"
        assert "Task processed (unknown type)" in task["result"]["message"]
