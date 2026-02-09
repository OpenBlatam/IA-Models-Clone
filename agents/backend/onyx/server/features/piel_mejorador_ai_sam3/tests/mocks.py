"""
Mock objects for testing.
"""

from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any


class MockOpenRouterClient:
    """Mock OpenRouter client."""
    
    def __init__(self):
        self.chat_completion = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "content": "Mock response"
                }
            }]
        })
        self.process_image = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "content": "Mock image response"
                }
            }]
        })
        self.close = AsyncMock()


class MockTruthGPTClient:
    """Mock TruthGPT client."""
    
    def __init__(self):
        self.process_with_truthgpt = AsyncMock(return_value={
            "optimized": True,
            "result": "Mock optimized result"
        })
        self.optimize_query = AsyncMock(return_value="Mock optimized query")
        self.close = AsyncMock()


class MockTaskManager:
    """Mock task manager."""
    
    def __init__(self):
        self.create_task = AsyncMock(return_value="mock-task-id")
        self.get_task = AsyncMock(return_value={
            "id": "mock-task-id",
            "status": "pending"
        })
        self.update_task_status = AsyncMock()
        self.complete_task = AsyncMock()
        self.fail_task = AsyncMock()


def create_mock_agent(config: Any) -> Dict[str, Any]:
    """Create a mock agent with all dependencies."""
    return {
        "openrouter_client": MockOpenRouterClient(),
        "truthgpt_client": MockTruthGPTClient(),
        "task_manager": MockTaskManager(),
        "config": config,
    }




