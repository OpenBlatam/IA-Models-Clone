"""
Test helpers for Contador SAM3 Agent tests.

Centralizes common test patterns to eliminate duplication.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, Callable


def create_mock_openrouter_response(
    response_text: str = "Test response",
    tokens_used: int = 100,
    model: str = "anthropic/claude-3.5-sonnet"
) -> Dict[str, Any]:
    """
    Create a mock OpenRouter response dictionary.
    
    Args:
        response_text: Response text content
        tokens_used: Number of tokens used
        model: Model name
        
    Returns:
        Mock response dictionary
        
    Example:
        >>> mock_response = create_mock_openrouter_response("ISR calculado: $5,000")
    """
    return {
        "response": response_text,
        "tokens_used": tokens_used,
        "model": model
    }


def patch_openrouter_client(agent, mock_response: Dict[str, Any]):
    """
    Patch OpenRouter client with mock response.
    
    Args:
        agent: Agent instance to patch
        mock_response: Mock response dictionary
        
    Returns:
        Context manager for patching
        
    Example:
        >>> with patch_openrouter_client(agent, mock_response):
        ...     task_id = await agent.calcular_impuestos(...)
    """
    return patch.object(
        agent.openrouter_client,
        'chat_completion',
        return_value=mock_response
    )


async def assert_task_submitted(agent, task_id: str):
    """
    Assert that a task was submitted successfully.
    
    Args:
        agent: Agent instance
        task_id: Task ID to check
        
    Raises:
        AssertionError: If task was not submitted or status is invalid
    """
    assert task_id is not None, "Task ID should not be None"
    
    status = await agent.get_task_status(task_id)
    assert status["status"] in ["pending", "processing", "completed"], \
        f"Task status should be pending, processing, or completed, got {status['status']}"

