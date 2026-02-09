"""Tests for coaching service."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from ..services.coaching_service import DogTrainingCoach
from ..infrastructure.openrouter import OpenRouterClient


@pytest.fixture
def mock_openrouter_client():
    """Mock OpenRouter client."""
    client = MagicMock(spec=OpenRouterClient)
    client.generate_text = AsyncMock(return_value={
        "choices": [{"message": {"content": "Test advice"}}],
        "model": "test-model"
    })
    return client


@pytest.fixture
def coach(mock_openrouter_client):
    """Coaching service instance."""
    return DogTrainingCoach(openrouter_client=mock_openrouter_client)


@pytest.mark.asyncio
async def test_get_coaching_advice(coach):
    """Test coaching advice."""
    result = await coach.get_coaching_advice(
        question="How to train a dog?",
        dog_breed="Golden Retriever"
    )
    assert result["success"] is True
    assert "advice" in result


@pytest.mark.asyncio
async def test_create_training_plan(coach):
    """Test training plan creation."""
    result = await coach.create_training_plan(
        dog_breed="German Shepherd",
        dog_age="1 year",
        training_goals=["obedience"]
    )
    assert result["success"] is True
    assert "plan" in result

