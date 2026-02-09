"""
Tests for AITutor class.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from core.tutor import AITutor
from config.tutor_config import TutorConfig, OpenRouterConfig


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = TutorConfig()
    config.openrouter.api_key = "test_api_key"
    return config


@pytest.fixture
def tutor(mock_config):
    """Create a tutor instance for testing."""
    with patch('core.tutor.httpx.AsyncClient'):
        return AITutor(mock_config)


@pytest.mark.asyncio
async def test_tutor_initialization(tutor):
    """Test tutor initialization."""
    assert tutor is not None
    assert tutor.config is not None


@pytest.mark.asyncio
async def test_ask_question(tutor):
    """Test asking a question."""
    with patch.object(tutor.client, 'post', new_callable=AsyncMock) as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test answer"}}],
            "model": "test-model",
            "usage": {}
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        result = await tutor.ask_question(
            question="Test question",
            subject="matematicas",
            difficulty="intermedio"
        )
        
        assert result is not None
        assert "answer" in result
        assert result["answer"] == "Test answer"


@pytest.mark.asyncio
async def test_explain_concept(tutor):
    """Test explaining a concept."""
    with patch.object(tutor, 'ask_question', new_callable=AsyncMock) as mock_ask:
        mock_ask.return_value = {
            "answer": "Test explanation",
            "model": "test-model",
            "usage": {},
            "timestamp": "2024-01-01T00:00:00"
        }
        
        result = await tutor.explain_concept(
            concept="derivadas",
            subject="matematicas",
            difficulty="avanzado"
        )
        
        assert result is not None
        assert "answer" in result
        mock_ask.assert_called_once()


@pytest.mark.asyncio
async def test_generate_exercise(tutor):
    """Test generating exercises."""
    with patch.object(tutor, 'ask_question', new_callable=AsyncMock) as mock_ask:
        mock_ask.return_value = {
            "answer": "Test exercises",
            "model": "test-model",
            "usage": {},
            "timestamp": "2024-01-01T00:00:00"
        }
        
        result = await tutor.generate_exercise(
            topic="algebra",
            subject="matematicas",
            difficulty="intermedio",
            num_exercises=5
        )
        
        assert result is not None
        assert "answer" in result
        mock_ask.assert_called_once()


def test_build_system_prompt(tutor):
    """Test system prompt building."""
    prompt = tutor._build_system_prompt("matematicas", "intermedio")
    
    assert "tutor educativo" in prompt.lower()
    assert "matematicas" in prompt.lower()
    assert "intermedio" in prompt.lower()


def test_build_prompt(tutor):
    """Test user prompt building."""
    prompt = tutor._build_prompt(
        question="Test question",
        subject="ciencias",
        difficulty="basico",
        context="Test context"
    )
    
    assert "Test question" in prompt
    assert "Test context" in prompt






