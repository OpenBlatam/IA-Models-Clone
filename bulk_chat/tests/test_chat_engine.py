"""
Tests for Chat Engine
====================
"""

import pytest
import asyncio
from bulk_chat.core.chat_engine import ContinuousChatEngine
from bulk_chat.core.chat_session import ChatSession, ChatState


@pytest.fixture
def chat_engine():
    """Create chat engine for testing."""
    return ContinuousChatEngine(
        auto_continue=False,  # Disable auto-continue for testing
        enable_cache=False,
        enable_metrics=False,
    )


@pytest.mark.asyncio
async def test_create_session(chat_engine):
    """Test session creation."""
    session = await chat_engine.create_session(
        user_id="test_user",
        initial_message="Test message",
    )
    
    assert session is not None
    assert session.user_id == "test_user"
    assert len(session.messages) == 1
    assert session.messages[0].content == "Test message"


@pytest.mark.asyncio
async def test_pause_resume(chat_engine):
    """Test pause and resume functionality."""
    session = await chat_engine.create_session()
    
    await chat_engine.pause_session(session.session_id, "Test pause")
    assert session.is_paused
    assert session.state == ChatState.PAUSED
    
    await chat_engine.resume_session(session.session_id)
    assert not session.is_paused
    assert session.state == ChatState.ACTIVE


@pytest.mark.asyncio
async def test_stop_session(chat_engine):
    """Test session stop."""
    session = await chat_engine.create_session()
    
    await chat_engine.stop_session(session.session_id)
    assert session.is_stopped()
    assert session.state == ChatState.STOPPED


@pytest.mark.asyncio
async def test_add_user_message(chat_engine):
    """Test adding user message."""
    session = await chat_engine.create_session()
    
    await chat_engine.add_user_message(session.session_id, "Hello")
    
    assert len(session.messages) == 1
    assert session.messages[0].role == "user"
    assert session.messages[0].content == "Hello"


@pytest.mark.asyncio
async def test_session_not_found(chat_engine):
    """Test error handling for non-existent session."""
    with pytest.raises(ValueError):
        await chat_engine.pause_session("non_existent_session")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
































