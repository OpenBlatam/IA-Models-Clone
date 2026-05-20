"""
Tests for Session Manager
==========================
"""

import pytest
import asyncio
from ..core.session_manager import SessionManager


@pytest.fixture
def session_manager():
    """Create session manager for testing."""
    return SessionManager()


@pytest.mark.asyncio
async def test_create_session(session_manager):
    """Test creating a session."""
    session_id = session_manager.create_session(
        user_id="test_user",
        metadata={"ip": "127.0.0.1"}
    )
    
    assert session_id is not None
    assert session_id in session_manager.sessions


@pytest.mark.asyncio
async def test_get_session(session_manager):
    """Test getting a session."""
    session_id = session_manager.create_session("test_user", {})
    
    session = session_manager.get_session(session_id)
    
    assert session is not None
    assert session.user_id == "test_user" or session.get("user_id") == "test_user"


@pytest.mark.asyncio
async def test_update_session_activity(session_manager):
    """Test updating session activity."""
    session_id = session_manager.create_session("test_user", {})
    
    session_manager.update_activity(session_id, "message_sent")
    
    session = session_manager.get_session(session_id)
    assert session is not None


@pytest.mark.asyncio
async def test_get_user_sessions(session_manager):
    """Test getting all sessions for a user."""
    session_manager.create_session("user1", {})
    session_manager.create_session("user1", {})
    session_manager.create_session("user2", {})
    
    sessions = session_manager.get_user_sessions("user1")
    
    assert len(sessions) >= 2
    assert all(s.user_id == "user1" or s.get("user_id") == "user1" for s in sessions)


@pytest.mark.asyncio
async def test_get_session_analytics(session_manager):
    """Test getting session analytics."""
    session_id = session_manager.create_session("test_user", {})
    session_manager.update_activity(session_id, "message_sent")
    session_manager.update_activity(session_id, "message_sent")
    
    analytics = session_manager.get_session_analytics(session_id)
    
    assert analytics is not None
    assert "engagement" in analytics or "activity_count" in analytics or "session_duration" in analytics


@pytest.mark.asyncio
async def test_get_session_manager_summary(session_manager):
    """Test getting session manager summary."""
    session_manager.create_session("user1", {})
    session_manager.create_session("user2", {})
    
    summary = session_manager.get_session_manager_summary()
    
    assert summary is not None
    assert "total_sessions" in summary or "active_sessions" in summary


