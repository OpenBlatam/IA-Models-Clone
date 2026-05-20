"""
Tests for Session Storage
=========================
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from ..core.session_storage import JSONSessionStorage
from ..core.chat_session import ChatSession, ChatState


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def storage(temp_dir):
    """Create session storage for testing."""
    return JSONSessionStorage(storage_path=str(temp_dir))


@pytest.mark.asyncio
async def test_save_session(storage):
    """Test saving a session."""
    session = ChatSession(
        session_id="test_session",
        user_id="test_user",
        state=ChatState.ACTIVE
    )
    
    await storage.save_session(session)
    
    # Verify file exists
    assert (Path(storage.storage_path) / f"{session.session_id}.json").exists()


@pytest.mark.asyncio
async def test_load_session(storage):
    """Test loading a session."""
    session = ChatSession(
        session_id="test_session",
        user_id="test_user",
        state=ChatState.ACTIVE
    )
    
    await storage.save_session(session)
    loaded = await storage.load_session("test_session")
    
    assert loaded is not None
    assert loaded.session_id == "test_session"
    assert loaded.user_id == "test_user"


@pytest.mark.asyncio
async def test_load_session_not_found(storage):
    """Test loading non-existent session."""
    loaded = await storage.load_session("non_existent")
    
    assert loaded is None


@pytest.mark.asyncio
async def test_delete_session(storage):
    """Test deleting a session."""
    session = ChatSession(
        session_id="test_session",
        user_id="test_user"
    )
    
    await storage.save_session(session)
    await storage.delete_session("test_session")
    
    loaded = await storage.load_session("test_session")
    assert loaded is None


@pytest.mark.asyncio
async def test_list_sessions(storage):
    """Test listing sessions."""
    session1 = ChatSession(session_id="session1", user_id="user1")
    session2 = ChatSession(session_id="session2", user_id="user2")
    
    await storage.save_session(session1)
    await storage.save_session(session2)
    
    sessions = await storage.list_sessions()
    
    assert len(sessions) >= 2
    assert any(s.session_id == "session1" for s in sessions)
    assert any(s.session_id == "session2" for s in sessions)


