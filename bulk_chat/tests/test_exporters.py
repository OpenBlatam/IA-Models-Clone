"""
Tests for Conversation Exporters
=================================
"""

import pytest
import tempfile
import json
from pathlib import Path
from ..core.exporters import ConversationExporter
from ..core.chat_session import ChatSession, ChatMessage


@pytest.fixture
def exporter():
    """Create conversation exporter for testing."""
    return ConversationExporter()


@pytest.fixture
def sample_session():
    """Create sample session for testing."""
    session = ChatSession(
        session_id="test_session",
        user_id="test_user"
    )
    session.messages = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi there!"),
    ]
    return session


@pytest.mark.asyncio
async def test_export_json(exporter, sample_session):
    """Test exporting to JSON."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        await exporter.export_to_json(sample_session, temp_path)
        
        # Verify file exists and is valid JSON
        assert Path(temp_path).exists()
        with open(temp_path, 'r') as f:
            data = json.load(f)
            assert "session_id" in data or "messages" in data
    finally:
        Path(temp_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_export_markdown(exporter, sample_session):
    """Test exporting to Markdown."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        temp_path = f.name
    
    try:
        await exporter.export_to_markdown(sample_session, temp_path)
        
        # Verify file exists
        assert Path(temp_path).exists()
        content = Path(temp_path).read_text()
        assert len(content) > 0
    finally:
        Path(temp_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_export_csv(exporter, sample_session):
    """Test exporting to CSV."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
    
    try:
        await exporter.export_to_csv(sample_session, temp_path)
        
        # Verify file exists
        assert Path(temp_path).exists()
        content = Path(temp_path).read_text()
        assert len(content) > 0
    finally:
        Path(temp_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_export_html(exporter, sample_session):
    """Test exporting to HTML."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        temp_path = f.name
    
    try:
        await exporter.export_to_html(sample_session, temp_path)
        
        # Verify file exists and contains HTML
        assert Path(temp_path).exists()
        content = Path(temp_path).read_text()
        assert "<html" in content.lower() or "<!doctype" in content.lower()
    finally:
        Path(temp_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_export_txt(exporter, sample_session):
    """Test exporting to TXT."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_path = f.name
    
    try:
        await exporter.export_to_txt(sample_session, temp_path)
        
        # Verify file exists
        assert Path(temp_path).exists()
        content = Path(temp_path).read_text()
        assert len(content) > 0
    finally:
        Path(temp_path).unlink(missing_ok=True)


