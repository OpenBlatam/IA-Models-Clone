"""
Tests for Search Engine
========================
"""

import pytest
from ..core.search_engine import SearchEngine


@pytest.fixture
def search_engine():
    """Create search engine for testing."""
    return SearchEngine()


@pytest.mark.asyncio
async def test_index_document(search_engine):
    """Test indexing a document."""
    doc_id = await search_engine.index(
        doc_id="doc1",
        content="This is a test document about Python programming",
        metadata={"category": "programming", "language": "en"}
    )
    
    assert doc_id == "doc1"
    assert "doc1" in search_engine.indexed_documents


@pytest.mark.asyncio
async def test_search_documents(search_engine):
    """Test searching documents."""
    await search_engine.index("doc1", "Python programming tutorial")
    await search_engine.index("doc2", "JavaScript tutorial")
    await search_engine.index("doc3", "Python advanced topics")
    
    results = await search_engine.search("Python", limit=10)
    
    assert len(results) >= 2
    assert any(r.doc_id == "doc1" for r in results)
    assert any(r.doc_id == "doc3" for r in results)


@pytest.mark.asyncio
async def test_search_with_filters(search_engine):
    """Test searching with metadata filters."""
    await search_engine.index(
        "doc1",
        "Python tutorial",
        metadata={"category": "beginner", "language": "en"}
    )
    await search_engine.index(
        "doc2",
        "Python advanced",
        metadata={"category": "advanced", "language": "en"}
    )
    
    results = await search_engine.search(
        "Python",
        filters={"category": "beginner"},
        limit=10
    )
    
    assert len(results) >= 1
    assert all(r.metadata.get("category") == "beginner" for r in results)


@pytest.mark.asyncio
async def test_delete_document(search_engine):
    """Test deleting a document from index."""
    await search_engine.index("doc1", "Test content")
    
    assert "doc1" in search_engine.indexed_documents
    
    await search_engine.delete("doc1")
    
    assert "doc1" not in search_engine.indexed_documents


@pytest.mark.asyncio
async def test_get_index_stats(search_engine):
    """Test getting index statistics."""
    await search_engine.index("doc1", "Content 1")
    await search_engine.index("doc2", "Content 2")
    
    stats = search_engine.get_index_stats()
    
    assert stats["total_documents"] >= 2


