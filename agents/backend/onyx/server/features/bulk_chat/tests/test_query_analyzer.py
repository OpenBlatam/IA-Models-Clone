"""
Tests for Query Analyzer
=========================
"""

import pytest
import asyncio
from ..core.query_analyzer import QueryAnalyzer


@pytest.fixture
def query_analyzer():
    """Create query analyzer for testing."""
    return QueryAnalyzer()


@pytest.mark.asyncio
async def test_analyze_query(query_analyzer):
    """Test analyzing a query."""
    query_id = query_analyzer.analyze_query(
        query="SELECT * FROM users WHERE id = 1",
        execution_time=0.5,
        metadata={"database": "test_db"}
    )
    
    assert query_id is not None
    assert query_id in query_analyzer.queries


@pytest.mark.asyncio
async def test_get_slow_queries(query_analyzer):
    """Test getting slow queries."""
    query_analyzer.analyze_query("SELECT * FROM table1", 0.1)  # Fast
    query_analyzer.analyze_query("SELECT * FROM table2", 5.0)   # Slow
    query_analyzer.analyze_query("SELECT * FROM table3", 10.0)  # Very slow
    
    slow_queries = query_analyzer.get_slow_queries(threshold_seconds=1.0, limit=10)
    
    assert len(slow_queries) >= 2


@pytest.mark.asyncio
async def test_get_query_patterns(query_analyzer):
    """Test getting query patterns."""
    query_analyzer.analyze_query("SELECT * FROM users WHERE id = 1", 0.5)
    query_analyzer.analyze_query("SELECT * FROM users WHERE id = 2", 0.6)
    query_analyzer.analyze_query("SELECT * FROM users WHERE id = 3", 0.7)
    
    patterns = query_analyzer.get_query_patterns()
    
    assert patterns is not None
    assert isinstance(patterns, list) or isinstance(patterns, dict)


@pytest.mark.asyncio
async def test_get_query_statistics(query_analyzer):
    """Test getting query statistics."""
    query_analyzer.analyze_query("SELECT * FROM table1", 0.5)
    query_analyzer.analyze_query("SELECT * FROM table2", 1.0)
    query_analyzer.analyze_query("SELECT * FROM table3", 1.5)
    
    stats = query_analyzer.get_query_statistics()
    
    assert stats is not None
    assert "total_queries" in stats or "avg_execution_time" in stats or "slow_queries" in stats


@pytest.mark.asyncio
async def test_get_query_analyzer_summary(query_analyzer):
    """Test getting query analyzer summary."""
    query_analyzer.analyze_query("SELECT * FROM table1", 0.5)
    query_analyzer.analyze_query("SELECT * FROM table2", 2.0)
    
    summary = query_analyzer.get_query_analyzer_summary()
    
    assert summary is not None
    assert "total_queries" in summary or "slow_queries_count" in summary


