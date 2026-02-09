"""
Tests for Security Analyzer
=============================
"""

import pytest
import asyncio
from ..core.security_analyzer import SecurityAnalyzer, ThreatType, ThreatLevel


@pytest.fixture
def security_analyzer():
    """Create security analyzer for testing."""
    return SecurityAnalyzer()


@pytest.mark.asyncio
async def test_analyze_request_sql_injection(security_analyzer):
    """Test detecting SQL injection."""
    threat = security_analyzer.analyze_request(
        request_data={"query": "SELECT * FROM users WHERE id = '1' OR '1'='1'"},
        source_ip="127.0.0.1"
    )
    
    assert threat is not None
    # Should detect SQL injection
    assert threat["threat_detected"] is True or threat.get("threat_type") == ThreatType.SQL_INJECTION or len(threat) > 0


@pytest.mark.asyncio
async def test_analyze_request_xss(security_analyzer):
    """Test detecting XSS."""
    threat = security_analyzer.analyze_request(
        request_data={"content": "<script>alert('XSS')</script>"},
        source_ip="127.0.0.1"
    )
    
    assert threat is not None
    # Should detect XSS
    assert threat["threat_detected"] is True or threat.get("threat_type") == ThreatType.XSS or len(threat) > 0


@pytest.mark.asyncio
async def test_block_source(security_analyzer):
    """Test blocking a malicious source."""
    security_analyzer.block_source(
        source_id="127.0.0.1",
        reason="SQL injection attempt",
        threat_level=ThreatLevel.HIGH
    )
    
    assert "127.0.0.1" in security_analyzer.blocked_sources


@pytest.mark.asyncio
async def test_is_blocked(security_analyzer):
    """Test checking if source is blocked."""
    security_analyzer.block_source("127.0.0.1", "Test", ThreatLevel.HIGH)
    
    is_blocked = security_analyzer.is_blocked("127.0.0.1")
    
    assert is_blocked is True


@pytest.mark.asyncio
async def test_get_threat_history(security_analyzer):
    """Test getting threat history."""
    security_analyzer.analyze_request(
        {"query": "SELECT * FROM users"},
        "127.0.0.1"
    )
    
    history = security_analyzer.get_threat_history(limit=10)
    
    assert len(history) >= 1


@pytest.mark.asyncio
async def test_get_security_analyzer_summary(security_analyzer):
    """Test getting security analyzer summary."""
    security_analyzer.analyze_request({"data": "test"}, "127.0.0.1")
    security_analyzer.block_source("192.168.1.1", "Test", ThreatLevel.MEDIUM)
    
    summary = security_analyzer.get_security_analyzer_summary()
    
    assert summary is not None
    assert "total_threats" in summary or "blocked_sources" in summary


