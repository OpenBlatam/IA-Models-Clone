"""
Debug API Test
==============
Test the debugging tools.
"""

import pytest
import requests
from pathlib import Path
from debug_api import APIDebugger


@pytest.fixture
def api_debugger():
    """Create API debugger instance."""
    return APIDebugger(base_url="http://localhost:8000")


@pytest.mark.debug
def test_health_check(api_debugger):
    """Test health check."""
    result = api_debugger.health_check()
    assert "status" in result or "error" in result


@pytest.mark.debug
def test_endpoint_get(api_debugger):
    """Test GET endpoint."""
    result = api_debugger.test_endpoint("GET", "/health")
    assert "status" in result or "error" in result


@pytest.mark.debug
def test_request_history(api_debugger):
    """Test request history."""
    api_debugger.test_endpoint("GET", "/health")
    history = api_debugger.get_request_history()
    assert len(history) > 0


@pytest.mark.debug
def test_save_history(api_debugger, tmp_path):
    """Test saving history."""
    api_debugger.test_endpoint("GET", "/health")
    history_file = tmp_path / "history.json"
    api_debugger.save_history(history_file)
    assert history_file.exists()


@pytest.mark.debug
def test_print_summary(api_debugger, capsys):
    """Test printing summary."""
    api_debugger.test_endpoint("GET", "/health")
    api_debugger.print_summary()
    captured = capsys.readouterr()
    assert "Request Summary" in captured.out



