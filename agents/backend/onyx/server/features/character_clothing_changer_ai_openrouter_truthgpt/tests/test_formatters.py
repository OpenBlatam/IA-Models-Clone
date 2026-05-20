"""
Tests for formatters
"""

import pytest
from utils.formatters import (
    format_response,
    format_error,
    format_duration,
    format_file_size,
    format_percentage,
    format_json_safe
)


def test_format_response():
    """Test format_response"""
    result = format_response(
        data={"key": "value"},
        success=True,
        message="Success"
    )
    assert result["success"] is True
    assert result["data"] == {"key": "value"}
    assert result["message"] == "Success"
    assert "timestamp" in result


def test_format_error():
    """Test format_error"""
    result = format_error(
        error="Test error",
        error_code="TEST_ERROR"
    )
    assert result["success"] is False
    assert result["error"] == "Test error"
    assert result["error_code"] == "TEST_ERROR"
    assert "timestamp" in result


def test_format_duration():
    """Test format_duration"""
    assert format_duration(45.5) == "45.50s"
    assert "m" in format_duration(125.5)
    assert "h" in format_duration(3661.5)


def test_format_file_size():
    """Test format_file_size"""
    assert "KB" in format_file_size(1024)
    assert "MB" in format_file_size(1048576)
    assert "GB" in format_file_size(1073741824)


def test_format_percentage():
    """Test format_percentage"""
    result = format_percentage(50, 100)
    assert result == "50.00%"
    
    result = format_percentage(0, 100)
    assert result == "0.00%"


def test_format_json_safe():
    """Test format_json_safe"""
    from decimal import Decimal
    from datetime import datetime
    
    data = {
        "decimal": Decimal("10.5"),
        "datetime": datetime.now(),
        "normal": "value"
    }
    
    result = format_json_safe(data)
    assert isinstance(result["decimal"], float)
    assert isinstance(result["datetime"], str)
    assert result["normal"] == "value"

