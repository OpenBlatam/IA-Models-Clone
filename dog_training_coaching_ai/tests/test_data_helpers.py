"""
Tests for Data Helpers
======================
"""

import pytest
from datetime import datetime
from ...utils.data_helpers import (
    merge_dicts,
    filter_none_values,
    chunk_list,
    calculate_average,
    format_duration,
    parse_date_range
)


def test_merge_dicts():
    """Test merging dictionaries."""
    dict1 = {"a": 1, "b": 2}
    dict2 = {"c": 3, "d": 4}
    result = merge_dicts(dict1, dict2)
    assert result == {"a": 1, "b": 2, "c": 3, "d": 4}


def test_filter_none_values():
    """Test filtering None values."""
    data = {"a": 1, "b": None, "c": "test", "d": None}
    result = filter_none_values(data)
    assert result == {"a": 1, "c": "test"}


def test_chunk_list():
    """Test chunking list."""
    items = [1, 2, 3, 4, 5, 6, 7]
    result = chunk_list(items, 3)
    assert result == [[1, 2, 3], [4, 5, 6], [7]]


def test_calculate_average():
    """Test calculating average."""
    assert calculate_average([1, 2, 3, 4, 5]) == 3.0
    assert calculate_average([]) is None


def test_format_duration():
    """Test formatting duration."""
    assert format_duration(30) == "30s"
    assert format_duration(90) == "1m 30s"
    assert format_duration(3661) == "1h 1m 1s"


def test_parse_date_range():
    """Test parsing date range."""
    start, end = parse_date_range("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")
    assert isinstance(start, datetime)
    assert isinstance(end, datetime)

