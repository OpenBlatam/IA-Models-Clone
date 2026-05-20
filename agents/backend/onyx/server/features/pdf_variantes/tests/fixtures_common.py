"""
Common Fixtures for Playwright Tests
====================================
Centralized fixtures to avoid duplication across test files.
"""

import pytest
import os
from typing import Dict, Any


@pytest.fixture
def api_base_url():
    """API base URL from environment or default."""
    return os.getenv("API_BASE_URL", "http://localhost:8000")


@pytest.fixture
def auth_headers():
    """Standard authentication headers."""
    return {
        "Authorization": "Bearer test_token_123",
        "X-User-ID": "test_user_123",
        "Content-Type": "application/json"
    }


@pytest.fixture
def sample_pdf():
    """Sample PDF content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\nxref\n0 2\ntrailer\n<<\n/Size 2\n>>\nstartxref\n20\n%%EOF"


@pytest.fixture
def test_data():
    """Common test data."""
    return {
        "valid_file_ids": ["file_1", "file_2", "file_3"],
        "invalid_file_ids": ["", "nonexistent", "invalid_id_123"],
        "variant_types": ["summary", "outline", "highlights", "notes", "quiz", "presentation"],
        "invalid_variant_types": ["invalid", "unknown", ""],
        "valid_options": {
            "max_length": 500,
            "style": "academic",
            "language": "en"
        },
        "invalid_options": {
            "max_length": -1,
            "style": "",
            "language": "invalid"
        }
    }


@pytest.fixture
def ci_timeout():
    """Timeout for CI environments."""
    return int(os.getenv("CI_TIMEOUT", "30000"))


@pytest.fixture
def performance_thresholds():
    """Performance thresholds for tests."""
    return {
        "response_time_avg": 1.0,
        "response_time_p95": 2.0,
        "response_time_p99": 3.0,
        "throughput_min": 5.0,
        "memory_increase_max": 100.0
    }



