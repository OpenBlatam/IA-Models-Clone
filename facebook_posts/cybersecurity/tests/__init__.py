from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import pytest
import asyncio
from typing import Dict, Any, Optional
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Cybersecurity Toolkit Tests
Automated testing with pytest and pytest-asyncio for edge cases and network layer mocking.
"""

__version__ = "1.0.0"
__author__ = "Cybersecurity Toolkit Team"

# Test configuration

# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]

# Global test configuration
TEST_CONFIG = {
    "timeout": 5.0,
    "max_workers": 10,
    "retry_attempts": 2,
    "chunk_size": 100,
    "enable_mocking": True,
    "mock_network_delay": 0.1,
    "mock_network_failure_rate": 0.1
}

# Test fixtures and utilities
def get_test_config() -> Dict[str, Any]:
    """Get test configuration."""
    return TEST_CONFIG.copy()

def create_mock_network_response(success: bool = True, delay: float = 0.1, 
                               data: Optional[Any] = None, error: Optional[str] = None) -> Dict[str, Any]:
    """Create a mock network response for testing."""
    return {
        "success": success,
        "data": data,
        "error": error,
        "delay": delay,
        "timestamp": asyncio.get_event_loop().time()
    }

# Export test utilities
__all__ = [
    "TEST_CONFIG",
    "get_test_config", 
    "create_mock_network_response"
] 