"""
Pytest Configuration
===================
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def reset_event_loop(event_loop):
    """Reset event loop for each test."""
    yield
    event_loop.close()
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)


