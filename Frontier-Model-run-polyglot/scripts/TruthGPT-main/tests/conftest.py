"""
Pytest Configuration
Shared fixtures and configuration for pytest (if used)
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Pytest fixtures
@pytest.fixture(scope="session")
def project_root_path():
    """Return project root path"""
    return project_root

@pytest.fixture(scope="session")
def test_data_dir():
    """Return test data directory"""
    return project_root / "tests" / "data"

@pytest.fixture(scope="function")
def temp_dir(tmp_path):
    """Return temporary directory for tests"""
    return tmp_path








