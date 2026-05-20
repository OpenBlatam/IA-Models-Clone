"""
Pytest Plugins
Custom pytest plugins and hooks
"""

import pytest
from typing import Dict, Any


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "performance: Performance tests"
    )
    config.addinivalue_line(
        "markers", "security: Security tests"
    )
    config.addinivalue_line(
        "markers", "api: API endpoint tests"
    )
    config.addinivalue_line(
        "markers", "async: Async tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names"""
    for item in items:
        # Add async marker for async tests
        if "async" in item.name.lower() or "asyncio" in item.name.lower():
            item.add_marker(pytest.mark.asyncio)
        
        # Add unit marker for unit tests
        if "test_" in item.name and "integration" not in item.name.lower():
            if "test_integration" not in item.nodeid:
                item.add_marker(pytest.mark.unit)
        
        # Add integration marker
        if "integration" in item.name.lower() or "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add API marker
        if "api" in item.name.lower() or "router" in item.name.lower() or "endpoint" in item.name.lower():
            item.add_marker(pytest.mark.api)
        
        # Add performance marker
        if "performance" in item.name.lower() or "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        
        # Add security marker
        if "security" in item.name.lower() or "auth" in item.name.lower():
            item.add_marker(pytest.mark.security)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before all tests"""
    # Setup code here
    yield
    # Teardown code here


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset mocks before each test"""
    yield
    # Cleanup after test


class TestSession:
    """Test session information"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
    
    def start(self):
        """Start test session"""
        from datetime import datetime
        self.start_time = datetime.utcnow()
    
    def end(self):
        """End test session"""
        from datetime import datetime
        self.end_time = datetime.utcnow()
    
    def get_duration(self):
        """Get session duration"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0


@pytest.fixture(scope="session")
def test_session():
    """Test session fixture"""
    session = TestSession()
    session.start()
    yield session
    session.end()



