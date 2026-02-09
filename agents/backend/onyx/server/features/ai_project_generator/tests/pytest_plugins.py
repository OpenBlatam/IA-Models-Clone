"""
Pytest plugins and hooks for enhanced testing
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any
import time


def pytest_configure(config):
    """Configure pytest with custom settings"""
    # Register custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "async: marks tests as async tests")
    config.addinivalue_line("markers", "security: marks tests as security tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "smoke: marks tests as smoke tests")
    config.addinivalue_line("markers", "regression: marks tests as regression tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names"""
    for item in items:
        # Auto-mark async tests
        if "async" in item.name.lower() or "asyncio" in str(item.fspath):
            item.add_marker(pytest.mark.async)
        
        # Auto-mark slow tests
        if "slow" in item.name.lower() or "stress" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Auto-mark integration tests
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        # Auto-mark unit tests
        if "test_" in item.name and "integration" not in item.name.lower():
            item.add_marker(pytest.mark.unit)


def pytest_runtest_setup(item):
    """Setup hook before each test"""
    # Add any setup logic here
    pass


def pytest_runtest_teardown(item):
    """Teardown hook after each test"""
    # Add any cleanup logic here
    pass


def pytest_sessionstart(session):
    """Called after the Session object has been created"""
    print("\n" + "="*70)
    print("Starting AI Project Generator Test Suite")
    print("="*70)


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished"""
    print("\n" + "="*70)
    print("Test Suite Execution Complete")
    print(f"Exit Status: {exitstatus}")
    print("="*70)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Make test result available in fixtures"""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)


@pytest.fixture(scope="session", autouse=True)
def test_session_info():
    """Session-level fixture to track test session info"""
    session_info = {
        "start_time": time.time(),
        "test_count": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0
    }
    yield session_info
    session_info["end_time"] = time.time()
    session_info["duration"] = session_info["end_time"] - session_info["start_time"]


class TestReporter:
    """Custom test reporter for better output"""
    
    @staticmethod
    def report_test_start(test_name: str):
        """Report test start"""
        print(f"\n▶ Running: {test_name}")
    
    @staticmethod
    def report_test_success(test_name: str, duration: float):
        """Report test success"""
        print(f"✓ PASSED: {test_name} ({duration:.2f}s)")
    
    @staticmethod
    def report_test_failure(test_name: str, error: str):
        """Report test failure"""
        print(f"✗ FAILED: {test_name}")
        print(f"  Error: {error}")
    
    @staticmethod
    def report_test_skip(test_name: str, reason: str):
        """Report test skip"""
        print(f"⊘ SKIPPED: {test_name} - {reason}")

