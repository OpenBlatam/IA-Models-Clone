"""
Testing Module

Provides:
- Testing utilities
- Test fixtures
- Mock utilities
"""

from .test_utils import (
    create_test_fixture,
    mock_model,
    mock_audio,
    assert_audio_valid
)

from .test_runner import (
    TestRunner,
    run_tests,
    create_test_suite
)

__all__ = [
    # Test utilities
    "create_test_fixture",
    "mock_model",
    "mock_audio",
    "assert_audio_valid",
    # Test runner
    "TestRunner",
    "run_tests",
    "create_test_suite"
]
