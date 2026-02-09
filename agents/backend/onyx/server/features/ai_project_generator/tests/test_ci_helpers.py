"""
CI/CD helpers for tests
"""

import pytest
import os
from typing import Dict, Any, Optional


class CIHelpers:
    """Helpers for CI/CD environments"""
    
    @staticmethod
    def is_ci() -> bool:
        """Check if running in CI environment"""
        ci_vars = [
            "CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS",
            "GITLAB_CI", "JENKINS_URL", "TRAVIS", "CIRCLECI"
        ]
        return any(os.getenv(var) for var in ci_vars)
    
    @staticmethod
    def get_ci_platform() -> Optional[str]:
        """Get CI platform name"""
        if os.getenv("GITHUB_ACTIONS"):
            return "github"
        elif os.getenv("GITLAB_CI"):
            return "gitlab"
        elif os.getenv("JENKINS_URL"):
            return "jenkins"
        elif os.getenv("TRAVIS"):
            return "travis"
        elif os.getenv("CIRCLECI"):
            return "circleci"
        return None
    
    @staticmethod
    def should_skip_slow_tests() -> bool:
        """Check if slow tests should be skipped"""
        return CIHelpers.is_ci() and os.getenv("SKIP_SLOW_TESTS", "false").lower() == "true"
    
    @staticmethod
    def get_test_timeout() -> float:
        """Get test timeout based on environment"""
        if CIHelpers.is_ci():
            return float(os.getenv("TEST_TIMEOUT", "300"))  # 5 minutes default
        return float(os.getenv("TEST_TIMEOUT", "600"))  # 10 minutes default


@pytest.fixture
def ci_helpers():
    """Fixture for CI helpers"""
    return CIHelpers


@pytest.fixture(autouse=True)
def skip_slow_in_ci(request):
    """Automatically skip slow tests in CI if configured"""
    if CIHelpers.should_skip_slow_tests():
        if "slow" in [mark.name for mark in request.node.iter_markers()]:
            pytest.skip("Slow tests skipped in CI")

