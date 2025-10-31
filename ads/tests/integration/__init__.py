"""
Integration Tests package for the ads feature.

This package contains all integration tests for component interactions:
- API integration tests (end-to-end API testing)
- Service integration tests (service layer interactions)
- Database integration tests (database operations and persistence)
"""

from . import test_api_integration
from . import test_service_integration
from . import test_database_integration

__all__ = [
    "test_api_integration",
    "test_service_integration",
    "test_database_integration"
]
