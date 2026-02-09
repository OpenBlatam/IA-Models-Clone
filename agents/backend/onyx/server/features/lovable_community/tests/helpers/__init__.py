"""
Helpers modulares para tests de Lovable Community

Provides test utilities organized into specific modules:
- test_helpers: Basic test data generation
- mock_helpers: Mock object creation
- assertion_helpers: Custom assertion functions
- advanced_helpers: Advanced testing utilities
- security_helpers: Security testing payloads
"""

from .test_helpers import (
    generate_chat_id,
    generate_user_id,
    create_chat_dict,
    create_publish_request,
    create_remix_request,
    create_vote_request,
    create_search_request,
)

from .mock_helpers import (
    create_mock_chat_service,
    create_mock_ranking_service,
    create_mock_db_session,
)

from .assertion_helpers import (
    assert_chat_response_valid,
    assert_chat_list_valid,
    assert_pagination_valid,
    assert_vote_response_valid,
    assert_remix_response_valid,
    assert_stats_valid,
)

from .advanced_helpers import (
    AsyncTestHelper,
    PerformanceHelper,
    DataFactory,
    MockVerifier,
    TestDataBuilder,
    SecurityTestHelper,
    BatchTestHelper,
)

from .security_helpers import (
    generate_sql_injection_payloads,
    generate_xss_payloads,
    generate_path_traversal_payloads,
    generate_command_injection_payloads,
    generate_large_inputs,
    generate_special_characters,
    sanitize_for_testing,
)

__all__ = [
    # Test Helpers
    "generate_chat_id",
    "generate_user_id",
    "create_chat_dict",
    "create_publish_request",
    "create_remix_request",
    "create_vote_request",
    "create_search_request",
    # Mock Helpers
    "create_mock_chat_service",
    "create_mock_ranking_service",
    "create_mock_db_session",
    # Assertion Helpers
    "assert_chat_response_valid",
    "assert_chat_list_valid",
    "assert_pagination_valid",
    "assert_vote_response_valid",
    "assert_remix_response_valid",
    "assert_stats_valid",
    # Advanced Helpers
    "AsyncTestHelper",
    "PerformanceHelper",
    "DataFactory",
    "MockVerifier",
    "TestDataBuilder",
    "SecurityTestHelper",
    "BatchTestHelper",
    # Security Helpers
    "generate_sql_injection_payloads",
    "generate_xss_payloads",
    "generate_path_traversal_payloads",
    "generate_command_injection_payloads",
    "generate_large_inputs",
    "generate_special_characters",
    "sanitize_for_testing",
]
