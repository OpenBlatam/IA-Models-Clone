"""
Helpers de test refactorizados y mejorados
"""

from .test_base_classes import (
    BaseAPITestCase,
    BaseServiceTestCase,
    BaseRouteTestMixin
)

from .test_common_patterns import (
    mock_dependencies,
    create_router_client,
    create_service_mock,
    assert_standard_response,
    assert_paginated_response,
    assert_error_response,
    create_test_data_factory
)

from .test_assertion_helpers_improved import (
    assert_response_structure,
    assert_response_status,
    assert_valid_uuid,
    assert_valid_timestamp,
    assert_audio_response,
    assert_pagination_response,
    assert_audio_data_valid,
    assert_array_shape,
    assert_array_dtype,
    assert_file_exists,
    assert_file_size_valid
)

from .test_mock_helpers import (
    create_mock_service,
    create_async_mock_service,
    create_test_client_with_mocks,
    create_sample_audio_data,
    create_sample_audio_file,
    create_mock_user,
    create_mock_song,
    create_mock_playlist
)

from .test_data_factories import (
    generate_song_data,
    generate_playlist_data,
    generate_user_data,
    generate_analytics_event,
    generate_batch_data,
    generate_multiple_songs,
    generate_multiple_playlists
)

from .test_refactored_patterns import (
    RefactoredTestClient,
    create_refactored_client,
    mock_service_context,
    mock_multiple_services,
    create_standard_mock_service,
    assert_standard_api_response,
    assert_list_response_structure,
    create_test_audio_data,
    create_test_audio_file_bytes,
    StandardTestMixin
)

from .test_test_utilities import (
    TestConfig,
    TestClientBuilder,
    create_test_client_builder,
    retry_on_failure,
    parametrize_http_methods,
    skip_if_service_unavailable,
    TestDataGenerator
)

__all__ = [
    # Base classes
    "BaseAPITestCase",
    "BaseServiceTestCase",
    "BaseRouteTestMixin",
    
    # Common patterns
    "mock_dependencies",
    "create_router_client",
    "create_service_mock",
    "assert_standard_response",
    "assert_paginated_response",
    "assert_error_response",
    "create_test_data_factory",
    
    # Assertion helpers
    "assert_response_structure",
    "assert_response_status",
    "assert_valid_uuid",
    "assert_valid_timestamp",
    "assert_audio_response",
    "assert_pagination_response",
    "assert_audio_data_valid",
    "assert_array_shape",
    "assert_array_dtype",
    "assert_file_exists",
    "assert_file_size_valid",
    
    # Mock helpers
    "create_mock_service",
    "create_async_mock_service",
    "create_test_client_with_mocks",
    "create_sample_audio_data",
    "create_sample_audio_file",
    "create_mock_user",
    "create_mock_song",
    "create_mock_playlist",
    
    # Data factories
    "generate_song_data",
    "generate_playlist_data",
    "generate_user_data",
    "generate_analytics_event",
    "generate_batch_data",
    "generate_multiple_songs",
    "generate_multiple_playlists",
    
    # Refactored patterns
    "RefactoredTestClient",
    "create_refactored_client",
    "mock_service_context",
    "mock_multiple_services",
    "create_standard_mock_service",
    "assert_standard_api_response",
    "assert_list_response_structure",
    "create_test_audio_data",
    "create_test_audio_file_bytes",
    "StandardTestMixin",
    
    # Test utilities
    "TestConfig",
    "TestClientBuilder",
    "create_test_client_builder",
    "retry_on_failure",
    "parametrize_http_methods",
    "skip_if_service_unavailable",
    "TestDataGenerator",
]

