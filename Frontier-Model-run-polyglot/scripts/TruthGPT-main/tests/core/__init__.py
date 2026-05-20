"""
Core Tests Package
Contains all unit and integration test files
"""

# Exports útiles para facilitar imports
from .fixtures.test_utils import (
    create_test_model,
    create_test_dataset,
    create_test_tokenizer,
    assert_model_valid,
    TestTimer
)

from .fixtures.test_helpers import (
    retry_on_failure,
    skip_if_no_cuda,
    performance_test,
    memory_profiler
)

__all__ = [
    # Test utilities
    'create_test_model',
    'create_test_dataset',
    'create_test_tokenizer',
    'assert_model_valid',
    'TestTimer',
    # Test helpers/decorators
    'retry_on_failure',
    'skip_if_no_cuda',
    'performance_test',
    'memory_profiler',
]

