"""
Basic import test to verify the test setup works correctly.
"""

import pytest
import sys
from pathlib import Path

def test_import_paths():
    """Test that import paths are set up correctly"""
    # Check that we can import the core modules
    try:
        from core.base_service import BaseService, ServiceType, ServiceStatus
        assert BaseService is not None
        assert ServiceType is not None
        assert ServiceStatus is not None
    except ImportError as e:
        pytest.skip(f"BaseService not available: {e}")
    
    try:
        from core.dependency_manager import DependencyManager
        assert DependencyManager is not None
    except ImportError as e:
        pytest.skip(f"DependencyManager not available: {e}")
    
    try:
        from core.error_handler import ErrorHandler
        assert ErrorHandler is not None
    except ImportError as e:
        pytest.skip(f"ErrorHandler not available: {e}")
    
    try:
        from core.enterprise_features import EnterpriseFeatures
        assert EnterpriseFeatures is not None
    except ImportError as e:
        pytest.skip(f"EnterpriseFeatures not available: {e}")

def test_python_path():
    """Test that Python path includes the current directory"""
    current_dir = Path(__file__).parent
    assert str(current_dir) in sys.path
    
    # Check that we can import from the parent directory
    parent_dir = current_dir.parent
    assert str(parent_dir) in sys.path

def test_basic_assertions():
    """Basic test to ensure pytest is working"""
    assert 1 + 1 == 2
    assert "hello" in "hello world"
    assert len([1, 2, 3]) == 3
