#!/usr/bin/env python3
"""
Test Validation Script for HeyGen AI
===================================

Simple script to validate that all test imports work correctly
without requiring pytest to be installed.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test that all core modules can be imported"""
    print("Testing core module imports...")
    
    try:
        from core.base_service import BaseService, ServiceType, ServiceStatus, HealthCheckResult
        print("✓ core.base_service imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import core.base_service: {e}")
        return False
    
    try:
        from core.dependency_manager import DependencyManager, ServicePriority, ServiceInfo, ServiceLifecycle
        print("✓ core.dependency_manager imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import core.dependency_manager: {e}")
        return False
    
    try:
        from core.error_handler import ErrorHandler, with_error_handling, with_retry
        print("✓ core.error_handler imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import core.error_handler: {e}")
        return False
    
    try:
        from core.config_manager import ConfigurationManager, get_config
        print("✓ core.config_manager imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import core.config_manager: {e}")
        return False
    
    try:
        from core.logging_service import LoggingService, LoggingConfig
        print("✓ core.logging_service imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import core.logging_service: {e}")
        return False
    
    try:
        from core.enterprise_features import (
            EnterpriseFeatures, User, Role, Permission, AuditLog, 
            SSOConfig, ComplianceConfig
        )
        print("✓ core.enterprise_features imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import core.enterprise_features: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of core classes"""
    print("\nTesting basic functionality...")
    
    try:
        from core.base_service import ServiceStatus, ServiceType
        
        # Test enum values
        assert ServiceStatus.RUNNING.value == "running"
        assert ServiceType.CORE.value == "core"
        print("✓ ServiceStatus and ServiceType enums work correctly")
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False
    
    try:
        from core.dependency_manager import ServicePriority, ServiceInfo
        
        # Test ServiceInfo creation
        info = ServiceInfo(
            name="test-service",
            service_type="test",
            priority=ServicePriority.NORMAL,
            status=ServiceStatus.RUNNING
        )
        assert info.name == "test-service"
        print("✓ ServiceInfo creation works correctly")
        
    except Exception as e:
        print(f"✗ ServiceInfo test failed: {e}")
        return False
    
    try:
        from core.enterprise_features import User, Role, Permission
        
        # Test data structure creation
        user = User(
            user_id="test_id",
            username="testuser",
            email="test@example.com",
            full_name="Test User"
        )
        assert user.username == "testuser"
        print("✓ Enterprise data structures work correctly")
        
    except Exception as e:
        print(f"✗ Enterprise data structures test failed: {e}")
        return False
    
    return True

def test_enterprise_features_initialization():
    """Test EnterpriseFeatures initialization"""
    print("\nTesting EnterpriseFeatures initialization...")
    
    try:
        from core.enterprise_features import EnterpriseFeatures
        
        # Create instance
        features = EnterpriseFeatures()
        assert features is not None
        assert features.service_name == "EnterpriseFeatures"
        print("✓ EnterpriseFeatures initialization works correctly")
        
        # Test basic properties
        assert hasattr(features, 'users')
        assert hasattr(features, 'roles')
        assert hasattr(features, 'permissions')
        assert hasattr(features, 'enterprise_stats')
        print("✓ EnterpriseFeatures has required attributes")
        
    except Exception as e:
        print(f"✗ EnterpriseFeatures initialization test failed: {e}")
        return False
    
    return True

def main():
    """Main validation function"""
    print("HeyGen AI Test Validation")
    print("=" * 40)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    # Test enterprise features
    if not test_enterprise_features_initialization():
        all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All tests passed! The test fixes are working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())





