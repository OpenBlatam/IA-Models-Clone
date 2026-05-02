"""
Test Verification Script
Verifies that all test files can be imported and basic structure is correct
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def verify_imports():
    """Verify all imports work"""
    print("=" * 60)
    print("TEST VERIFICATION")
    print("=" * 60)
    print()
    
    errors = []
    
    # Test core imports
    print("1. Testing core module imports...")
    try:
        from core import (
            OptimizationEngine, OptimizationConfig, OptimizationLevel,
            ModelManager, ModelConfig, ModelType,
            TrainingManager, TrainingConfig,
            InferenceEngine, InferenceConfig,
            MonitoringSystem, MetricsCollector,
            SystemMetrics, ModelMetrics, TrainingMetrics
        )
        print("   ✅ Core imports successful")
    except Exception as e:
        print(f"   ❌ Core imports failed: {e}")
        errors.append(f"Core imports: {e}")
    
    # Test test module imports
    print("\n2. Testing test module imports...")
    test_modules = [
        'tests.test_core',
        'tests.test_optimization',
        'tests.test_models',
        'tests.test_training',
        'tests.test_inference',
        'tests.test_monitoring',
        'tests.test_integration'
    ]
    
    for module_name in test_modules:
        try:
            __import__(module_name)
            print(f"   ✅ {module_name} imports successfully")
        except Exception as e:
            print(f"   ❌ {module_name} import failed: {e}")
            errors.append(f"{module_name}: {e}")
    
    # Test test class imports
    print("\n3. Testing test class imports...")
    try:
        from tests.test_core import TestCoreComponents
        from tests.test_optimization import TestOptimizationEngine
        from tests.test_models import TestModelManager
        from tests.test_training import TestTrainingManager
        from tests.test_inference import TestInferenceEngine
        from tests.test_monitoring import TestMonitoringSystem
        from tests.test_integration import TestIntegration
        print("   ✅ All test classes imported successfully")
    except Exception as e:
        print(f"   ❌ Test class imports failed: {e}")
        errors.append(f"Test classes: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if errors:
        print("❌ VERIFICATION FAILED")
        print(f"Found {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ VERIFICATION PASSED")
        print("\nAll imports successful. Tests are ready to run!")
        print("\nTo run tests:")
        print("  python run_unified_tests.py")
        return True

if __name__ == "__main__":
    success = verify_imports()
    sys.exit(0 if success else 1)









