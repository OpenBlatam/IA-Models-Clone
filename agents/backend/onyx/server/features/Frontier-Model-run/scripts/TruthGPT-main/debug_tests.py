"""
Debug Script for TruthGPT Tests
Checks Python installation, imports, dependencies, and test readiness
"""

import sys
import os
from pathlib import Path

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def check_python_version():
    """Check Python version"""
    print_section("Python Version Check")
    version = sys.version_info
    print(f"  Python Version: {version.major}.{version.minor}.{version.micro}")
    print(f"  Version String: {sys.version.split()[0]}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("  ⚠ WARNING: Python 3.7+ required")
        return False
    else:
        print("  ✓ Python version is compatible")
        return True

def check_imports():
    """Check if core modules can be imported"""
    print_section("Import Check")
    
    # Add current directory to path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    imports_to_check = [
        ("unittest", "Standard library"),
        ("torch", "PyTorch"),
        ("torch.nn", "PyTorch NN"),
        ("core", "TruthGPT Core"),
        ("core.optimization", "Optimization Engine"),
        ("core.models", "Model Manager"),
        ("core.training", "Training Manager"),
        ("core.inference", "Inference Engine"),
        ("core.monitoring", "Monitoring System"),
    ]
    
    results = {}
    for module_name, description in imports_to_check:
        try:
            if module_name == "core":
                __import__(module_name)
                print(f"  ✓ {module_name:25} - {description}")
                results[module_name] = True
            else:
                module = __import__(module_name, fromlist=[''])
                print(f"  ✓ {module_name:25} - {description}")
                results[module_name] = True
        except ImportError as e:
            print(f"  ✗ {module_name:25} - {description} - ERROR: {e}")
            results[module_name] = False
        except Exception as e:
            print(f"  ✗ {module_name:25} - {description} - ERROR: {e}")
            results[module_name] = False
    
    return results

def check_core_exports():
    """Check if core module exports required classes"""
    print_section("Core Module Exports Check")
    
    try:
        from core import (
            OptimizationEngine, OptimizationConfig, OptimizationLevel,
            ModelManager, ModelConfig, ModelType,
            TrainingManager, TrainingConfig,
            InferenceEngine, InferenceConfig,
            MonitoringSystem
        )
        
        exports = [
            "OptimizationEngine", "OptimizationConfig", "OptimizationLevel",
            "ModelManager", "ModelConfig", "ModelType",
            "TrainingManager", "TrainingConfig",
            "InferenceEngine", "InferenceConfig",
            "MonitoringSystem"
        ]
        
        for export in exports:
            if export in dir():
                print(f"  ✓ {export}")
            else:
                print(f"  ✗ {export} - NOT FOUND")
                return False
        
        print("\n  All core exports available ✓")
        return True
        
    except ImportError as e:
        print(f"  ✗ Failed to import from core: {e}")
        return False

def check_test_files():
    """Check if test files exist and can be imported"""
    print_section("Test Files Check")
    
    project_root = Path(__file__).parent
    tests_dir = project_root / "tests"
    
    test_files = [
        "test_core.py",
        "test_optimization.py",
        "test_models.py",
        "test_training.py",
        "test_inference.py",
        "test_monitoring.py",
        "test_integration.py",
    ]
    
    results = {}
    for test_file in test_files:
        test_path = tests_dir / test_file
        if test_path.exists():
            try:
                # Try to import the test module
                module_name = f"tests.{test_file[:-3]}"
                __import__(module_name)
                print(f"  ✓ {test_file:25} - Found and importable")
                results[test_file] = True
            except Exception as e:
                print(f"  ✗ {test_file:25} - Found but import error: {e}")
                results[test_file] = False
        else:
            print(f"  ✗ {test_file:25} - NOT FOUND")
            results[test_file] = False
    
    return results

def check_test_runner():
    """Check if test runner can be imported"""
    print_section("Test Runner Check")
    
    project_root = Path(__file__).parent
    runner_file = project_root / "run_unified_tests.py"
    
    if runner_file.exists():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("run_unified_tests", runner_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if UnifiedTestRunner exists
            if hasattr(module, 'UnifiedTestRunner'):
                print("  ✓ run_unified_tests.py - Found and valid")
                print("  ✓ UnifiedTestRunner class exists")
                return True
            else:
                print("  ✗ UnifiedTestRunner class not found")
                return False
        except Exception as e:
            print(f"  ✗ Error loading test runner: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("  ✗ run_unified_tests.py - NOT FOUND")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print_section("Dependencies Check")
    
    required_packages = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "psutil": "PSUtil (optional)",
    }
    
    results = {}
    for package, description in required_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {package:15} - {description:20} - Version: {version}")
            results[package] = True
        except ImportError:
            optional = "(optional)" in description
            status = "⚠" if optional else "✗"
            print(f"  {status} {package:15} - {description:20} - NOT INSTALLED")
            results[package] = optional
    
    return results

def check_unittest_compatibility():
    """Check unittest compatibility"""
    print_section("Unittest Compatibility Check")
    
    try:
        import unittest
        
        # Check if makeSuite exists (deprecated)
        if hasattr(unittest, 'makeSuite'):
            print("  ⚠ unittest.makeSuite() is available (deprecated)")
        else:
            print("  ✓ unittest.makeSuite() not available (expected in Python 3.13+)")
        
        # Check if TestLoader exists
        if hasattr(unittest, 'TestLoader'):
            loader = unittest.TestLoader()
            print("  ✓ unittest.TestLoader() is available")
            
            # Verify loadTestsFromTestCase exists
            if hasattr(loader, 'loadTestsFromTestCase'):
                print("  ✓ loadTestsFromTestCase() method available")
                return True
            else:
                print("  ✗ loadTestsFromTestCase() method not found")
                return False
        else:
            print("  ✗ unittest.TestLoader() not available")
            return False
            
    except Exception as e:
        print(f"  ✗ Error checking unittest: {e}")
        return False

def main():
    """Main debug function"""
    print("="*70)
    print("  TruthGPT Test Debug Tool")
    print("="*70)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check dependencies
    deps_results = check_dependencies()
    
    # Check unittest compatibility
    unittest_ok = check_unittest_compatibility()
    
    # Check imports
    import_results = check_imports()
    
    # Check core exports
    if import_results.get("core", False):
        core_exports_ok = check_core_exports()
    else:
        core_exports_ok = False
        print_section("Core Module Exports Check")
        print("  ⚠ Skipped - core module not importable")
    
    # Check test files
    test_files_results = check_test_files()
    
    # Check test runner
    runner_ok = check_test_runner()
    
    # Summary
    print_section("Debug Summary")
    
    print(f"\n  Python Version:     {'✓ OK' if python_ok else '✗ FAIL'}")
    print(f"  Dependencies:       {sum(deps_results.values())}/{len(deps_results)} installed")
    print(f"  Unittest:           {'✓ Compatible' if unittest_ok else '✗ Issue'}")
    print(f"  Core Imports:        {sum(import_results.values())}/{len(import_results)} OK")
    print(f"  Core Exports:        {'✓ OK' if core_exports_ok else '✗ FAIL'}")
    print(f"  Test Files:          {sum(test_files_results.values())}/{len(test_files_results)} OK")
    print(f"  Test Runner:         {'✓ OK' if runner_ok else '✗ FAIL'}")
    
    # Overall status
    all_ok = (
        python_ok and 
        unittest_ok and 
        import_results.get("core", False) and 
        core_exports_ok and 
        all(test_files_results.values()) and 
        runner_ok
    )
    
    print("\n" + "="*70)
    if all_ok:
        print("  ✅ ALL CHECKS PASSED - Tests should run successfully")
        print("="*70)
        print("\n  To run tests:")
        print("    python run_unified_tests.py")
        return 0
    else:
        print("  ⚠ SOME ISSUES FOUND - Review the output above")
        print("="*70)
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)









