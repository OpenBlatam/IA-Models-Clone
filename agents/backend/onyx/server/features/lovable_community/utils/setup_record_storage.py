"""
Setup Script for Record Storage

This script helps set up and verify the RecordStorage implementation.
Run this to ensure everything is properly configured.
"""

import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all required modules are available."""
    print("Checking requirements...")
    
    required_modules = ['json', 'logging', 'pathlib', 'typing']
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module} - MISSING")
            missing.append(module)
    
    if missing:
        print(f"\n❌ Missing required modules: {', '.join(missing)}")
        return False
    
    print("\n✅ All required modules available")
    return True

def check_implementation():
    """Check if RecordStorage is properly implemented."""
    print("\nChecking implementation...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils.record_storage import RecordStorage
        
        print("  ✅ RecordStorage class imported successfully")
        
        required_methods = ['read', 'write', 'update', 'add', 'delete', 'get']
        missing_methods = []
        
        for method in required_methods:
            if hasattr(RecordStorage, method):
                print(f"  ✅ Method '{method}' exists")
            else:
                print(f"  ❌ Method '{method}' - MISSING")
                missing_methods.append(method)
        
        if missing_methods:
            print(f"\n❌ Missing methods: {', '.join(missing_methods)}")
            return False
        
        print("\n✅ All required methods present")
        return True
        
    except ImportError as e:
        print(f"\n❌ Failed to import RecordStorage: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error checking implementation: {e}")
        return False

def check_context_managers():
    """Verify context managers are used."""
    print("\nChecking context managers...")
    
    try:
        import inspect
        from utils.record_storage import RecordStorage
        
        source = inspect.getsource(RecordStorage)
        
        if 'with open(' in source:
            print("  ✅ Context managers found in code")
        else:
            print("  ❌ Context managers not found")
            return False
        
        if 'f.close()' in source:
            print("  ⚠️  Manual close() found - should use context managers")
            return False
        
        print("\n✅ Context managers properly implemented")
        return True
        
    except Exception as e:
        print(f"\n❌ Error checking context managers: {e}")
        return False

def check_error_handling():
    """Verify error handling is present."""
    print("\nChecking error handling...")
    
    try:
        import inspect
        from utils.record_storage import RecordStorage
        
        source = inspect.getsource(RecordStorage)
        
        methods_to_check = ['read', 'write', 'update']
        all_good = True
        
        for method in methods_to_check:
            method_source = inspect.getsource(getattr(RecordStorage, method))
            if 'try:' in method_source and 'except' in method_source:
                print(f"  ✅ {method}() has error handling")
            else:
                print(f"  ❌ {method}() missing error handling")
                all_good = False
        
        if not all_good:
            return False
        
        print("\n✅ Error handling present in all methods")
        return True
        
    except Exception as e:
        print(f"\n❌ Error checking error handling: {e}")
        return False

def create_test_directory():
    """Create test directory for examples."""
    print("\nSetting up test directory...")
    
    test_dir = Path("test_data")
    try:
        test_dir.mkdir(exist_ok=True)
        print(f"  ✅ Created/verified directory: {test_dir}")
        return True
    except Exception as e:
        print(f"  ❌ Failed to create directory: {e}")
        return False

def run_quick_test():
    """Run a quick functionality test."""
    print("\nRunning quick functionality test...")
    
    try:
        from utils.record_storage import RecordStorage
        
        test_file = "test_data/setup_test.json"
        storage = RecordStorage(test_file)
        
        storage.write([{"id": "1", "name": "Test"}])
        records = storage.read()
        
        if len(records) == 1 and records[0]["name"] == "Test":
            print("  ✅ Basic functionality test passed")
            
            storage.update("1", {"age": 30})
            updated = storage.get("1")
            
            if updated["name"] == "Test" and updated["age"] == 30:
                print("  ✅ Update merging test passed")
                return True
            else:
                print("  ❌ Update merging test failed")
                return False
        else:
            print("  ❌ Basic functionality test failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Quick test failed: {e}")
        return False

def main():
    """Run all setup checks."""
    print("=" * 70)
    print("  Record Storage - Setup and Verification")
    print("=" * 70)
    
    checks = [
        ("Requirements", check_requirements),
        ("Implementation", check_implementation),
        ("Context Managers", check_context_managers),
        ("Error Handling", check_error_handling),
        ("Test Directory", create_test_directory),
        ("Quick Test", run_quick_test),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} check failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 70)
    print("  Setup Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {name}")
    
    print(f"\n  Total: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✅ Setup complete! RecordStorage is ready to use.")
        return 0
    else:
        print("\n❌ Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())


