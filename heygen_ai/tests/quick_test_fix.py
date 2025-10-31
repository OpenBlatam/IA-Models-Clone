"""
Quick test fix for HeyGen AI system.
Addresses immediate test issues and provides a working test runner.
"""

import sys
import os
import subprocess
from pathlib import Path

def find_python():
    """Find available Python executable."""
    candidates = [
        "python",
        "python3", 
        "py",
        r"C:\Users\USER\AppData\Local\Programs\Python\Python311\python.exe"
    ]
    
    for candidate in candidates:
        try:
            result = subprocess.run([candidate, "--version"], 
                                  capture_output=True, timeout=5)
            if result.returncode == 0:
                return candidate
        except:
            continue
    return None

def run_tests():
    """Run tests with proper Python executable."""
    python_exe = find_python()
    if not python_exe:
        print("[ERROR] No Python executable found")
        return False
    
    print(f"[OK] Using Python: {python_exe}")
    
    # Test basic imports first
    try:
        result = subprocess.run([
            python_exe, "-c", 
            "import sys; print('Python import test: OK')"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("[OK] Python import test passed")
        else:
            print(f"[ERROR] Python import test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Python test error: {e}")
        return False
    
    # Run pytest if available
    try:
        result = subprocess.run([
            python_exe, "-m", "pytest", "--version"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Pytest available")
            
            # Run actual tests
            print("\n🚀 Running tests...")
            test_result = subprocess.run([
                python_exe, "-m", "pytest", 
                "test_basic_imports.py", "-v"
            ], cwd=Path(__file__).parent, timeout=30)
            
            return test_result.returncode == 0
        else:
            print("⚠️ Pytest not available, trying direct execution")
            return run_direct_tests(python_exe)
            
    except Exception as e:
        print(f"❌ Pytest error: {e}")
        return run_direct_tests(python_exe)

def run_direct_tests(python_exe):
    """Run tests directly without pytest."""
    print("🔧 Running direct tests...")
    
    test_files = [
        "test_basic_imports.py",
        "test_core_structures.py"
    ]
    
    success_count = 0
    total_count = len(test_files)
    
    for test_file in test_files:
        test_path = Path(__file__).parent / test_file
        if test_path.exists():
            try:
                result = subprocess.run([
                    python_exe, str(test_path)
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    print(f"✅ {test_file}: PASSED")
                    success_count += 1
                else:
                    print(f"❌ {test_file}: FAILED")
                    print(f"   Error: {result.stderr}")
            except Exception as e:
                print(f"❌ {test_file}: ERROR - {e}")
        else:
            print(f"⚠️ {test_file}: NOT FOUND")
    
    print(f"\n📊 Results: {success_count}/{total_count} tests passed")
    return success_count == total_count

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
