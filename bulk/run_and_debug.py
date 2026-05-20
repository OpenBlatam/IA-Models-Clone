"""
Run tests and debug any issues
This script runs all tests and provides detailed debugging information
"""

import sys
import os
import subprocess
import traceback
from pathlib import Path

def find_python():
    """Find Python executable."""
    for cmd in ['python', 'python3', 'py']:
        try:
            result = subprocess.run([cmd, '--version'], capture_output=True, timeout=5)
            if result.returncode == 0:
                return cmd
        except:
            continue
    return None

def check_server(python_cmd):
    """Check if server is running."""
    try:
        result = subprocess.run(
            [python_cmd, '-c', 
             "import requests; r = requests.get('http://localhost:8000/api/health', timeout=2); exit(0 if r.status_code == 200 else 1)"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False

def run_test(python_cmd, test_file):
    """Run a single test file."""
    print(f"\n{'='*70}")
    print(f"Running: {test_file}")
    print(f"{'='*70}")
    
    if not Path(test_file).exists():
        print(f"❌ File not found: {test_file}")
        return False
    
    try:
        result = subprocess.run(
            [python_cmd, test_file],
            cwd=Path(__file__).parent,
            timeout=300,
            capture_output=False
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"⏱️  Timeout running {test_file}")
        return False
    except Exception as e:
        print(f"❌ Error running {test_file}: {e}")
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("=" * 70)
    print("BUL API - Test Runner & Debugger")
    print("=" * 70)
    print()
    
    # Find Python
    print("[1/4] Finding Python...")
    python_cmd = find_python()
    if not python_cmd:
        print("❌ Python not found!")
        print("\nPlease install Python from: https://www.python.org/downloads/")
        print("Or ensure Python is in your PATH")
        return 1
    
    print(f"✅ Found Python: {python_cmd}")
    try:
        version_result = subprocess.run([python_cmd, '--version'], capture_output=True, timeout=5)
        print(f"   Version: {version_result.stdout.decode().strip()}")
    except:
        pass
    
    # Check dependencies
    print("\n[2/4] Checking dependencies...")
    try:
        result = subprocess.run(
            [python_cmd, '-c', 'import requests, json, time, sys'],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ Core dependencies available")
        else:
            print("❌ Missing core dependencies")
            print("   Install with: pip install requests")
            return 1
    except Exception as e:
        print(f"❌ Error checking dependencies: {e}")
        return 1
    
    # Check server
    print("\n[3/4] Checking server...")
    server_running = check_server(python_cmd)
    if server_running:
        print("✅ Server is running on http://localhost:8000")
    else:
        print("⚠️  Server is NOT running")
        print("   Start server with: python api_frontend_ready.py")
        print("   Continuing tests anyway...")
    
    # Run tests
    print("\n[4/4] Running tests...")
    test_files = [
        "test_api_responses.py",
        "test_api_advanced.py",
        "test_security.py",
    ]
    
    results = {}
    for test_file in test_files:
        success = run_test(python_cmd, test_file)
        results[test_file] = success
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_file, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_file}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)









