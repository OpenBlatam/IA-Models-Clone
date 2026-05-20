"""
Debug and Test Runner Script
Finds Python, checks dependencies, verifies server, and runs tests
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path

def find_python():
    """Find available Python executable"""
    # Common Python installation paths
    possible_paths = [
        "python",
        "python3",
        "py",
        shutil.which("python"),
        shutil.which("python3"),
        shutil.which("py"),
    ]
    
    # Windows common paths
    if sys.platform == "win32":
        user_profile = os.environ.get("USERPROFILE", "")
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        possible_paths.extend([
            f"{local_appdata}\\Programs\\Python\\Python311\\python.exe",
            f"{local_appdata}\\Programs\\Python\\Python312\\python.exe",
            f"{local_appdata}\\Programs\\Python\\Python310\\python.exe",
            f"C:\\Python311\\python.exe",
            f"C:\\Python312\\python.exe",
            f"C:\\Python310\\python.exe",
        ])
        
        # Check venv in project
        project_root = Path(__file__).parent.parent.parent.parent.parent
        venv_paths = [
            project_root / "venv_ultra_advanced" / "Scripts" / "python.exe",
            project_root / "venv_ultra_quality" / "Scripts" / "python.exe",
        ]
        for venv_path in venv_paths:
            if venv_path.exists():
                possible_paths.append(str(venv_path))
    
    for python_path in possible_paths:
        if not python_path:
            continue
        try:
            result = subprocess.run(
                [python_path, "--version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✓ Found Python: {python_path}")
                print(f"  Version: {result.stdout.decode().strip()}")
                return python_path
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
    
    return None

def check_dependencies(python_path):
    """Check if required dependencies are installed"""
    print("\n" + "="*60)
    print("Checking Dependencies...")
    print("="*60)
    
    required_modules = [
        "requests",
        "colorama",
        "fastapi",
        "uvicorn",
    ]
    
    missing = []
    for module in required_modules:
        try:
            result = subprocess.run(
                [python_path, "-c", f"import {module}"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✓ {module} installed")
            else:
                print(f"✗ {module} NOT installed")
                missing.append(module)
        except Exception as e:
            print(f"✗ {module} - Error checking: {e}")
            missing.append(module)
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print(f"Install with: {python_path} -m pip install {' '.join(missing)}")
        return False
    return True

def check_server():
    """Check if API server is running"""
    print("\n" + "="*60)
    print("Checking Server Status...")
    print("="*60)
    
    try:
        import requests
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code == 200:
            print("✓ Server is running on http://localhost:8000")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"⚠ Server responded with status {response.status_code}")
            return False
    except ImportError:
        print("⚠ Cannot check server (requests not installed)")
        return None
    except Exception as e:
        print(f"✗ Server is NOT running: {e}")
        print("  Start server with: python api_frontend_ready.py")
        return False

def run_tests(python_path):
    """Run test scripts"""
    print("\n" + "="*60)
    print("Running Tests...")
    print("="*60)
    
    test_files = [
        "test_api_responses.py",
        "test_api_advanced.py",
        "test_security.py",
    ]
    
    results = {}
    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"⚠ {test_file} not found, skipping...")
            continue
        
        print(f"\n--- Running {test_file} ---")
        try:
            result = subprocess.run(
                [python_path, test_file],
                cwd=Path(__file__).parent,
                timeout=300
            )
            results[test_file] = result.returncode == 0
            if result.returncode == 0:
                print(f"✓ {test_file} completed successfully")
            else:
                print(f"✗ {test_file} failed (exit code: {result.returncode})")
        except subprocess.TimeoutExpired:
            print(f"✗ {test_file} timed out")
            results[test_file] = False
        except Exception as e:
            print(f"✗ {test_file} error: {e}")
            results[test_file] = False
    
    return results

def main():
    print("="*60)
    print("BUL API - Test Runner & Debugger")
    print("="*60)
    
    # Find Python
    python_path = find_python()
    if not python_path:
        print("\n✗ ERROR: Python not found!")
        print("\nPlease install Python or ensure it's in your PATH")
        print("Download from: https://www.python.org/downloads/")
        return 1
    
    # Check dependencies
    deps_ok = check_dependencies(python_path)
    if not deps_ok:
        print("\n⚠ Some dependencies are missing. Install them first.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    # Check server
    server_ok = check_server()
    if server_ok is False:
        print("\n⚠ Server is not running. Tests may fail.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("\nStart server with:")
            print(f"  {python_path} api_frontend_ready.py")
            return 1
    
    # Run tests
    test_results = run_tests(python_path)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(1 for v in test_results.values() if v)
    total = len(test_results)
    print(f"\nTests passed: {passed}/{total}")
    
    for test_file, passed in test_results.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {test_file}")
    
    if all(test_results.values()):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed. Check output above for details.")
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









