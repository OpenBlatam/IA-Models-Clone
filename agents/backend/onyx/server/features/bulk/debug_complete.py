"""
Complete Debug Script for BUL API Tests
Identifies and helps fix all issues
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path

def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_issue(severity, message):
    """Print issue with severity."""
    icons = {
        "ERROR": "❌",
        "WARNING": "⚠️",
        "INFO": "ℹ️",
        "SUCCESS": "✅"
    }
    print(f"{icons.get(severity, '•')} {severity}: {message}")

def find_python_executables():
    """Find all Python executables on the system."""
    print_header("Finding Python Executables")
    
    python_paths = []
    
    # Check common locations
    common_paths = [
        "python",
        "python3",
        "py",
        shutil.which("python"),
        shutil.which("python3"),
        shutil.which("py"),
    ]
    
    # Windows specific paths
    if sys.platform == "win32":
        user_profile = os.environ.get("USERPROFILE", "")
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        program_files = os.environ.get("ProgramFiles", "")
        
        common_paths.extend([
            f"{local_appdata}\\Programs\\Python\\Python311\\python.exe",
            f"{local_appdata}\\Programs\\Python\\Python312\\python.exe",
            f"{local_appdata}\\Programs\\Python\\Python310\\python.exe",
            f"{program_files}\\Python311\\python.exe",
            f"{program_files}\\Python312\\python.exe",
            f"{program_files}\\Python310\\python.exe",
        ])
    
    for path in common_paths:
        if not path:
            continue
            
        try:
            if path in ["python", "python3", "py"]:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    version = result.stdout.decode().strip()
                    python_paths.append({
                        "path": path,
                        "version": version,
                        "type": "command"
                    })
                    print_issue("SUCCESS", f"Found: {path} - {version}")
            elif os.path.exists(path):
                try:
                    result = subprocess.run(
                        [path, "--version"],
                        capture_output=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        version = result.stdout.decode().strip()
                        python_paths.append({
                            "path": path,
                            "version": version,
                            "type": "file"
                        })
                        print_issue("SUCCESS", f"Found: {path} - {version}")
                except Exception as e:
                    print_issue("WARNING", f"{path} exists but cannot execute: {e}")
        except Exception:
            continue
    
    if not python_paths:
        print_issue("ERROR", "No working Python found!")
        return None
    
    return python_paths[0]  # Return first working Python

def check_current_python():
    """Check current Python installation."""
    print_header("Checking Current Python")
    
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Check if it's a venv
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Running in virtual environment")
        print(f"  Prefix: {sys.prefix}")
        print(f"  Base prefix: {sys.base_prefix}")
        
        # Check if venv is broken
        if not os.path.exists(sys.executable):
            print_issue("ERROR", f"Virtual environment points to non-existent Python: {sys.executable}")
            return False
    
    return True

def check_dependencies(python_cmd):
    """Check if required dependencies are installed."""
    print_header("Checking Dependencies")
    
    required = ["requests", "json", "time", "sys"]
    optional = ["colorama", "websockets"]
    
    missing_required = []
    missing_optional = []
    
    for module in required:
        try:
            result = subprocess.run(
                [python_cmd, "-c", f"import {module}"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                print_issue("SUCCESS", f"{module} installed")
            else:
                missing_required.append(module)
                print_issue("ERROR", f"{module} NOT installed")
        except Exception as e:
            missing_required.append(module)
            print_issue("ERROR", f"{module} - Error: {e}")
    
    for module in optional:
        try:
            result = subprocess.run(
                [python_cmd, "-c", f"import {module}"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                print_issue("SUCCESS", f"{module} installed (optional)")
            else:
                missing_optional.append(module)
                print_issue("WARNING", f"{module} NOT installed (optional)")
        except Exception:
            missing_optional.append(module)
            print_issue("WARNING", f"{module} NOT installed (optional)")
    
    return missing_required, missing_optional

def check_server(python_cmd):
    """Check if API server is running."""
    print_header("Checking Server Status")
    
    try:
        result = subprocess.run(
            [python_cmd, "-c",
             "import requests; r = requests.get('http://localhost:8000/api/health', timeout=2); print(r.status_code)"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            status_code = result.stdout.decode().strip()
            if status_code == "200":
                print_issue("SUCCESS", "Server is running on http://localhost:8000")
                return True
            else:
                print_issue("WARNING", f"Server responded with status {status_code}")
                return False
        else:
            print_issue("WARNING", "Cannot check server (requests may not be installed)")
            return None
    except subprocess.TimeoutExpired:
        print_issue("WARNING", "Server check timed out")
        return None
    except Exception as e:
        print_issue("WARNING", f"Server is NOT running: {e}")
        print_issue("INFO", "Start server with: python api_frontend_ready.py")
        return False

def verify_test_files():
    """Verify test files are syntactically correct."""
    print_header("Verifying Test Files")
    
    test_files = [
        "test_api_responses.py",
        "test_api_advanced.py",
        "test_security.py",
    ]
    
    all_ok = True
    for test_file in test_files:
        if not Path(test_file).exists():
            print_issue("ERROR", f"{test_file} not found")
            all_ok = False
            continue
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, test_file, 'exec')
            print_issue("SUCCESS", f"{test_file} - Syntax OK")
        except SyntaxError as e:
            print_issue("ERROR", f"{test_file} - Syntax error at line {e.lineno}: {e.msg}")
            all_ok = False
        except Exception as e:
            print_issue("ERROR", f"{test_file} - Error: {e}")
            all_ok = False
    
    return all_ok

def provide_solutions(python_info, missing_req, missing_opt, server_ok, tests_ok):
    """Provide solutions for identified issues."""
    print_header("Solutions & Recommendations")
    
    issues_found = False
    
    if not python_info:
        print_issue("ERROR", "Python not found on system")
        print("\nSOLUTION:")
        print("  1. Install Python from: https://www.python.org/downloads/")
        print("  2. Make sure to check 'Add Python to PATH' during installation")
        print("  3. Restart terminal after installation")
        issues_found = True
    elif python_info["type"] == "command":
        print_issue("SUCCESS", f"Using Python: {python_info['path']}")
    else:
        print_issue("INFO", f"Using Python: {python_info['path']}")
    
    if missing_req:
        print_issue("ERROR", f"Missing required modules: {', '.join(missing_req)}")
        print("\nSOLUTION:")
        python_cmd = python_info["path"] if python_info else "python"
        print(f"  {python_cmd} -m pip install {' '.join(missing_req)}")
        print(f"  Or: {python_cmd} -m pip install -r requirements.txt")
        issues_found = True
    
    if missing_opt:
        print_issue("WARNING", f"Missing optional modules: {', '.join(missing_opt)}")
        print("\nOPTIONAL (for enhanced features):")
        python_cmd = python_info["path"] if python_info else "python"
        print(f"  {python_cmd} -m pip install {' '.join(missing_opt)}")
    
    if server_ok is False:
        print_issue("WARNING", "Server is not running")
        print("\nSOLUTION:")
        python_cmd = python_info["path"] if python_info else "python"
        print(f"  {python_cmd} api_frontend_ready.py")
        print("  Wait for 'Application startup complete' message")
        issues_found = True
    
    if not tests_ok:
        print_issue("ERROR", "Some test files have syntax errors")
        print("\nSOLUTION:")
        print("  Fix the syntax errors in the test files listed above")
        issues_found = True
    
    if not issues_found:
        print_issue("SUCCESS", "No issues found! Ready to run tests.")
        print("\nNext steps:")
        python_cmd = python_info["path"] if python_info else "python"
        print(f"  1. Start server: {python_cmd} api_frontend_ready.py")
        print(f"  2. Run tests: {python_cmd} test_api_responses.py")
        print(f"  3. Or use: run_all_tests.bat")
    
    return issues_found

def main():
    """Main debug function."""
    print("=" * 70)
    print("  BUL API - Complete Debug Report")
    print("=" * 70)
    
    # Check current Python
    current_ok = check_current_python()
    
    # Find working Python
    python_info = find_python_executables()
    
    if not python_info:
        print("\n" + "=" * 70)
        print("  CRITICAL: No working Python found!")
        print("=" * 70)
        provide_solutions(None, [], [], None, True)
        return 1
    
    python_cmd = python_info["path"]
    
    # Check dependencies
    missing_req, missing_opt = check_dependencies(python_cmd)
    
    # Check server
    server_ok = check_server(python_cmd)
    
    # Verify tests
    tests_ok = verify_test_files()
    
    # Provide solutions
    issues_found = provide_solutions(
        python_info,
        missing_req,
        missing_opt,
        server_ok,
        tests_ok
    )
    
    print("\n" + "=" * 70)
    print("  Debug Complete")
    print("=" * 70)
    
    return 1 if issues_found else 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Debug interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)









