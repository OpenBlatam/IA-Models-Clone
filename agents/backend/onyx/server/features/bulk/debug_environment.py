"""
Comprehensive Environment Debug Script
Checks Python installation, dependencies, server status, and test readiness
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List, Dict

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def find_python_executables() -> List[str]:
    """Find all possible Python executables"""
    print_section("Finding Python Installation")
    
    possible_paths = []
    found_pythons = []
    
    # Check common commands
    commands = ["python", "python3", "py", "python.exe", "python3.exe"]
    for cmd in commands:
        path = shutil.which(cmd)
        if path:
            possible_paths.append(path)
    
    # Windows specific paths
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        program_files = os.environ.get("ProgramFiles", "")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", "")
        
        # Common installation locations
        search_paths = [
            f"{local_appdata}\\Programs\\Python",
            f"{program_files}\\Python*",
            f"{program_files_x86}\\Python*",
            "C:\\Python*",
            "C:\\Program Files\\Python*",
        ]
        
        # Check for venv in project
        project_root = Path(__file__).parent.parent.parent.parent.parent
        venv_paths = [
            project_root / "venv_ultra_advanced" / "Scripts" / "python.exe",
            project_root / "venv_ultra_quality" / "Scripts" / "python.exe",
            project_root / "venv" / "Scripts" / "python.exe",
        ]
        
        for venv_path in venv_paths:
            if venv_path.exists():
                possible_paths.append(str(venv_path))
                print(f"  ✓ Found venv: {venv_path}")
    
    # Test each path
    for python_path in set(possible_paths):
        if not python_path or not os.path.exists(python_path):
            continue
        try:
            result = subprocess.run(
                [python_path, "--version"],
                capture_output=True,
                timeout=5,
                text=True
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                found_pythons.append((python_path, version))
                print(f"  ✓ {python_path}")
                print(f"    Version: {version}")
        except Exception as e:
            continue
    
    if not found_pythons:
        print("  ✗ No Python installations found!")
        print("\n  To install Python:")
        print("  1. Download from: https://www.python.org/downloads/")
        print("  2. During installation, check 'Add Python to PATH'")
        print("  3. Restart your terminal after installation")
    
    return found_pythons

def check_dependencies(python_path: str) -> Dict[str, bool]:
    """Check if required dependencies are installed"""
    print_section("Checking Dependencies")
    
    required_modules = {
        "requests": "HTTP library for API calls",
        "colorama": "Terminal colors (optional)",
        "fastapi": "Web framework",
        "uvicorn": "ASGI server",
        "websockets": "WebSocket support (optional)",
        "pydantic": "Data validation",
    }
    
    results = {}
    missing = []
    
    for module, description in required_modules.items():
        try:
            result = subprocess.run(
                [python_path, "-c", f"import {module}"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                # Try to get version
                try:
                    version_result = subprocess.run(
                        [python_path, "-c", f"import {module}; print({module}.__version__)"],
                        capture_output=True,
                        timeout=5,
                        text=True
                    )
                    version = version_result.stdout.strip() if version_result.returncode == 0 else "unknown"
                    print(f"  ✓ {module:15} - {description}")
                    print(f"    Version: {version}")
                except:
                    print(f"  ✓ {module:15} - {description}")
                results[module] = True
            else:
                print(f"  ✗ {module:15} - NOT INSTALLED")
                results[module] = False
                missing.append(module)
        except Exception as e:
            print(f"  ✗ {module:15} - Error: {e}")
            results[module] = False
            missing.append(module)
    
    if missing:
        print(f"\n  ⚠ Missing dependencies: {', '.join(missing)}")
        print(f"\n  Install with:")
        print(f"    {python_path} -m pip install {' '.join(missing)}")
        print(f"\n  Or install all requirements:")
        print(f"    {python_path} -m pip install -r requirements.txt")
    
    return results

def check_server_status() -> bool:
    """Check if the API server is running"""
    print_section("Checking Server Status")
    
    try:
        import requests
    except ImportError:
        print("  ⚠ Cannot check server (requests not installed)")
        return False
    
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("  ✓ Server is RUNNING on http://localhost:8000")
            print(f"    Status: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"  ⚠ Server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("  ✗ Server is NOT running")
        print("\n  To start the server:")
        print("    cd C:\\blatam-academy\\agents\\backend\\onyx\\server\\features\\bulk")
        print("    python api_frontend_ready.py")
        return False
    except Exception as e:
        print(f"  ✗ Error checking server: {e}")
        return False

def check_test_files() -> Dict[str, bool]:
    """Check if test files exist and are readable"""
    print_section("Checking Test Files")
    
    test_files = [
        "test_api_responses.py",
        "test_api_advanced.py",
        "test_security.py",
        "run_tests_debug.py",
        "api_frontend_ready.py",
    ]
    
    results = {}
    for test_file in test_files:
        file_path = Path(__file__).parent / test_file
        if file_path.exists():
            try:
                # Try to read first line to verify it's readable
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                print(f"  ✓ {test_file:25} - Found and readable")
                results[test_file] = True
            except Exception as e:
                print(f"  ✗ {test_file:25} - Error reading: {e}")
                results[test_file] = False
        else:
            print(f"  ✗ {test_file:25} - NOT FOUND")
            results[test_file] = False
    
    return results

def check_requirements_file() -> bool:
    """Check if requirements.txt exists"""
    print_section("Checking Requirements File")
    
    req_file = Path(__file__).parent / "requirements.txt"
    if req_file.exists():
        print(f"  ✓ requirements.txt found")
        try:
            with open(req_file, 'r') as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
            print(f"    Contains {len(lines)} dependencies")
            return True
        except Exception as e:
            print(f"  ✗ Error reading requirements.txt: {e}")
            return False
    else:
        print(f"  ✗ requirements.txt NOT FOUND")
        return False

def generate_fix_commands(python_path: Optional[str], missing_deps: List[str]) -> str:
    """Generate commands to fix issues"""
    print_section("Recommended Fix Commands")
    
    if not python_path:
        print("  1. Install Python first:")
        print("     Download from: https://www.python.org/downloads/")
        print("     Make sure to check 'Add Python to PATH'")
        return
    
    commands = []
    
    # Change to directory
    bulk_dir = Path(__file__).parent
    commands.append(f"cd {bulk_dir}")
    
    # Install dependencies
    if missing_deps:
        commands.append(f"{python_path} -m pip install {' '.join(missing_deps)}")
    else:
        commands.append(f"{python_path} -m pip install -r requirements.txt")
    
    # Start server
    commands.append(f"{python_path} api_frontend_ready.py")
    
    print("\n  Run these commands in order:")
    for i, cmd in enumerate(commands, 1):
        print(f"\n  {i}. {cmd}")
    
    return "\n".join(commands)

def main():
    """Main debug function"""
    print("="*70)
    print("  BUL API - Environment Debug Tool")
    print("="*70)
    
    # Find Python
    pythons = find_python_executables()
    python_path = pythons[0][0] if pythons else None
    
    if not python_path:
        print("\n" + "="*70)
        print("  CRITICAL: Python not found!")
        print("="*70)
        print("\nPlease install Python first:")
        print("  1. Download from: https://www.python.org/downloads/")
        print("  2. During installation, check 'Add Python to PATH'")
        print("  3. Restart your terminal")
        return 1
    
    # Check dependencies
    dep_results = check_dependencies(python_path)
    missing = [mod for mod, installed in dep_results.items() if not installed]
    
    # Check server
    server_running = check_server_status()
    
    # Check test files
    test_results = check_test_files()
    
    # Check requirements
    req_exists = check_requirements_file()
    
    # Summary
    print_section("Summary")
    
    print(f"\n  Python: {'✓ Found' if python_path else '✗ Not found'}")
    print(f"  Dependencies: {len([v for v in dep_results.values() if v])}/{len(dep_results)} installed")
    print(f"  Server: {'✓ Running' if server_running else '✗ Not running'}")
    print(f"  Test Files: {len([v for v in test_results.values() if v])}/{len(test_results)} found")
    print(f"  Requirements: {'✓ Found' if req_exists else '✗ Not found'}")
    
    # Generate fix commands
    if missing or not server_running:
        generate_fix_commands(python_path, missing)
    
    # Next steps
    print_section("Next Steps")
    
    if python_path and not missing:
        print("\n  ✓ Environment looks good!")
        print("\n  To run tests:")
        print(f"    1. Start server: {python_path} api_frontend_ready.py")
        print(f"    2. Run tests: {python_path} run_tests_debug.py")
    elif python_path:
        print("\n  ⚠ Install missing dependencies first")
        print(f"    {python_path} -m pip install {' '.join(missing)}")
    else:
        print("\n  ✗ Install Python first")
    
    return 0

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









