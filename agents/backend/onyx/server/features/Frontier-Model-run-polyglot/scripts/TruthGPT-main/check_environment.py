"""
Environment Check Script for TruthGPT
Checks Python, dependencies, and test readiness without requiring imports
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path

def find_python():
    """Find working Python executable"""
    print("\n" + "="*70)
    print("  Finding Python Installation")
    print("="*70)
    
    candidates = []
    
    # Check PATH commands
    for cmd in ["python", "python3", "py"]:
        path = shutil.which(cmd)
        if path:
            candidates.append(path)
    
    # Windows specific paths
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        candidates.extend([
            f"{local_appdata}\\Programs\\Python\\Python311\\python.exe",
            f"{local_appdata}\\Programs\\Python\\Python312\\python.exe",
            f"{local_appdata}\\Programs\\Python\\Python310\\python.exe",
            "C:\\Python311\\python.exe",
            "C:\\Python312\\python.exe",
            "C:\\Python310\\python.exe",
        ])
        
        # Check venv
        project_root = Path(__file__).parent.parent.parent.parent.parent.parent
        venv_paths = [
            project_root / "venv" / "Scripts" / "python.exe",
            project_root / "venv_ultra_advanced" / "Scripts" / "python.exe",
        ]
        for venv_path in venv_paths:
            if venv_path.exists():
                candidates.append(str(venv_path))
    
    # Test each candidate
    for python_path in candidates:
        if not python_path or not Path(python_path).exists():
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
                print(f"  ✓ Found: {python_path}")
                print(f"    Version: {version}")
                return python_path
        except:
            continue
    
    print("  ✗ No working Python found")
    return None

def check_files():
    """Check if test files exist"""
    print("\n" + "="*70)
    print("  Checking Test Files")
    print("="*70)
    
    project_root = Path(__file__).parent
    tests_dir = project_root / "tests"
    
    test_files = [
        "tests/test_core.py",
        "tests/test_optimization.py",
        "tests/test_models.py",
        "tests/test_training.py",
        "tests/test_inference.py",
        "tests/test_monitoring.py",
        "tests/test_integration.py",
        "run_unified_tests.py",
        "debug_tests.py",
    ]
    
    all_exist = True
    for test_file in test_files:
        file_path = project_root / test_file
        if file_path.exists():
            print(f"  ✓ {test_file}")
        else:
            print(f"  ✗ {test_file} - NOT FOUND")
            all_exist = False
    
    return all_exist

def check_core_module(python_path):
    """Check if core module exists"""
    print("\n" + "="*70)
    print("  Checking Core Module")
    print("="*70)
    
    project_root = Path(__file__).parent
    core_dir = project_root / "core"
    
    if not core_dir.exists():
        print("  ✗ core/ directory not found")
        return False
    
    required_files = [
        "core/__init__.py",
        "core/optimization.py",
        "core/models.py",
        "core/training.py",
        "core/inference.py",
        "core/monitoring.py",
        "core/architectures.py",
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - NOT FOUND")
            all_exist = False
    
    # Try to check imports if Python is available
    if python_path:
        try:
            result = subprocess.run(
                [python_path, "-c", "import sys; sys.path.insert(0, '.'); from core import OptimizationEngine; print('Core imports OK')"],
                cwd=project_root,
                capture_output=True,
                timeout=10,
                text=True
            )
            if result.returncode == 0:
                print("  ✓ Core module imports successfully")
            else:
                print(f"  ⚠ Core module import test failed: {result.stderr}")
        except Exception as e:
            print(f"  ⚠ Could not test imports: {e}")
    
    return all_exist

def check_dependencies(python_path):
    """Check dependencies"""
    print("\n" + "="*70)
    print("  Checking Dependencies")
    print("="*70)
    
    if not python_path:
        print("  ⚠ Cannot check dependencies - Python not found")
        return {}
    
    deps = {
        "torch": "PyTorch - Required",
        "numpy": "NumPy - Required",
        "psutil": "PSUtil - Optional",
    }
    
    results = {}
    for module, desc in deps.items():
        try:
            result = subprocess.run(
                [python_path, "-c", f"import {module}; print({module}.__version__)"],
                capture_output=True,
                timeout=5,
                text=True
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"  ✓ {module:15} - {desc:20} - Version: {version}")
                results[module] = True
            else:
                optional = "Optional" in desc
                status = "⚠" if optional else "✗"
                print(f"  {status} {module:15} - {desc:20} - NOT INSTALLED")
                results[module] = optional
        except Exception as e:
            print(f"  ✗ {module:15} - Error: {e}")
            results[module] = False
    
    return results

def main():
    """Main check function"""
    print("="*70)
    print("  TruthGPT Environment Check")
    print("="*70)
    
    # Find Python
    python_path = find_python()
    
    # Check files
    files_ok = check_files()
    
    # Check core module
    core_ok = check_core_module(python_path)
    
    # Check dependencies
    deps_results = check_dependencies(python_path)
    
    # Summary
    print("\n" + "="*70)
    print("  Summary")
    print("="*70)
    
    print(f"\n  Python:        {'✓ Found' if python_path else '✗ Not found'}")
    print(f"  Test Files:    {'✓ All present' if files_ok else '✗ Some missing'}")
    print(f"  Core Module:   {'✓ OK' if core_ok else '✗ Issues found'}")
    
    if deps_results:
        installed = sum(1 for v in deps_results.values() if v)
        total = len(deps_results)
        print(f"  Dependencies:  {installed}/{total} installed")
        
        missing = [k for k, v in deps_results.items() if not v]
        if missing:
            print(f"\n  ⚠ Missing: {', '.join(missing)}")
            if python_path:
                print(f"\n  Install with:")
                print(f"    {python_path} -m pip install {' '.join(missing)}")
    
    # Recommendations
    print("\n" + "="*70)
    print("  Recommendations")
    print("="*70)
    
    if not python_path:
        print("\n  1. Install Python from: https://www.python.org/downloads/")
        print("  2. Check 'Add Python to PATH' during installation")
        print("  3. Restart terminal after installation")
    elif not all(deps_results.values()):
        print("\n  1. Install missing dependencies:")
        missing = [k for k, v in deps_results.items() if not v]
        print(f"     {python_path} -m pip install {' '.join(missing)}")
    elif files_ok and core_ok:
        print("\n  ✓ Environment looks good!")
        print(f"\n  Run tests with:")
        print(f"    {python_path} run_unified_tests.py")
    
    return 0 if (python_path and files_ok and core_ok) else 1

if __name__ == "__main__":
    import os
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

