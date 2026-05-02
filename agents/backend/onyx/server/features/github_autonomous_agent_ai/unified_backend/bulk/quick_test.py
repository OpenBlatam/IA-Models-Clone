"""
Quick test script to verify environment and server status
Run this first to diagnose issues
"""

import sys
import os

def test_python():
    """Test if Python is working"""
    print("="*60)
    print("Python Environment Test")
    print("="*60)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path[:3]}...")
    return True

def test_imports():
    """Test if required modules can be imported"""
    print("\n" + "="*60)
    print("Testing Required Imports")
    print("="*60)
    
    modules = {
        "requests": "HTTP requests",
        "colorama": "Terminal colors (optional)",
        "json": "JSON handling",
        "time": "Time utilities",
    }
    
    missing = []
    for module, desc in modules.items():
        try:
            __import__(module)
            print(f"✓ {module:15} - {desc}")
        except ImportError:
            print(f"✗ {module:15} - MISSING ({desc})")
            missing.append(module)
    
    if missing:
        print(f"\n⚠ Missing modules: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True

def test_server():
    """Test if server is accessible"""
    print("\n" + "="*60)
    print("Testing Server Connection")
    print("="*60)
    
    try:
        import requests
        url = "http://localhost:8000/api/health"
        print(f"Checking: {url}")
        
        response = requests.get(url, timeout=5)
        print(f"✓ Server responded: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Response: {data}")
            return True
        else:
            print(f"⚠ Unexpected status: {response.status_code}")
            return False
            
    except ImportError:
        print("✗ Cannot test (requests not installed)")
        return None
    except requests.exceptions.ConnectionError:
        print("✗ Server is NOT running")
        print("\nTo start server:")
        print("  python api_frontend_ready.py")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("BUL API - Quick Environment Test")
    print("="*60 + "\n")
    
    # Test Python
    test_python()
    
    # Test imports
    imports_ok = test_imports()
    
    # Test server
    server_ok = test_server()
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Python: ✓ Working")
    print(f"Imports: {'✓ OK' if imports_ok else '✗ Missing modules'}")
    print(f"Server: {'✓ Running' if server_ok else '✗ Not running' if server_ok is False else '? Cannot test'}")
    
    if not imports_ok:
        print("\n⚠ Install missing modules before running tests")
        return 1
    
    if server_ok is False:
        print("\n⚠ Start the server before running tests")
        return 1
    
    if server_ok:
        print("\n✓ Environment looks good! You can run tests now:")
        print("  python test_api_responses.py")
        print("  python test_api_advanced.py")
        print("  python test_security.py")
        return 0
    else:
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
